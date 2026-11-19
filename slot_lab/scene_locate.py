"""
scene_locate.py - ask the game itself where every button and object is.

Two layers are read:
  html  the HUD drawn as normal HTML over the canvas (Spin, Autoplay, Turbo, Menu,
        bet +/-, balance/win/bet values). Read with getBoundingClientRect(); each
        one also gets a CSS selector (e.g. button[aria-label="Spin"]).
  pixi  the canvas scene (reels, symbols, characters).

PixiJS 8 calls globalThis.__PIXI_APP_INIT__(app) when the game starts (it's the
hook the Pixi DevTools extension uses). We define that function before the
game's code runs, keep the app, then walk app.stage and read getBounds() of
every visible sprite / clickable object. No image matching needed.

  python scene_locate.py "<game url>"                 press Enter to dump, q + Enter to quit
  python scene_locate.py "<game url>" --every 5       dump automatically every 5 s
  python scene_locate.py "<game url>" --only spin,bet,autoplay
  python scene_locate.py "<game url>" --pixi           include canvas objects too

Each dump writes scene_elements.json with, per element:
  name         texture alias/file (e.g. spin_icon) or Pixi label / text
  page_x/y     centre in page CSS pixels  (use with Playwright page.mouse.click)
  screen_x/y   centre in physical screen pixels (use with an external RPA tool)
  box          [x, y, w, h] in screen pixels, interactive, cursor, path in the tree
"""
import argparse
import json
import os
import sys
import threading

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))

HOOK_JS = r"""
(() => {
  const keep = (app) => { window.__PIXI_APP__ = app; };
  const prevApp = globalThis.__PIXI_APP_INIT__;
  globalThis.__PIXI_APP_INIT__ = (app, v) => { keep(app); window.__PIXI_VERSION__ = v; prevApp && prevApp(app, v); };
  // root container each renderer actually draws (one entry per canvas/renderer);
  // games don't always draw app.stage, so this is the reliable source
  window.__PIXI_ROOTS__ = new Map();
  const remember = (r, root) => {
    const obj = root && root.container ? root.container : root;  // v7/v8 accept {container}
    if (obj && obj.children) window.__PIXI_ROOTS__.set(r, obj);
  };
  const prevR = globalThis.__PIXI_RENDERER_INIT__;
  globalThis.__PIXI_RENDERER_INIT__ = (r, v) => {
    window.__PIXI_RENDERER__ = r;
    const orig = r.render;
    r.render = function (root, ...rest) { remember(this, root); return orig.call(this, root, ...rest); };
    prevR && prevR(r, v);
  };

  // PixiJS 4-7 have no init hook, but expose window.PIXI: wrap render() on the classes.
  const patch = () => {
    const P = window.PIXI;
    if (!P) return false;
    let done = false;
    for (const k of ["Renderer", "WebGLRenderer", "CanvasRenderer"]) {
      const C = P[k];
      if (!C || !C.prototype || !C.prototype.render || C.prototype.render.__hooked) continue;
      const orig = C.prototype.render;
      C.prototype.render = function (root, ...rest) {
        remember(this, root);
        return orig.call(this, root, ...rest);
      };
      C.prototype.render.__hooked = true;
      window.__PIXI_VERSION__ = window.__PIXI_VERSION__ || P.VERSION;
      done = true;
    }
    return done;
  };
  const timer = setInterval(() => { if (patch()) clearInterval(timer); }, 50);
  setTimeout(() => clearInterval(timer), 120000);
})();
"""

TOP_JS = r"""
() => {
  const border = (window.outerWidth - window.innerWidth) / 2;
  return { dpr: window.devicePixelRatio || 1, screenX: window.screenX, screenY: window.screenY,
           border, toolbar: window.outerHeight - window.innerHeight - border };
}
"""

DUMP_JS = r"""
(top) => {
  // frame CSS px -> top page CSS px (iframe offset) -> physical screen px
  // (window position + browser toolbar height, times DPR). `top` is measured in the top page.
  const dpr = top.dpr, ox = top.ox, oy = top.oy;
  const toScreen = (x, y) => [Math.round((top.screenX + top.border + ox + x) * dpr),
                              Math.round((top.screenY + top.toolbar + oy + y) * dpr)];

  // 1) HTML controls layered over the canvas (this game's HUD: Spin, Autoplay, bet +/-...)
  const out = [];
  const visible = (el, r) => {
    if (r.width < 4 || r.height < 4) return false;
    const cs = getComputedStyle(el);
    return cs.visibility !== "hidden" && cs.display !== "none" && +cs.opacity !== 0;
  };
  const cssPath = (el) => {
    if (el.id) return "#" + el.id;
    const aria = el.getAttribute("aria-label");
    if (aria) return `${el.tagName.toLowerCase()}[aria-label="${aria}"]`;
    // drop state classes (is-*) and framework hashes (svelte-1psueiy) that change between builds
    const cls = [...el.classList].filter(c => !/^(is-|svelte-)/.test(c) && /^[\w-]+$/.test(c)).map(c => "." + c).join("");
    return el.tagName.toLowerCase() + cls;
  };
  const className = (el) => [...el.classList].find(c => !/^(is-|svelte-)/.test(c)) || el.tagName.toLowerCase();
  const iconOf = (el) => {  // file name of an element's <img> / CSS background image
    const cs = getComputedStyle(el);
    const css = [cs.backgroundImage, cs.maskImage, cs.webkitMaskImage].find(v => v && v.includes("url("));
    const src = el.tagName === "IMG" ? el.currentSrc || el.src : css ? css.match(/url\("?([^")]+)"?\)/)[1] : "";
    return src && !src.startsWith("data:") ? src.split(/[?#]/)[0].split("/").pop() : "";
  };
  const domSel = "button, [role=button], a[href], input, select, [class*=__value], [class*=__label]";
  for (const el of document.querySelectorAll(domSel)) {
    const r = el.getBoundingClientRect();
    if (!visible(el, r)) continue;
    const text = (el.innerText || el.value || "").trim().replace(/\s+/g, " ");
    const [scx, scy] = toScreen(r.left + r.width / 2, r.top + r.height / 2);
    const [bx, by] = toScreen(r.left, r.top);
    const isControl = el.matches("button, [role=button], a[href], input, select");
    out.push({
      layer: "html", name: el.getAttribute("aria-label") || text.slice(0, 30) || cssPath(el),
      selector: cssPath(el), label: el.getAttribute("aria-label") || "", text: text.slice(0, 60),
      icon: [el, ...el.querySelectorAll("*")].map(iconOf).find(Boolean) || "",
      interactive: isControl && !el.disabled, disabled: !!el.disabled,
      classes: el.className && el.className.baseVal === undefined ? String(el.className) : "",
      page_x: Math.round(ox + r.left + r.width / 2), page_y: Math.round(oy + r.top + r.height / 2),
      screen_x: scx, screen_y: scy, box: [bx, by, Math.round(r.width * dpr), Math.round(r.height * dpr)],
      on_screen: r.right > 0 && r.bottom > 0 && r.left < innerWidth && r.top < innerHeight, path: cssPath(el),
    });
  }

  // unique CSS selector: the short form when it is unique, else a nth-of-type chain
  const uniqueSel = (el) => {
    const short = cssPath(el);
    try { if (document.querySelectorAll(short).length === 1) return short; } catch (e) {}
    const parts = [];
    for (let n = el; n && n.nodeType === 1 && n !== document.body; n = n.parentElement) {
      if (n.id) { parts.unshift("#" + CSS.escape(n.id)); break; }
      const same = [...(n.parentElement ? n.parentElement.children : [])].filter(s => s.tagName === n.tagName);
      parts.unshift(n.tagName.toLowerCase() + (same.length > 1 ? `:nth-of-type(${same.indexOf(n) + 1})` : ""));
    }
    return parts.join(" > ");
  };
  const assetUrl = (el) => {
    const cs = getComputedStyle(el);
    const css = [cs.backgroundImage, cs.maskImage, cs.webkitMaskImage].find(v => v && v.includes("url("));
    const src = el.tagName === "IMG" ? el.currentSrc || el.src : css ? css.match(/url\("?([^")]+)"?\)/)[1] : "";
    return src && !src.startsWith("data:") ? src : "";
  };
  const box = (r) => {
    const [scx, scy] = toScreen(r.left + r.width / 2, r.top + r.height / 2);
    const [bx, by] = toScreen(r.left, r.top);
    return { page_x: Math.round(ox + r.left + r.width / 2), page_y: Math.round(oy + r.top + r.height / 2),
             screen_x: scx, screen_y: scy, box: [bx, by, Math.round(r.width * dpr), Math.round(r.height * dpr)],
             on_screen: r.right > 0 && r.bottom > 0 && r.left < innerWidth && r.top < innerHeight };
  };
  const pointer = (el) => getComputedStyle(el).cursor === "pointer";

  // 1b) every other visible element that is an asset, a clickable control, or a text readout
  for (const el of document.querySelectorAll("body *")) {
    const r = el.getBoundingClientRect();
    if (el.tagName === "CANVAS" || !visible(el, r)) continue;
    const src = assetUrl(el);
    const file = src ? src.split(/[?#]/)[0].split("/").pop() : "";
    // top-most element of a clickable region (div/svg controls that aren't <button>s)
    const clickable = pointer(el) && !(el.parentElement && pointer(el.parentElement)) && !el.matches(domSel);
    const own = [...el.childNodes].filter(n => n.nodeType === 3).map(n => n.textContent.trim()).join(" ").trim();
    const readout = own && own.length <= 40 && !el.matches(domSel) && !el.closest("button");
    if (!src && !clickable && !readout) continue;
    // a control's icon: its own image, or the first image inside it
    let icon = file;
    if (clickable && !icon) for (const d of el.querySelectorAll("*")) { const u = assetUrl(d); if (u) { icon = u.split(/[?#]/)[0].split("/").pop(); break; } }
    const text = (el.innerText || "").trim().replace(/\s+/g, " ");
    const label = el.getAttribute("aria-label") || el.getAttribute("title") || "";
    out.push({
      layer: "html",
      name: clickable ? (label || text.slice(0, 30) || className(el))
                      : file ? "asset " + file : "text " + own.slice(0, 30),
      asset: src, icon, selector: uniqueSel(el), label, text: (clickable ? text : own).slice(0, 60),
      interactive: clickable, path: src || uniqueSel(el), ...box(r),
    });
  }

  // 2) PixiJS scene graph(s): v8 app from the init hook, v4-7 roots from the render() wrapper
  // what each renderer really draws comes first; app.stage only if nothing was drawn yet
  const roots = [...(window.__PIXI_ROOTS__ || new Map())];
  const app = window.__PIXI_APP__;
  if (app && app.stage && !roots.some(([r]) => r === app.renderer)) roots.push([app.renderer, app.stage]);
  const canvases = [];
  if (!roots.length)
    return { version: null, canvas: null, dpr, elements: out };

  const fileName = (s) => s ? String(s).split(/[?#]/)[0].split("/").pop().replace(/-[A-Za-z0-9_-]{8}(?=\.)/, "") : "";
  const textureName = (t) => {
    if (!t) return "";
    const src = t.source || {};        // v8
    const res = src.resource;
    const bt = t.baseTexture || {};    // v4-7
    const ids = t.textureCacheIds || [];
    return t.label || src.label || (res && (res.currentSrc || res.src)) ||
      (ids.find(i => !/^(data:|blob:)/.test(i)) || "") || bt.imageUrl || "";
  };

  // which asset a node draws: sprite-sheet file + frame, or Spine attachments (+ their atlas page)
  const assetOf = (node) => {
    let sheet = "", frame = "", parts = [], anims = [];
    const t = node.texture;
    if (t && t.width > 1) {
      const src = t.source || {}, res = src.resource, bt = t.baseTexture || {};
      sheet = fileName(src.label || (res && (res.currentSrc || res.src)) || bt.imageUrl ||
                       (bt.resource && bt.resource.url) || (bt.textureCacheIds || [])[0] || "");
      frame = t.label || (t.textureCacheIds || []).find(i => !/^(data:|blob:|https?:)/.test(i)) || "";
      if (frame.includes("/")) frame = fileName(frame);
      if (frame === sheet) frame = "";
    }
    const sk = node.skeleton;
    if (sk && sk.slots) {
      for (const s of sk.slots) {
        // Spine <=4.2: slot.attachment; 4.3 moved it onto the slot's pose
        const a = s.attachment || (s.getAttachment && s.getAttachment()) ||
          (s.pose && s.pose.attachment) || (s.applied && s.applied.attachment) || (s.appliedPose && s.appliedPose.attachment);
        if (!a || !a.name) continue;
        if (parts.length < 8 && !parts.includes(a.name)) parts.push(a.name);
        const pg = a.region && a.region.page;
        if (!sheet && pg && pg.name) sheet = fileName(pg.name);
      }
    }
    // the animation a Spine object is playing names what it shows (reel symbols: "l1_static")
    const tracks = (node.state && node.state.tracks) || [];
    for (const t of tracks) if (t && t.animation && t.animation.name && !anims.includes(t.animation.name)) anims.push(t.animation.name);
    return { sheet, frame, parts, anims };
  };

  // Objects attached to Spine slots (Pixi 8: includeInBuild = false) are positioned by Spine at
  // draw time, so Pixi's cached world transform and getBounds() are stale for them. Rebuild their
  // transform from local transforms, root -> node.
  const mul = (m, n) => ({ a: m.a * n.a + m.c * n.b, b: m.b * n.a + m.d * n.b, c: m.a * n.c + m.c * n.d,
                           d: m.b * n.c + m.d * n.d, tx: m.a * n.tx + m.c * n.ty + m.tx, ty: m.b * n.tx + m.d * n.ty + m.ty });
  const manualWorld = (node) => {
    const chain = [];
    for (let n = node; n; n = n.parent) chain.unshift(n);
    let M = { a: 1, b: 0, c: 0, d: 1, tx: 0, ty: 0 };
    for (const n of chain) {
      if (n.updateLocalTransform) n.updateLocalTransform();
      if (n.localTransform) M = mul(M, n.localTransform);
    }
    return M;
  };
  const detachedBounds = (node) => {
    const M = manualWorld(node);
    const s = Math.hypot(M.a, M.b) || 1;
    const sk = node.skeleton && node.skeleton.data;
    // size: the Spine export's setup-pose box when known, else the object's own texture size
    const w = ((sk && sk.width) || (node.texture && node.texture.width) || 40) * s;
    const h = ((sk && sk.height) || (node.texture && node.texture.height) || 40) * s;
    return { x: M.tx - w / 2, y: M.ty - h / 2, width: w, height: h, approx: true };
  };

  let rect, sx, sy;
  const walk = (node, path, detached) => {
    if (!node || !node.visible || node.worldAlpha === 0) return;
    detached = detached || node.includeInBuild === false;
    const label = node.label || node.name || "";
    // named nodes give readable paths; unnamed ones fall back to their child index
    const here = path;
    const interactive = node.eventMode === "static" || node.eventMode === "dynamic" ||
      node.interactive === true || node.buttonMode === true;
    const tex = textureName(node.texture);
    const text = typeof node.text === "string" ? node.text : "";
    const spine = node.skeleton || node.spineData ? "spine" : "";
    const named = /button|btn/i.test(label);
    const sprite = !!(node.texture && node.texture.width > 1);  // drawn image, even if unnamed
    if (interactive || tex || text || spine || named || sprite) {
      let b;
      try { b = detached ? detachedBounds(node) : node.getBounds(); } catch (e) { b = null; }
      if (b && b.width > 1 && b.height > 1) {
        const px = rect.left + b.x * sx, py = rect.top + b.y * sy, pw = b.width * sx, ph = b.height * sy;
        const onPage = px + pw > 0 && py + ph > 0 && px < window.innerWidth && py < window.innerHeight;
        const [scx, scy] = toScreen(px + pw / 2, py + ph / 2);
        const [bx, by] = toScreen(px, py);
        const a = assetOf(node);
        out.push({
          layer: "pixi",
          name: (named && label) || label || a.frame || fileName(tex) || (text && `text:${text.slice(0, 30)}`) ||
                (a.parts.length ? `spine ${a.parts[0]}` : `${spine || "node"} ${here}`),
          texture: tex.length > 120 ? tex.slice(0, 120) + "..." : tex,
          sheet: a.sheet, frame: a.frame, parts: a.parts, anims: a.anims,
          label, text: text.slice(0, 60), interactive, cursor: node.cursor || "",
          page_x: Math.round(ox + px + pw / 2), page_y: Math.round(oy + py + ph / 2),
          screen_x: scx, screen_y: scy,
          box: [bx, by, Math.round(pw * dpr), Math.round(ph * dpr)],
          on_screen: onPage, path: here, approx_box: !!b.approx,
        });
      }
    }
    (node.children || []).forEach((c, i) => walk(c, here + "/" + (c.label || c.name || i), detached));
  };
  for (const [renderer, root] of roots) {
    // v8: renderer.canvas (renderer.view is a ViewSystem); v4-7: renderer.view is the canvas
    const canvas = [renderer.canvas, renderer.view && renderer.view.canvas, renderer.view]
      .find(c => c && c.getBoundingClientRect);
    if (!canvas) continue;
    rect = canvas.getBoundingClientRect();
    const scr = renderer.screen || { width: canvas.width, height: canvas.height };
    sx = rect.width / scr.width; sy = rect.height / scr.height;
    canvases.push([Math.round(ox + rect.left), Math.round(oy + rect.top), Math.round(rect.width), Math.round(rect.height)]);
    walk(root, "", false);
  }
  return { version: window.__PIXI_VERSION__ || (window.PIXI && window.PIXI.VERSION), canvas: canvases,
           dpr, elements: out };
}
"""


def frame_offset(frame):
    """Top-left of an iframe in top-page CSS px (sums nested iframes)."""
    ox = oy = 0.0
    while frame.parent_frame:
        box = frame.frame_element().bounding_box()
        if not box:
            return None  # hidden iframe
        ox, oy = ox + box["x"], oy + box["y"]
        frame = frame.parent_frame
    return ox, oy


def collect(page):
    top = page.evaluate(TOP_JS)
    data = {"version": None, "canvas": [], "dpr": top["dpr"], "frames": [], "elements": []}
    for frame in page.frames:
        try:
            off = frame_offset(frame)
            if off is None:
                continue
            d = frame.evaluate(DUMP_JS, {**top, "ox": off[0], "oy": off[1]})
        except Exception:
            continue  # frame navigated away or cross-origin hiccup
        data["version"] = data["version"] or d["version"]
        data["canvas"] += d["canvas"] or []
        data["frames"].append({"url": frame.url[:150], "offset": off, "elements": len(d["elements"])})
        data["elements"] += d["elements"]
    return data


def dump(page, only, show_pixi):
    data = collect(page)
    els = [e for e in data["elements"] if e["on_screen"] and (show_pixi or e["layer"] == "html")]
    if only:
        keys = [k.strip().lower() for k in only.split(",")]
        els = [e for e in els if any(k in (e["name"] + e["label"] + e["text"] + e["path"]).lower() for k in keys)]
    print(f"\nPixi {data['version']}  canvas at {data['canvas']}  dpr {data['dpr']}  "
          f"{len(data['elements'])} elements ({len(els)} shown)")
    print(f"  {'name':26} {'layer':5} {'click':>5}  {'page x,y':>11}  {'screen x,y':>11}  {'w x h':>9}  selector / text")
    for e in sorted(els, key=lambda e: (e["layer"], not e["interactive"], e["page_y"], e["page_x"])):
        extra = e["selector"] if e["layer"] == "html" and e["interactive"] else (e["text"] or e["path"])
        print(f"  {e['name'][:26]:26} {e['layer']:5} {'yes' if e['interactive'] else '':>5}  "
              f"{e['page_x']:>5},{e['page_y']:<5}  {e['screen_x']:>5},{e['screen_y']:<5}  "
              f"{str(e['box'][2]) + 'x' + str(e['box'][3]):>9}  {extra[:60]}")
    with open(os.path.join(HERE, "scene_elements.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("  -> scene_elements.json")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("--every", type=float, help="dump automatically every N seconds")
    ap.add_argument("--only", help="comma-separated filters, e.g. spin,autoplay,bet")
    ap.add_argument("--seconds", type=float, help="quit after N seconds (with --every)")
    ap.add_argument("--pixi", action="store_true", help="also list canvas objects (reel symbols, characters)")
    args = ap.parse_args()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, args=["--start-maximized"])
        page = browser.new_page(no_viewport=True)
        page.add_init_script(HOOK_JS)  # runs before any of the game's scripts
        page.goto(args.url, wait_until="domcontentloaded")
        print("Game opening. The scene fills in once loading finishes.")
        if args.every:
            stop = threading.Event()
            if not args.seconds:
                threading.Thread(target=lambda: (input("Press Enter to quit.\n"), stop.set()), daemon=True).start()
            waited = 0.0
            while not stop.is_set() and (not args.seconds or waited < args.seconds):
                page.wait_for_timeout(args.every * 1000)
                waited += args.every
                dump(page, args.only, args.pixi)
        else:
            while input("\nEnter = dump coordinates, q = quit: ").strip().lower() != "q":
                dump(page, args.only, args.pixi)
        browser.close()


if __name__ == "__main__":
    main()
