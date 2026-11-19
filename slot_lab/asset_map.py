"""
asset_map.py - map every asset to where it is on screen, for automation.

Opens the game (same hooks as scene_locate.py). When the screen you want is up
(reels visible, menu open...), press Enter and it snapshots:

  maps/<time>/screenshot.png
  maps/<time>/asset_map.json   controls (click point, selector, icon asset), readouts
                               (balance/bet text), and asset -> every place it's drawn
  maps/<time>/mindmap.md       Mermaid mind map: game -> layers -> controls / scene -> assets
  maps/<time>/map.html         open in a browser: screenshot with every box, searchable list

  python asset_map.py "<game url>"                  press Enter to snapshot (again for more), q to quit
  python asset_map.py "<game url>" --wait 40        snapshot once after 40 s

Snapshot each screen you care about (base game, free spins, menu) - positions and
visible assets change between them.
"""
import argparse
import base64
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime

from playwright.sync_api import sync_playwright

from scene_locate import HOOK_JS, collect

HERE = os.path.dirname(os.path.abspath(__file__))
CONTROL_NAME = re.compile(r"button|btn", re.I)


def css_box(e, dpr):
    """Element box in page CSS px: [left, top, width, height]."""
    w, h = e["box"][2] / dpr, e["box"][3] / dpr
    return [round(e["page_x"] - w / 2), round(e["page_y"] - h / 2), round(w), round(h)]


def asset_key(e):
    if e["layer"] == "html":
        return e.get("icon") or (e.get("asset") or "").split("?")[0].split("/")[-1]
    if e.get("frame"):
        return f"{e.get('sheet') or '?'} : {e['frame']}"
    if e.get("anims"):
        return "spine anim : " + ", ".join(e["anims"][:2])
    if e.get("parts"):
        return f"{e.get('sheet') or 'spine'} : " + ", ".join(e["parts"][:3])
    return e.get("sheet") or ""


def build(data):
    dpr = data["dpr"]
    els = [e for e in data["elements"] if e["on_screen"]]
    for e in els:
        e["css_box"] = css_box(e, dpr)
        e["asset_key"] = asset_key(e)

    # Spine-slot objects (reel symbols) have reliable centres but no reliable size; they sit on
    # a grid, so size each one by the median distance to its nearest neighbour.
    approx = [e for e in els if e.get("approx_box")]
    if len(approx) >= 2:
        gaps = []
        for e in approx:
            d = [max(abs(e["page_x"] - o["page_x"]), abs(e["page_y"] - o["page_y"])) for o in approx if o is not e]
            d = [v for v in d if v >= 10]
            if d:
                gaps.append(min(d))
        if gaps:
            cell = sorted(gaps)[len(gaps) // 2]
            for e in approx:
                e["css_box"] = [round(e["page_x"] - cell / 2), round(e["page_y"] - cell / 2), cell, cell]

    # controls: real clickables, plus engine buttons by name (Pixi 4 engines handle input
    # themselves, so their buttons aren't flagged interactive). Nested wrappers of one button
    # share its box: keep the outermost, which carries the meaningful name.
    controls, seen = [], {}
    for e in els:
        named = e["layer"] == "pixi" and CONTROL_NAME.search(e.get("label") or "")
        if not (e["interactive"] or named) or e["css_box"][2] < 8 or e["css_box"][3] < 8:
            continue
        if e["layer"] == "pixi" and e.get("label") == "" and not named:
            continue
        k = tuple(round(v / 4) for v in e["css_box"])
        if k in seen:
            c = seen[k]
            c["asset"] = c["asset"] or e["asset_key"]
            continue
        c = {"name": e["name"], "layer": e["layer"], "click_page": [e["page_x"], e["page_y"]],
             "click_screen": [e["screen_x"], e["screen_y"]], "box": e["css_box"],
             "selector": e.get("selector", ""), "path": e["path"], "asset": e["asset_key"], "text": e.get("text", "")}
        seen[k] = c
        controls.append(c)
    # a Pixi button's own sprite usually sits one level below it: borrow its asset
    for c in controls:
        if c["layer"] == "pixi" and not c["asset"]:
            kid = next((e for e in els if e["path"].startswith(c["path"] + "/") and e["asset_key"]), None)
            c["asset"] = kid["asset_key"] if kid else ""

    readouts = [{"text": e["text"], "layer": e["layer"], "at_page": [e["page_x"], e["page_y"]], "box": e["css_box"],
                 "selector": e.get("selector", ""), "path": e["path"]}
                for e in els if e.get("text") and not e["interactive"] and len(e["text"]) <= 40]

    assets = defaultdict(list)
    for e in els:
        if e["asset_key"]:
            assets[e["asset_key"]].append({"name": e["name"], "layer": e["layer"], "path": e["path"],
                                           "center_page": [e["page_x"], e["page_y"]], "box": e["css_box"],
                                           "clickable": bool(e["interactive"]), "selector": e.get("selector", "")})
    return els, controls, readouts, dict(sorted(assets.items()))


def mermaid(controls, readouts, assets, els):
    def q(s):  # mermaid-safe node text
        return re.sub(r"[()\[\]{}\"`]", " ", str(s)).strip()[:60] or "?"
    L = ["```mermaid", "mindmap", "  root((Game))"]
    for layer, title in (("html", "HTML layer"), ("pixi", "Canvas layer")):
        cs = [c for c in controls if c["layer"] == layer]
        rs = [r for r in readouts if r["layer"] == layer]
        if not cs and not rs and not any(e["layer"] == layer for e in els):
            continue
        L.append(f"    {title}")
        if cs:
            L.append("      Controls")
            for c in cs[:40]:
                L.append(f"        {q(c['name'])} at {c['click_page'][0]},{c['click_page'][1]}")
                if c["asset"]:
                    L.append(f"          {q(c['asset'])}")
        if rs:
            L.append("      Readouts")
            L += [f"        {q(r['text'])}" for r in rs[:25]]
        if layer == "pixi":  # scene containers, two levels, with how many objects each holds
            groups = defaultdict(lambda: defaultdict(int))
            for e in els:
                if e["layer"] == "pixi":
                    p = [s for s in e["path"].split("/") if s and not s.isdigit()]
                    if p:
                        groups[p[0]][p[1] if len(p) > 1 else "(direct)"] += 1
            L.append("      Scene")
            for g, subs in list(groups.items())[:12]:
                L.append(f"        {q(g)}")
                for s, n in sorted(subs.items(), key=lambda kv: -kv[1])[:8]:
                    L.append(f"          {q(s)} x{n}")
    sheets = defaultdict(list)
    for k in assets:
        sheet, _, frame = k.partition(" : ")
        sheets[sheet].append(frame)
    L.append("    Assets on screen")
    for sheet, frames in list(sheets.items())[:30]:
        L.append(f"      {q(sheet)}")
        L += [f"        {q(f)}" for f in frames[:10] if f]
    L.append("```")
    return "\n".join(L)


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Asset map</title><style>
:root{--bg:#f6f7f9;--fg:#1d2330;--mut:#667085;--line:#d7dbe2;--card:#fff;--acc:#2563eb;--ctl:#e11d48}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#111418;--fg:#e6e9ef;--mut:#98a2b3;--line:#2b313a;--card:#181c22;--acc:#60a5fa;--ctl:#fb7185}}
*{box-sizing:border-box}body{margin:0;font:13px/1.45 system-ui,sans-serif;background:var(--bg);color:var(--fg);display:grid;grid-template-columns:minmax(0,1fr) 380px;height:100vh}
#stage{overflow:auto;padding:12px}#wrap{position:relative;display:inline-block}#wrap img{display:block;max-width:none}
.b{position:absolute;border:1px solid var(--acc);background:transparent;cursor:pointer}.b.c{border:2px solid var(--ctl)}
.b.hide{display:none}.b.on{background:rgba(37,99,235,.25);border-width:3px;z-index:5}
aside{border-left:1px solid var(--line);background:var(--card);display:flex;flex-direction:column;min-height:0}
header{padding:12px;border-bottom:1px solid var(--line)}h1{font-size:15px;margin:0 0 8px}
input{width:100%;padding:7px 9px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg)}
.tabs{display:flex;gap:6px;margin-top:8px;flex-wrap:wrap}.tabs button{border:1px solid var(--line);background:var(--bg);color:var(--fg);border-radius:6px;padding:4px 8px;cursor:pointer}
.tabs button.on{border-color:var(--acc);color:var(--acc)}#list{overflow:auto;flex:1;padding:6px 0}
.it{padding:6px 12px;border-bottom:1px solid var(--line);cursor:pointer}.it:hover,.it.on{background:var(--bg)}
.it b{display:block;word-break:break-all}.it small{color:var(--mut);display:block;word-break:break-all}
#det{border-top:1px solid var(--line);padding:10px 12px;max-height:38vh;overflow:auto;font:12px/1.4 ui-monospace,monospace;white-space:pre-wrap;word-break:break-all}
@media (max-width:760px){body{grid-template-columns:1fr;grid-template-rows:55vh 45vh}aside{border-left:0;border-top:1px solid var(--line)}}
</style></head><body><div id="stage"><div id="wrap"><img id="shot" alt="Game screenshot with asset boxes"></div></div>
<aside><header><h1>Asset map</h1><input id="q" placeholder="Search name, asset, selector..." aria-label="Search">
<div class="tabs" role="tablist"><button data-t="controls" class="on">Controls</button><button data-t="readouts">Readouts</button><button data-t="assets">Assets</button><button data-t="all">Everything</button></div></header>
<div id="list"></div><div id="det">Click a box or a row.</div></aside>
<script>
const D=__DATA__;const shot=document.getElementById('shot');shot.src='data:image/png;base64,'+D.shot;
const wrap=document.getElementById('wrap'),list=document.getElementById('list'),det=document.getElementById('det');
let tab='controls',sel=null;
const rows={controls:D.controls.map(c=>({t:c.name,s:`click page (${c.click_page}) · ${c.asset||'no asset'}`,box:c.box,c:1,d:c})),
 readouts:D.readouts.map(r=>({t:r.text,s:`at (${r.at_page}) · ${r.selector||r.path}`,box:r.box,d:r})),
 assets:Object.entries(D.assets).flatMap(([k,us])=>us.map(u=>({t:k,s:`${u.name} · (${u.center_page})`,box:u.box,c:u.clickable,d:Object.assign({asset:k},u)}))),
 all:D.elements.map(e=>({t:e.name,s:`${e.layer} · ${e.asset_key||''} · (${e.page_x},${e.page_y})`,box:e.css_box,c:e.interactive,d:e}))};
function show(r){sel=r;det.textContent=JSON.stringify(r.d,null,2);document.querySelectorAll('.b').forEach(b=>b.classList.toggle('on',b._r===r));
 document.querySelectorAll('.it').forEach(i=>i.classList.toggle('on',i._r===r));const b=[...document.querySelectorAll('.b')].find(b=>b._r===r);b&&b.scrollIntoView({block:'center',inline:'center'})}
function render(){const q=document.getElementById('q').value.toLowerCase();wrap.querySelectorAll('.b').forEach(b=>b.remove());list.innerHTML='';
 for(const r of rows[tab]){if(q&&!JSON.stringify(r.d).toLowerCase().includes(q)&&!r.t.toLowerCase().includes(q))continue;
  const b=document.createElement('div');b.className='b'+(r.c?' c':'');Object.assign(b.style,{left:r.box[0]+'px',top:r.box[1]+'px',width:r.box[2]+'px',height:r.box[3]+'px'});
  b.title=r.t;b._r=r;b.onclick=()=>show(r);wrap.appendChild(b);
  const i=document.createElement('div');i.className='it';i._r=r;i.innerHTML='<b></b><small></small>';i.querySelector('b').textContent=r.t;i.querySelector('small').textContent=r.s;i.onclick=()=>show(r);list.appendChild(i)}}
document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{tab=b.dataset.t;document.querySelectorAll('.tabs button').forEach(x=>x.classList.toggle('on',x===b));render()});
document.getElementById('q').oninput=render;render();
</script></body></html>"""


def snapshot(page, url, out_root):
    out = os.path.join(out_root, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out)
    shot = page.screenshot()
    open(os.path.join(out, "screenshot.png"), "wb").write(shot)
    data = collect(page)
    els, controls, readouts, assets = build(data)
    doc = {"url": url, "pixi": data["version"], "canvas": data["canvas"], "dpr": data["dpr"],
           "controls": controls, "readouts": readouts, "assets": assets}
    json.dump(doc, open(os.path.join(out, "asset_map.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    open(os.path.join(out, "mindmap.md"), "w", encoding="utf-8").write(
        "# Asset mind map\n\nRender with any Mermaid viewer (VS Code: Markdown Preview Mermaid Support).\n\n"
        + mermaid(controls, readouts, assets, els) + "\n")
    payload = {"shot": base64.b64encode(shot).decode(), "controls": controls, "readouts": readouts,
               "assets": assets, "elements": [{k: e.get(k) for k in ("name", "layer", "path", "asset_key", "selector",
                                                                     "page_x", "page_y", "css_box", "interactive", "text")}
                                              for e in els]}
    open(os.path.join(out, "map.html"), "w", encoding="utf-8").write(
        HTML.replace("__DATA__", json.dumps(payload).replace("</", "<\\/")))

    print(f"\n{len(els)} objects on screen, {len(controls)} controls, {len(readouts)} readouts, {len(assets)} assets")
    for c in controls[:30]:
        print(f"  {c['name'][:32]:32} click {c['click_page'][0]:>5},{c['click_page'][1]:<5} "
              f"{c['asset'][:40]:40} {c['selector'][:40]}")
    print(f"-> {out}\\map.html  (open in a browser)")
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("--wait", type=float, help="snapshot once after N seconds, then quit")
    ap.add_argument("--out", default=os.path.join(HERE, "maps"))
    args = ap.parse_args()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, args=["--start-maximized"])
        page = browser.new_page(no_viewport=True)
        page.add_init_script(HOOK_JS)
        page.goto(args.url, wait_until="domcontentloaded")
        if args.wait:
            page.wait_for_timeout(args.wait * 1000)
            snapshot(page, args.url, args.out)
        else:
            while input("\nEnter = snapshot this screen, q = quit: ").strip().lower() != "q":
                snapshot(page, args.url, args.out)
        browser.close()


if __name__ == "__main__":
    main()
