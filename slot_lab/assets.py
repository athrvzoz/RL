"""
assets.py - list every asset a game uses, from a capture folder (no replaying needed).

Sources combined:
  loaded      every response recorded in traffic.jsonl / session.har during the capture
  code        asset file names written in the game's JS / CSS / HTML (its full asset list,
              including bonus/free-spin assets that only load later)
  .atlas      sprite-sheet page images listed inside Spine/Pixi atlas files
  .json       file names inside JSON manifests (sound lists, spritesheets...)

  python assets.py                         newest capture
  python assets.py captures\\20261104_114321

Writes assets.md (tables per type) and assets.csv into the capture folder.
"""
import base64
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict
from urllib.parse import urljoin, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = ("png|jpe?g|webp|avif|gif|svg|m4a|mp3|ogg|wav|aac|webm|mp4|json|atlas|skel|fnt|xml|"
       "woff2?|ttf|otf|ktx2?|basis|css|js")
REF = re.compile(r"""["'`(]([^"'`()\s<>{}]{1,300}?\.(?:%s))(?:\?[^"'`)\s]*)?["'`)]""" % EXT, re.I)
SKIP_HOSTS = ("visualstudio.com", "monitor.azure.com", "servicebus.windows.net", "google-analytics", "doubleclick")

KIND = {
    "image": "png jpg jpeg webp avif gif svg ktx ktx2 basis",
    "audio": "m4a mp3 ogg wav aac",
    "video": "webm mp4",
    "animation (Spine)": "skel",
    "atlas / spritesheet": "atlas",
    "font": "woff woff2 ttf otf fnt",
    "data / config": "json xml",
    "code / style": "js css",
}
EXT_KIND = {e: k for k, exts in KIND.items() for e in exts.split()}


def kind_of(url):
    ext = os.path.splitext(urlparse(url).path)[1].lower().lstrip(".")
    return EXT_KIND.get(ext, "other")


def clean(url):
    u = urlparse(url)
    return f"{u.scheme}://{u.netloc}{u.path}"


def main():
    cap = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join(HERE, "captures", "*")))[-1]
    cap = os.path.abspath(cap)

    # ── 1. what was loaded ────────────────────────────────────────────────────
    assets = {}  # url -> info
    doc_url = None
    bodies = {}  # url -> text body for parsing (js/css/html/json/atlas)

    def add(url, how, **kv):
        url = clean(url)
        if not url.startswith("http") or any(h in url for h in SKIP_HOSTS):
            return None
        a = assets.setdefault(url, {"url": url, "file": os.path.basename(urlparse(url).path) or url,
                                    "kind": kind_of(url), "loaded": False, "status": "", "bytes": "",
                                    "found_in": set()})
        a["found_in"].add(how)
        a.update({k: v for k, v in kv.items() if v not in (None, "")})
        return a

    har_path = os.path.join(cap, "session.har")
    entries = []
    if os.path.exists(har_path):
        try:
            entries = json.load(open(har_path, encoding="utf-8"))["log"]["entries"]
        except ValueError:
            print("note: session.har is incomplete (capture was closed before Enter) - using traffic.jsonl only")
    for e in entries:
        url, res = e["request"]["url"], e["response"]
        if e["request"]["method"] != "GET" or not (200 <= res["status"] < 300):
            continue
        c = res.get("content", {})
        if c.get("mimeType", "").startswith("text/html") and doc_url is None:
            doc_url = url
        a = add(url, "loaded", loaded=True, status=res["status"], bytes=c.get("size") or "")
        if a and c.get("text") and re.search(r"\.(js|css|html?|json|atlas|xml)$|/$", urlparse(url).path, re.I):
            raw = base64.b64decode(c["text"]) if c.get("encoding") == "base64" else c["text"].encode("utf-8", "replace")
            bodies[clean(url)] = raw.decode("utf-8", "replace")
    for line in open(os.path.join(cap, "traffic.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        if r["kind"] == "http_response" and r.get("method", "GET") == "GET" and 200 <= r["status"] < 300:
            if r.get("type") == "document" and doc_url is None:
                doc_url = r["url"]
            a = add(r["url"], "loaded", loaded=True, status=r["status"])
            if a and r.get("saved_as") and not str(r["saved_as"]).startswith("<"):
                p = os.path.join(cap, "sources", *r["saved_as"].split("/"))
                if os.path.exists(p) and clean(r["url"]) not in bodies and os.path.getsize(p) < 30_000_000:
                    try:
                        bodies[clean(r["url"])] = open(p, encoding="utf-8").read()
                    except UnicodeDecodeError:
                        pass
                if os.path.exists(p):
                    a["bytes"] = a["bytes"] or os.path.getsize(p)

    game_root = urljoin(doc_url or "", ".")

    # ── 2. references inside code, atlases and manifests ──────────────────────
    def resolve(ref, base_url):
        if ref.startswith(("http://", "https://")):
            return ref
        if ref.startswith("//"):
            return "https:" + ref
        if ref.startswith("/"):
            return urljoin(base_url, ref)
        if re.match(r"^\.?/?assets/", ref) and game_root:  # Vite "./assets/x" is relative to the page
            return urljoin(game_root, ref)
        return urljoin(base_url, ref)

    for src_url, text in list(bodies.items()):
        path = urlparse(src_url).path.lower()
        how = "atlas" if path.endswith(".atlas") else "manifest" if path.endswith((".json", ".xml")) else "code"
        if how == "atlas":  # page images are bare lines like "dwtd_master.webp"
            for ln in text.splitlines():
                ln = ln.strip()
                if re.fullmatch(r"[^:\s]+\.(png|jpe?g|webp|avif|ktx2?|basis)", ln, re.I):
                    add(resolve(ln, src_url), "atlas:" + os.path.basename(path))
            continue
        for m in REF.finditer(text):
            ref = m.group(1)
            if ref.startswith(("data:", "blob:")) or not re.fullmatch(r"[\w@%~+./:-]+", ref):
                continue  # minified code fragments like "e,n.svg" are not file names
            a = add(resolve(ref, src_url), f"{how}:{os.path.basename(path) or 'page'}")
            if a and a["url"] == clean(src_url):
                a["found_in"].discard(f"{how}:{os.path.basename(path)}")

    # a name in the code that was actually loaded from a resolution folder (assets/hd/x.atlas)
    # is the same asset, not a missing one: fold it into the loaded copy
    # Likewise a build-time name ("spin_icon.png") whose shipped file carries a content
    # hash ("spin_icon-BkdvTbGh.png") is the same asset.
    def unhashed(name):
        return re.sub(r"-[A-Za-z0-9_-]{8}(?=\.\w+$)", "", name)

    loaded_by_name = defaultdict(list)
    for a in assets.values():
        if a["loaded"] or unhashed(a["file"]) != a["file"]:
            loaded_by_name[unhashed(a["file"])].append(a)
    for url in [u for u, a in assets.items() if not a["loaded"]]:
        twins = [t for t in loaded_by_name[unhashed(assets[url]["file"])] if t["url"] != url]
        if len(twins) == 1:
            twins[0]["found_in"] |= assets.pop(url)["found_in"]

    # ── 3. tables ─────────────────────────────────────────────────────────────
    rows = sorted(assets.values(), key=lambda a: (a["kind"], a["url"]))
    by_kind = defaultdict(list)
    for a in rows:
        by_kind[a["kind"]].append(a)
    n_loaded = sum(a["loaded"] for a in rows)

    def where(a):
        tags = sorted(t for t in a["found_in"] if t != "loaded")
        return ", ".join(tags[:3]) + (f" +{len(tags) - 3}" if len(tags) > 3 else "")

    L = [f"# Assets - {os.path.basename(cap)}\n",
         f"- Game: `{(doc_url or '?')[:110]}`",
         f"- **{len(rows)} assets**: {n_loaded} loaded during the capture, "
         f"{len(rows) - n_loaded} referenced but not loaded yet\n",
         "| type | total | loaded | not loaded |", "|---|---|---|---|"]
    for k, items in sorted(by_kind.items()):
        ld = sum(a["loaded"] for a in items)
        L.append(f"| {k} | {len(items)} | {ld} | {len(items) - ld} |")
    for k, items in sorted(by_kind.items()):
        L.append(f"\n## {k} ({len(items)})\n")
        L.append("| # | file | loaded | size | referenced in | url |\n|---|---|---|---|---|---|")
        for i, a in enumerate(items, 1):
            size = f"{int(a['bytes']):,}" if str(a["bytes"]).isdigit() else ""
            L.append(f"| {i} | `{a['file']}` | {'yes' if a['loaded'] else '**no**'} | {size} | {where(a)} | {a['url']} |")
    md = os.path.join(cap, "assets.md")
    open(md, "w", encoding="utf-8").write("\n".join(L))

    with open(os.path.join(cap, "assets.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kind", "file", "loaded", "bytes", "referenced_in", "url"])
        for a in rows:
            w.writerow([a["kind"], a["file"], a["loaded"], a["bytes"], where(a), a["url"]])

    print(f"{len(rows)} assets ({n_loaded} loaded, {len(rows) - n_loaded} not loaded yet)")
    for k, items in sorted(by_kind.items()):
        print(f"  {k:22} {len(items):4}   not loaded: {sum(not a['loaded'] for a in items)}")
    print(f"-> {md}\n-> {os.path.join(cap, 'assets.csv')}")


if __name__ == "__main__":
    main()
