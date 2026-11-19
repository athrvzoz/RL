"""
get_images.py - save a game's images (png/webp/jpg/gif/svg...) as real files.

Three sources, cheapest first:
  1. har        images the browser loaded during the capture (inside session.har)
  2. embedded   data:image/...;base64 strings inside the game's JSON/JS files
                (many engines pack whole sprite sheets into JSON)
  3. download   images listed in assets.csv that never loaded - fetched from their URL

  python get_images.py                              newest capture
  python get_images.py captures\\20261104_114321
  python get_images.py captures\\20261104_114321 --no-download    only 1 + 2
  python get_images.py captures\\20261104_114321 --all            also audio, fonts, spine...

Run assets.py on the capture first so assets.csv exists.
Files go to <capture>/images/ (or <capture>/files/ with --all), mirroring the URL path.
For personal study only - the artwork belongs to the game provider.
"""
import argparse
import base64
import csv
import glob
import json
import os
import re
import sys
import urllib.request
from urllib.parse import urlparse, unquote

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".avif", ".gif", ".svg", ".ktx", ".ktx2", ".basis"}
DATA_URI = re.compile(r"data:image/([a-z0-9.+-]+);base64,([A-Za-z0-9+/=]{200,})", re.I)
MIME_EXT = {"png": ".png", "jpeg": ".jpg", "jpg": ".jpg", "webp": ".webp", "gif": ".gif",
            "svg+xml": ".svg", "avif": ".avif"}


def out_path(root, url):
    u = urlparse(url)
    parts = [re.sub(r'[<>:"|?*]', "_", p) for p in [u.netloc] + unquote(u.path).strip("/").split("/") if p]
    full = os.path.join(root, *parts)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    return full


def save(path, data, log, how):
    if os.path.exists(path) and os.path.getsize(path) == len(data):
        return False
    with open(path, "wb") as f:
        f.write(data)
    log.append((how, path, len(data)))
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", nargs="?")
    ap.add_argument("--no-download", action="store_true", help="don't fetch anything from the network")
    ap.add_argument("--all", action="store_true", help="every asset type, not just images")
    args = ap.parse_args()
    cap = os.path.abspath(args.capture or sorted(glob.glob(os.path.join(HERE, "captures", "*")))[-1])
    root = os.path.join(cap, "files" if args.all else "images")
    wanted = (lambda url: True) if args.all else \
        (lambda url: os.path.splitext(urlparse(url).path)[1].lower() in IMAGE_EXT)

    log, have, doc_url = [], set(), None

    # ── 1. from the HAR ───────────────────────────────────────────────────────
    har = os.path.join(cap, "session.har")
    entries = []
    if os.path.exists(har):
        try:
            entries = json.load(open(har, encoding="utf-8"))["log"]["entries"]
        except ValueError:
            print("note: session.har is incomplete (capture closed before Enter); skipping it")
    for e in entries:
        url, c = e["request"]["url"], e["response"].get("content", {})
        if doc_url is None and c.get("mimeType", "").startswith("text/html"):
            doc_url = url
        if url.startswith("data:") or not wanted(url) or not c.get("text") or e["response"]["status"] != 200:
            continue
        data = base64.b64decode(c["text"]) if c.get("encoding") == "base64" else c["text"].encode("utf-8")
        save(out_path(root, url.split("?")[0]), data, log, "har")
        have.add(url.split("?")[0])

    # ── 2. data: URIs embedded in the saved JSON / JS sources ────────────────
    emb_root = os.path.join(root, "_embedded")
    for src in glob.glob(os.path.join(cap, "sources", "**", "*"), recursive=True):
        if not src.lower().endswith((".json", ".js", ".css", ".html", ".atlas", ".xml")) or os.path.getsize(src) > 60_000_000:
            continue
        text = open(src, encoding="utf-8", errors="replace").read()
        if "data:image" not in text:
            continue
        stem = os.path.splitext(os.path.basename(src))[0]
        names = {}
        if src.lower().endswith(".json"):  # name each image after the JSON key that holds it
            def walk(o, path):
                if isinstance(o, dict):
                    for k, v in o.items():
                        walk(v, path + [k])
                elif isinstance(o, list):
                    for i, v in enumerate(o):
                        walk(v, path + [str(i)])
                elif isinstance(o, str) and o.startswith("data:image"):
                    key = next((p for p in reversed(path) if re.search(r"\.\w{2,5}$", p)), None) or "_".join(path[-2:])
                    names[o[:300]] = key
            try:
                walk(json.loads(text), [])
            except ValueError:
                pass
        seen = {}
        for m in DATA_URI.finditer(text):
            ext = MIME_EXT.get(m.group(1).lower(), "." + m.group(1).lower())
            key = names.get(m.group(0)[:300]) or f"image_{len(seen) + 1}"
            base = re.sub(r"[^\w.-]+", "_", os.path.splitext(key)[0])
            seen[base] = seen.get(base, 0) + 1
            if seen[base] > 1:
                base += f"_{seen[base]}"
            try:
                data = base64.b64decode(m.group(2))
            except ValueError:
                continue
            os.makedirs(os.path.join(emb_root, stem), exist_ok=True)
            save(os.path.join(emb_root, stem, base + ext), data, log, "embedded")

    # ── 3. download the rest listed in assets.csv ────────────────────────────
    failed = []
    csv_path = os.path.join(cap, "assets.csv")
    if not os.path.exists(csv_path):
        print("note: no assets.csv - run `python assets.py` on this capture to also get images that never loaded")
    elif not args.no_download:
        for line in open(os.path.join(cap, "traffic.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            if doc_url is None and r["kind"] == "http_request" and r.get("type") == "document":
                doc_url = r["url"]
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                 "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"}
        if doc_url:
            headers["Referer"] = doc_url  # some game CDNs refuse requests without it
        rows = [r for r in csv.DictReader(open(csv_path, encoding="utf-8"))
                if wanted(r["url"]) and r["url"].split("?")[0] not in have]
        for i, r in enumerate(rows, 1):
            path = out_path(root, r["url"])
            if os.path.exists(path) and os.path.getsize(path) > 0:
                continue
            try:
                with urllib.request.urlopen(urllib.request.Request(r["url"], headers=headers), timeout=20) as resp:
                    save(path, resp.read(), log, "download")
                print(f"  [{i}/{len(rows)}] ok    {r['file']}")
            except Exception as e:
                failed.append((r["url"], str(e)[:80]))
                print(f"  [{i}/{len(rows)}] FAIL  {r['file']}  ({str(e)[:60]})")

    # ── report ───────────────────────────────────────────────────────────────
    by = {}
    for how, _, n in log:
        by.setdefault(how, [0, 0])
        by[how][0] += 1
        by[how][1] += n
    print(f"\nSaved to {root}")
    for how in ("har", "embedded", "download"):
        if how in by:
            print(f"  {how:9} {by[how][0]:4} files  {by[how][1] / 1e6:6.1f} MB")
    if not log:
        print("  nothing new (already saved, or nothing found)")
    if failed:
        print(f"  {len(failed)} downloads failed - e.g. {failed[0][1]}")
        with open(os.path.join(root, "_failed.txt"), "w", encoding="utf-8") as f:
            f.writelines(f"{u}\t{err}\n" for u, err in failed)


if __name__ == "__main__":
    main()
