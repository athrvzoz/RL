"""
locate.py - find where the game's buttons are on your screen (for RPA clicking).

The game draws everything on one <canvas>, so there are no HTML elements to
select. Instead we take the game's own button images (saved in the capture's
session.har) and search for them in a screenshot with OpenCV template matching.

  python locate.py extract              pull *_icon*.png etc. out of the latest capture
  python locate.py find                 screenshot the screen, print button coordinates
  python locate.py find --image shot.png    search a saved screenshot instead
  python locate.py find --only spin,stop    just these templates

Output: name, centre (x, y) in screen pixels, box, matched scale and score,
plus locate_result.png with boxes drawn and locate_result.json for your RPA tool.
"""
import argparse
import base64
import ctypes
import glob
import json
import os
import re
import sys

import cv2
import numpy as np
from PIL import ImageGrab

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATES = os.path.join(HERE, "templates")
WANTED = re.compile(r"(icon|button|btn|badge|logo)[^/]*\.(png|webp)$", re.I)


def latest_har():
    hars = sorted(glob.glob(os.path.join(HERE, "captures", "*", "session.har")))
    if not hars:
        sys.exit("No captures/*/session.har found - run capture.py first.")
    return hars[-1]


def extract(har_path):
    os.makedirs(TEMPLATES, exist_ok=True)
    har = json.load(open(har_path, encoding="utf-8"))
    n = 0
    for e in har["log"]["entries"]:
        url, content = e["request"]["url"], e["response"].get("content", {})
        name = url.split("?")[0].rsplit("/", 1)[-1]
        if not WANTED.search(name) or not content.get("text"):
            continue
        data = base64.b64decode(content["text"]) if content.get("encoding") == "base64" \
            else content["text"].encode("latin-1")
        # strip Vite's content hash: spin_icon-BkdvTbGh.png -> spin_icon.png
        clean = re.sub(r"-[A-Za-z0-9_-]{8}(?=\.)", "", name)
        with open(os.path.join(TEMPLATES, clean), "wb") as f:
            f.write(data)
        n += 1
        print("  saved templates/" + clean)
    print(f"{n} templates from {har_path}")


def load_template(path):
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    bgr, alpha = img[:, :, :3], img[:, :, 3]
    mask = cv2.merge([(alpha > 128).astype(np.uint8) * 255] * 3)  # match only opaque pixels
    return bgr, mask


def best_match(screen, tpl, mask, scales):
    best = (-1.0, None, None)
    for s in scales:
        t = cv2.resize(tpl, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        m = cv2.resize(mask, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        h, w = t.shape[:2]
        if h < 8 or w < 8 or h > screen.shape[0] or w > screen.shape[1]:
            continue
        res = cv2.matchTemplate(screen, t, cv2.TM_CCORR_NORMED, mask=m)
        res[~np.isfinite(res)] = 0
        _, score, _, loc = cv2.minMaxLoc(res)
        if score > best[0]:
            best = (score, loc, (w, h, s))
    return best


def find(args):
    if args.image:
        screen = cv2.imdecode(np.fromfile(args.image, np.uint8), cv2.IMREAD_COLOR)
        offset = (0, 0)
    else:
        try:  # report real pixels even with Windows display scaling (125%, 150%...)
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass
        shot = ImageGrab.grab(all_screens=True)
        screen = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)
        # all_screens: (0,0) of the image is the top-left of the leftmost monitor
        offset = (ctypes.windll.user32.GetSystemMetrics(76), ctypes.windll.user32.GetSystemMetrics(77)) \
            if os.name == "nt" else (0, 0)

    paths = sorted(glob.glob(os.path.join(TEMPLATES, "*.png")) + glob.glob(os.path.join(TEMPLATES, "*.webp")))
    if args.only:
        keys = [k.strip().lower() for k in args.only.split(",")]
        paths = [p for p in paths if any(k in os.path.basename(p).lower() for k in keys)]
    if not paths:
        sys.exit("No templates - run: python locate.py extract")

    scales = np.linspace(args.min_scale, args.max_scale, args.steps)
    found, draw = [], screen.copy()
    for p in paths:
        tpl, mask = load_template(p)
        if tpl is None:
            continue
        score, loc, size = best_match(screen, tpl, mask, scales)
        name = os.path.splitext(os.path.basename(p))[0]
        if loc is None or score < args.threshold:
            print(f"  {name:28} not found (best score {score:.3f})")
            continue
        w, h, s = size
        x, y = loc[0] + offset[0], loc[1] + offset[1]
        cx, cy = x + w // 2, y + h // 2
        found.append({"name": name, "x": cx, "y": cy, "box": [x, y, w, h],
                      "scale": round(float(s), 2), "score": round(float(score), 3)})
        print(f"  {name:28} click at ({cx}, {cy})   box {w}x{h}  scale {s:.2f}  score {score:.3f}")
        cv2.rectangle(draw, loc, (loc[0] + w, loc[1] + h), (0, 255, 0), 2)
        cv2.putText(draw, name, (loc[0], max(12, loc[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imencode(".png", draw)[1].tofile(os.path.join(HERE, "locate_result.png"))
    with open(os.path.join(HERE, "locate_result.json"), "w") as f:
        json.dump(found, f, indent=2)
    print(f"\n{len(found)} found -> locate_result.json, boxes drawn in locate_result.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--har", help="default: newest captures/*/session.har")
    f = sub.add_parser("find")
    f.add_argument("--image", help="search this screenshot instead of the live screen")
    f.add_argument("--only", help="comma-separated name parts, e.g. spin,autoplay")
    f.add_argument("--threshold", type=float, default=0.90)
    f.add_argument("--min-scale", type=float, default=0.3)
    f.add_argument("--max-scale", type=float, default=1.5)
    f.add_argument("--steps", type=int, default=25)
    args = ap.parse_args()
    if args.cmd == "extract":
        extract(args.har or latest_har())
    else:
        find(args)


if __name__ == "__main__":
    main()
