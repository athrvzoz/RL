"""
capture.py - record everything a web game does in YOUR OWN browser session.

What it saves (in captures/<timestamp>/):
  traffic.jsonl   every HTTP request/response + every WebSocket frame, in order
  ws/             one file per WebSocket connection, frames decoded where possible
  sources/        the game's files (HTML, JS, CSS, JSON, WASM...) mirrored by URL
  session.har     standard HAR file (open in DevTools: Network -> Import HAR)
  summary.md      endpoints, WebSocket URLs, message types and sample payloads

Usage:
  python capture.py "<game url>"                 play, then press Enter to stop
  python capture.py "<game url>" --seconds 120   stop automatically after 120 s
  python capture.py "<game url>" --all-assets    also save images/audio/fonts

It only watches the browser it opens; it never sends requests of its own.
"""
import argparse
import base64
import gzip
import json
import os
import re
import sys
import time
import zlib
from collections import Counter, defaultdict
from datetime import datetime
from urllib.parse import urlparse, unquote

from playwright.sync_api import sync_playwright

try:
    import msgpack  # optional: pip install msgpack
except ImportError:
    msgpack = None

SOURCE_TYPES = {"document", "script", "stylesheet", "xhr", "fetch", "manifest", "other"}
ASSET_TYPES = {"image", "media", "font"}
STATIC_EXT = re.compile(r"\.(m4a|mp3|ogg|wav|png|jpe?g|webp|gif|svg|atlas|skel|fnt|json|woff2?|ttf|js|css)$", re.I)
MAX_INLINE = 200_000  # chars of body kept in traffic.jsonl; full body is in sources/


# ── decoding helpers ─────────────────────────────────────────────────────────
def try_json(text):
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def decode_text_frame(text):
    """Text frames: plain JSON, or Engine.IO / Socket.IO packets like 42["spin",{...}]."""
    j = try_json(text)
    if j is not None:
        return {"format": "json", "value": j}
    # Engine.IO packet type is the first digit: 0 open, 1 close, 2 ping, 3 pong, 4 message
    m = re.match(r"^([0-6])(.*)$", text, re.S)
    if m:
        eio_type, rest = m.groups()
        names = {"0": "open", "1": "close", "2": "ping", "3": "pong", "4": "message", "5": "upgrade", "6": "noop"}
        out = {"format": "engine.io", "eio": names[eio_type]}
        if eio_type == "4" and rest:
            # Socket.IO: type digit, optional "/namespace,", optional ack id, then JSON
            s = re.match(r"^([0-6])(?:(/[^,]*),)?(\d*)(.*)$", rest, re.S)
            if s:
                sio_type, ns, ack, payload = s.groups()
                sio_names = {"0": "connect", "1": "disconnect", "2": "event", "3": "ack",
                             "4": "error", "5": "binary_event", "6": "binary_ack"}
                out.update(format="socket.io", sio=sio_names[sio_type], namespace=ns or "/",
                           ack_id=ack or None)
                pj = try_json(payload)
                if isinstance(pj, list) and pj and out["sio"] == "event":
                    out["event"], out["args"] = pj[0], pj[1:]
                else:
                    out["value"] = pj if pj is not None else payload
                return out
        pj = try_json(rest)
        out["value"] = pj if pj is not None else rest
        return out
    return {"format": "text", "value": text}


def decode_binary_frame(data: bytes):
    """Binary frames: try gzip/zlib, utf-8 JSON, msgpack; always keep hex + base64."""
    out = {"format": "binary", "length": len(data),
           "hex": data[:4096].hex(), "base64": base64.b64encode(data).decode()}
    candidates = [("raw", data)]
    for name, fn in (("gzip", gzip.decompress), ("zlib", zlib.decompress),
                     ("deflate", lambda b: zlib.decompress(b, -15))):
        try:
            candidates.append((name, fn(data)))
        except Exception:
            pass
    for how, blob in candidates:
        try:
            text = blob.decode("utf-8")
            if text.isprintable() or text.strip().startswith(("{", "[")):
                out.update(decoded_via=f"{how}+utf8", value=decode_text_frame(text))
                return out
        except UnicodeDecodeError:
            pass
        if msgpack:
            try:
                out.update(decoded_via=f"{how}+msgpack", value=msgpack.unpackb(blob, raw=False, strict_map_key=False))
                return out
            except Exception:
                pass
    out["note"] = "unknown binary format (maybe protobuf) - look for the decoder in sources/"
    return out


def decode_frame(payload):
    if isinstance(payload, (bytes, bytearray)):
        return decode_binary_frame(bytes(payload))
    return decode_text_frame(payload)


def frame_label(dec):
    """Short name used to group frames in the summary."""
    v = dec.get("value")
    if dec.get("event"):
        return f"socket.io event '{dec['event']}'"
    if dec.get("eio") in ("ping", "pong", "open"):
        return f"engine.io {dec['eio']}"
    if isinstance(v, dict) and dec.get("format") == "binary":
        v = v.get("value")
    if isinstance(v, dict):
        for k in ("type", "action", "cmd", "command", "method", "event", "msg", "op", "name"):
            if k in v and isinstance(v[k], (str, int)):
                return f"{k}={v[k]}"
        return "keys: " + ",".join(list(v)[:6])
    return dec.get("format", "?")


# ── file helpers ─────────────────────────────────────────────────────────────
def safe_path(root, url, content_type=""):
    u = urlparse(url)
    path = unquote(u.path) or "/"
    if path.endswith("/"):
        path += "index.html"
    name, ext = os.path.splitext(path)
    if not ext:
        ext = {"application/json": ".json", "text/html": ".html", "text/css": ".css",
               "application/javascript": ".js", "text/javascript": ".js"}.get(
            content_type.split(";")[0].strip(), ".bin")
    if u.query:  # same path, different query -> different file
        name += "__" + re.sub(r"[^A-Za-z0-9]+", "_", u.query)[:60]
    rel = re.sub(r'[<>:"|?*]', "_", f"{u.netloc}{name}{ext}").lstrip("/")
    full = os.path.join(root, *rel.split("/"))
    os.makedirs(os.path.dirname(full), exist_ok=True)
    return full, rel


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    # Windows consoles often can't print unicode; never let logging crash a capture
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("--seconds", type=int, help="stop automatically after N seconds")
    ap.add_argument("--all-assets", action="store_true", help="also save images, audio and fonts")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures"))
    args = ap.parse_args()

    out = os.path.join(args.out, datetime.now().strftime("%Y%m%d_%H%M%S"))
    src_root, ws_root = os.path.join(out, "sources"), os.path.join(out, "ws")
    os.makedirs(src_root), os.makedirs(ws_root)
    traffic = open(os.path.join(out, "traffic.jsonl"), "w", encoding="utf-8")
    t0 = time.time()

    endpoints = Counter()            # "METHOD host/path" -> count
    samples = {}                     # endpoint -> first request/response bodies
    sources = []                     # (type, rel path, bytes)
    ws_info = {}                     # ws id -> {url, file, counts}
    ws_samples = defaultdict(dict)   # ws id -> {label: first decoded frame}

    def log(kind, **kv):
        kv = {"t": round(time.time() - t0, 3), "kind": kind, **kv}
        traffic.write(json.dumps(kv, ensure_ascii=False, default=str) + "\n")
        traffic.flush()

    def on_request(req):
        log("http_request", method=req.method, url=req.url, type=req.resource_type,
            headers=req.headers, body=req.post_data, frame=req.frame.url if req.frame else None)

    def on_response(res):
        req = res.request
        rtype = req.resource_type
        ctype = res.headers.get("content-type", "")
        body_text, saved = None, None
        keep = rtype in SOURCE_TYPES or (args.all_assets and rtype in ASSET_TYPES) \
            or ctype.startswith("application/wasm")
        if keep and not (300 <= res.status < 400):
            try:
                body = res.body()
                full, saved = safe_path(src_root, res.url, ctype)
                with open(full, "wb") as f:
                    f.write(body)
                sources.append((rtype, saved, len(body)))
                if any(x in ctype for x in ("json", "text", "javascript", "xml")):
                    body_text = body.decode("utf-8", "replace")
            except Exception as e:
                saved = f"<could not read body: {e}>"
        log("http_response", method=req.method, url=res.url, status=res.status, type=rtype,
            headers=res.headers, saved_as=saved,
            body=(body_text[:MAX_INLINE] if body_text and rtype in ("xhr", "fetch", "other") else None))
        u = urlparse(res.url)
        if rtype in ("xhr", "fetch") and not STATIC_EXT.search(u.path):  # API calls, not game files
            key = f"{req.method} {u.netloc}{u.path}"
            endpoints[key] += 1
            if key not in samples:
                samples[key] = {"request": req.post_data, "status": res.status,
                                "response": (body_text or "")[:3000]}
        print(f"  {res.status} {rtype:10} {res.url[:110]}")

    def on_websocket(ws):
        wid = len(ws_info) + 1
        host = re.sub(r"[^A-Za-z0-9.]+", "_", urlparse(ws.url).netloc)
        path = os.path.join(ws_root, f"{wid:02d}_{host}.jsonl")
        f = open(path, "w", encoding="utf-8")
        info = ws_info[wid] = {"url": ws.url, "file": os.path.relpath(path, out),
                               "sent": 0, "received": 0, "labels": Counter()}
        print(f"  [WS {wid} OPEN] {ws.url}")
        log("ws_open", ws=wid, url=ws.url)

        def frame(direction, payload):
            dec = decode_frame(payload)
            label = frame_label(dec)
            info["sent" if direction == "send" else "received"] += 1
            info["labels"][f"{direction}: {label}"] += 1
            ws_samples[wid].setdefault(f"{direction}: {label}", dec)
            raw = payload if isinstance(payload, str) else base64.b64encode(payload).decode()
            rec = {"t": round(time.time() - t0, 3), "dir": direction, "label": label,
                   "raw": raw, "raw_is_base64": not isinstance(payload, str), "decoded": dec}
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            f.flush()
            log("ws_frame", ws=wid, dir=direction, label=label, decoded=dec)
            print(f"  [WS {wid} {'>>' if direction == 'send' else '<<'}] {label}  "
                  f"{(raw if isinstance(raw, str) else '')[:80]}")

        ws.on("framesent", lambda p: frame("send", p))
        ws.on("framereceived", lambda p: frame("recv", p))
        ws.on("socketerror", lambda e: log("ws_error", ws=wid, error=str(e)))
        ws.on("close", lambda _: (log("ws_close", ws=wid), f.close(), print(f"  [WS {wid} CLOSED]")))

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(record_har_path=os.path.join(out, "session.har"),
                                      record_har_content="embed", viewport=None)
        page = context.new_page()
        context.on("request", on_request)
        context.on("response", on_response)
        page.on("websocket", on_websocket)
        # web workers can open their own sockets; log that they exist
        page.on("worker", lambda w: (log("worker", url=w.url), print(f"  [WORKER] {w.url}")))
        page.on("console", lambda m: log("console", type=m.type, text=m.text))

        print(f"Saving to {out}\nOpening the game...")
        page.goto(args.url, wait_until="domcontentloaded")
        try:
            if args.seconds:
                print(f"Recording for {args.seconds} s - play now.")
                page.wait_for_timeout(args.seconds * 1000)
            else:
                input("\n>>> Play some spins in the browser, then press Enter here to stop...\n")
        except KeyboardInterrupt:
            pass
        context.close()  # writes the HAR
        browser.close()
    traffic.close()

    # ── summary.md ────────────────────────────────────────────────────────────
    L = [f"# Capture summary\n", f"- URL: `{args.url[:120]}...`",
         f"- Duration: {time.time() - t0:.0f} s", f"- Folder: `{out}`\n"]
    L.append("## WebSocket connections\n")
    if not ws_info:
        L.append("_None seen. The game probably uses plain HTTP - see API endpoints below._\n")
    for wid, i in ws_info.items():
        L.append(f"### WS {wid}: `{i['url']}`")
        L.append(f"- frames sent {i['sent']}, received {i['received']}  -> `{i['file']}`\n")
        L.append("| direction: message type | count |\n|---|---|")
        L += [f"| {k} | {v} |" for k, v in i["labels"].most_common()]
        L.append("\n**First example of each message type:**\n")
        for k, dec in ws_samples[wid].items():
            if "ping" in k or "pong" in k:
                continue
            L.append(f"`{k}`\n```json\n{json.dumps(dec, indent=2, ensure_ascii=False, default=str)[:2500]}\n```")
    L.append("\n## API endpoints (XHR / fetch)\n")
    L.append("| endpoint | calls |\n|---|---|")
    L += [f"| `{k}` | {v} |" for k, v in endpoints.most_common()]
    for k, s in samples.items():
        L.append(f"\n### `{k}`  (status {s['status']})")
        if s["request"]:
            L.append(f"Request body:\n```\n{s['request'][:2000]}\n```")
        L.append(f"Response body:\n```\n{s['response']}\n```")
    L.append("\n## Saved sources\n")
    L.append("| type | file | bytes |\n|---|---|---|")
    L += [f"| {t} | `sources/{r}` | {n:,} |" for t, r, n in sorted(sources)]
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"\nDone. {len(sources)} files saved, {len(ws_info)} WebSocket(s), "
          f"{sum(endpoints.values())} API calls.\nOpen {os.path.join(out, 'summary.md')}")


if __name__ == "__main__":
    main()
