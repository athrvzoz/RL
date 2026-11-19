"""
Print a booklet HTML file to PDF with headless Microsoft Edge, then check the result.

Usage:
    python build_pdf.py booklet.html D:\\Langchain\\RL_Guides\\03_DQN.pdf
    python build_pdf.py booklet.html out.pdf --pages 3,7      # also save those pages as PNGs to look at

Prints the page count and how much text is on each page (pages with very little
text are flagged: usually a figure or table that got pushed onto its own page).
Saves  <html folder>/_check/contact.png  (all pages, small) for a quick visual check.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

EDGE_PATHS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("html")
    parser.add_argument("pdf")
    parser.add_argument("--pages", default="", help="comma-separated 1-based page numbers to render as PNG")
    args = parser.parse_args()

    html = Path(args.html).resolve()
    pdf = Path(args.pdf).resolve()
    pdf.parent.mkdir(parents=True, exist_ok=True)
    edge = next(p for p in EDGE_PATHS if os.path.exists(p))
    subprocess.run([edge, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", html.as_uri()],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180, check=True)

    import pypdfium2 as pdfium
    from PIL import Image

    doc = pdfium.PdfDocument(str(pdf))
    print(f"{pdf}  ·  {len(doc)} pages  ·  {pdf.stat().st_size // 1024} KB")
    for i in range(len(doc)):
        text = doc[i].get_textpage().get_text_range().strip()
        first = " ".join(text.split())[:60]
        flag = "   <-- nearly empty?" if len(text) < 300 else ""
        print(f"  page {i + 1:>2}: {len(text):>5} chars  {first}{flag}")

    out = html.parent / "_check"
    out.mkdir(exist_ok=True)
    thumbs = [doc[i].render(scale=0.45).to_pil() for i in range(len(doc))]
    w, h = thumbs[0].size
    cols = 6
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (w * cols, h * rows), "white")
    for k, im in enumerate(thumbs):
        sheet.paste(im, ((k % cols) * w, (k // cols) * h))
    sheet.save(out / "contact.png")
    print(f"contact sheet: {out / 'contact.png'}")
    for n in filter(None, args.pages.split(",")):
        p = out / f"page{int(n)}.png"
        doc[int(n) - 1].render(scale=1.2).to_pil().save(p)
        print(f"page image: {p}")


if __name__ == "__main__":
    main()
