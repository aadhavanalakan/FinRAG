"""
make_architecture.py — render the FinRAG architecture diagram to assets/architecture.png.

Pure Pillow (no graphviz/matplotlib). Two lanes — INGEST (one-time) and ASK (per query)
— around the shared vector store. Run:  python -m scripts.make_architecture
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1760, 1020
BG = (13, 17, 23)
CARD = (26, 32, 44)
TEXT = (236, 239, 244)
SUB = (150, 162, 178)
ARROW = (90, 100, 116)
TEAL = (45, 212, 191)     # ingest
AMBER = (245, 158, 11)    # store
ROSE = (244, 63, 94)      # ask
OUT = Path("assets/architecture.png")

_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
]
_BOLD = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
]


def font(sz: int, bold: bool = False):
    for p in (_BOLD if bold else []) + _FONTS:
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            continue
    return ImageFont.load_default()


F_TITLE = font(40, bold=True)
F_LANE = font(20, bold=True)
F_CARD = font(23, bold=True)
F_SUB = font(17)
F_FOOT = font(18)


def text_center(d, cx, y, s, fnt, fill):
    w = d.textlength(s, font=fnt)
    d.text((cx - w / 2, y), s, font=fnt, fill=fill)


def wrap(d, s, fnt, max_w):
    words, lines, cur = s.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=fnt) <= max_w:
            cur = t
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines


def card(d, x, y, w, h, title, sub, accent):
    d.rounded_rectangle((x, y, x + w, y + h), radius=16, fill=CARD, outline=accent, width=3)
    d.rounded_rectangle((x, y, x + 10, y + h), radius=16, fill=accent)   # left accent bar
    cx = x + w / 2 + 4
    text_center(d, cx, y + 16, title, F_CARD, TEXT)
    lines = wrap(d, sub, F_SUB, w - 34)
    for i, ln in enumerate(lines[:3]):
        text_center(d, cx, y + 50 + i * 22, ln, F_SUB, SUB)
    return (x, y, w, h)


def row(d, items, y, h, margin=70, gap=34):
    n = len(items)
    avail = W - 2 * margin - (n - 1) * gap
    w = avail / n
    boxes = []
    for i, (t, s, a) in enumerate(items):
        x = margin + i * (w + gap)
        boxes.append(card(d, x, y, w, h, t, s, a))
    return boxes


def harrow(d, b1, b2):
    x1 = b1[0] + b1[2]; x2 = b2[0]; y = b1[1] + b1[3] / 2
    d.line((x1 + 4, y, x2 - 10, y), fill=ARROW, width=3)
    d.polygon([(x2 - 10, y - 7), (x2 - 10, y + 7), (x2, y)], fill=ARROW)


def varrow(d, x, y1, y2):
    d.line((x, y1, x, y2 - 10), fill=ARROW, width=3)
    d.polygon([(x - 7, y2 - 10), (x + 7, y2 - 10), (x, y2)], fill=ARROW)


def elbow(d, x1, y1, x2, y2):
    ymid = (y1 + y2) / 2
    d.line((x1, y1, x1, ymid), fill=ARROW, width=3)
    d.line((x1, ymid, x2, ymid), fill=ARROW, width=3)
    d.line((x2, ymid, x2, y2 - 10), fill=ARROW, width=3)
    d.polygon([(x2 - 7, y2 - 10), (x2 + 7, y2 - 10), (x2, y2)], fill=ARROW)


def main() -> None:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    text_center(d, W / 2, 34, "FinRAG — System Architecture", F_TITLE, TEXT)
    text_center(d, W / 2, 84, "Grounded, cited Q&A over SEC 10-K filings", F_SUB, SUB)

    # INGEST lane
    d.text((70, 150), "INGEST · one-time", font=F_LANE, fill=TEAL)
    ing = row(d, [
        ("SEC EDGAR", "10-K filings (auto-download)", TEAL),
        ("Parse", "tables → atomic Markdown", TEAL),
        ("Chunk", "fixed  vs  semantic", TEAL),
        ("Embed", "Qwen3-8B · 4096-d · cached", TEAL),
    ], y=185, h=118)
    for a, b in zip(ing, ing[1:]):
        harrow(d, a, b)

    # STORE (center)
    d.text((70, 430), "STORE", font=F_LANE, fill=AMBER)
    sw = 760
    sx = (W - sw) / 2
    store = card(d, sx, 462, sw, 110,
                 "Pinecone (fixed / semantic namespaces)  +  BM25 corpus",
                 "dense vectors + keyword index", AMBER)

    # ASK lane (define first so we can aim the store arrow at the Retrieve box)
    d.text((70, 690), "ASK · per query", font=F_LANE, fill=ROSE)
    ask = row(d, [
        ("Question", "user query", ROSE),
        ("Guardrails", "injection / advice block", ROSE),
        ("Hybrid Retrieve", "dense + BM25 → RRF · coverage", ROSE),
        ("Generate", "Nebius / OpenAI · cited", ROSE),
        ("Audit", "citation check", ROSE),
        ("Answer", "+ live latency / cost", ROSE),
    ], y=720, h=118)
    for a, b in zip(ask, ask[1:]):
        harrow(d, a, b)

    elbow(d, ing[-1][0] + ing[-1][2] / 2, 185 + 118, sx + sw - 90, 462)   # Embed -> Store
    retr = ask[2]
    varrow(d, retr[0] + retr[2] / 2, 462 + 110, 720)                       # Store -> Retrieve

    text_center(d, W / 2, 900,
                "Orchestrated as a LangGraph state machine  ·  query-planning for comparisons  ·  "
                "evaluated with hit@k / MRR / nDCG + RAGAS + an adversarial hard set",
                F_FOOT, SUB)

    OUT.parent.mkdir(exist_ok=True)
    img.save(OUT)
    print(f"Wrote {OUT.resolve()}  ({W}x{H})")


if __name__ == "__main__":
    main()
