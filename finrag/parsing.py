"""
parsing.py — table-aware parser for SEC 10-K (inline-XBRL) HTML filings.

The ONE rule of this project: do NOT destroy tables. So instead of stripping all
HTML, we walk each filing IN ORDER and emit a list of `Block`s:

  - prose blocks  -> clean, whitespace-normalized text, tagged with their 10-K
                     section ("Item 7", ...).
  - table blocks  -> the table rendered as a Markdown table, wrapped in protective
                     markers (<!--TABLE-->...<!--/TABLE-->) so neither chunker can
                     split it mid-row.

The filings are Workiva-generated inline XBRL: numbers are wrapped in <ix:...> tags,
but the *displayed* text already contains the human-readable figure, so we just read
the rendered text — no XBRL decoding needed.

Smoke test:  python -m finrag.parsing
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, XMLParsedAsHTMLWarning

# These files open with `<?xml ...?>`, so bs4 warns about using an HTML parser on
# XML. We deliberately want HTML parsing (lenient, recovers the rendered text).
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Default protective markers — mirror config/chunking.yaml `shared`. Passed in by
# the pipeline; defaulted here so the parser is usable standalone and in tests.
DEFAULT_TABLE_OPEN = "<!--TABLE-->"
DEFAULT_TABLE_CLOSE = "<!--/TABLE-->"

# Known filenames -> company id.
_COMPANY_BY_STEM = {
    "verizon_10k": "verizon",
    "att_10k": "att",
    "tmobile_10k": "tmobile",
}

_SECTION_RE = re.compile(r"^\s*Item\s+(\d+[A-Z]?)\.", re.IGNORECASE)
_PLACEHOLDER_RE = re.compile(r"@@TABLE_(\d+)@@")


# =============================================================================
# data model
# =============================================================================
@dataclass(frozen=True)
class Block:
    kind: str               # "prose" | "table"
    section: str | None     # e.g. "Item 7" — best-effort, may be None before the first heading
    content: str            # prose text, OR markdown table WITH protective markers

    @property
    def is_table(self) -> bool:
        return self.kind == "table"


@dataclass(frozen=True)
class ParsedDocument:
    company: str
    fiscal_year: int
    source_file: str
    blocks: list[Block]

    def tables(self) -> list[Block]:
        return [b for b in self.blocks if b.is_table]

    def prose(self) -> list[Block]:
        return [b for b in self.blocks if not b.is_table]

    def to_text(self) -> str:
        """Whole document as one ordered string (tables inline as marked Markdown)."""
        return "\n\n".join(b.content for b in self.blocks)

    def stats(self) -> dict:
        return {
            "company": self.company,
            "fiscal_year": self.fiscal_year,
            "blocks": len(self.blocks),
            "prose_blocks": len(self.prose()),
            "table_blocks": len(self.tables()),
            "sections": sorted({b.section for b in self.blocks if b.section}),
        }


# =============================================================================
# text + cell normalization
# =============================================================================
def _norm(text: str) -> str:
    """Replace non-breaking/thin spaces and collapse runs of whitespace."""
    text = text.replace("\xa0", " ").replace(" ", " ").replace("​", "")
    return re.sub(r"[ \t]+", " ", text).strip()


def _merge_symbols(cells: list[str]) -> list[str]:
    """
    Financial tables put '$', '%', and parentheses in their own cells. Merge them
    back onto the adjacent number so a value stays whole and readable:
      ['Consumer','$','106,807']      -> ['Consumer','$106,807']
      ['(', '462', ')']               -> ['(462)']
      ['1.6','%']                     -> ['1.6%']
    Empty cells are dropped.
    """
    out: list[str] = []
    prefix = ""
    for raw in cells:
        c = _norm(raw)
        if c == "":
            continue
        if c == "$":
            prefix += "$"
            continue
        if c == "(":
            prefix += "("
            continue
        if c == ")":
            if out:
                out[-1] += ")"
            continue
        if c == "%":
            if out:
                out[-1] += "%"
            continue
        out.append(prefix + c)
        prefix = ""
    return out


# =============================================================================
# table handling
# =============================================================================
def _row_cells(tr) -> list[str]:
    return _merge_symbols([td.get_text() for td in tr.find_all(["td", "th"])])


def _is_data_table(table, min_cells: int) -> bool:
    """
    A 'data' table has real content: >=2 rows, >=min_cells nonempty cells, and at
    least one digit. Everything else (spacer/layout tables, horizontal rules) is
    not a data table and gets unwrapped to plain text instead.
    """
    rows = table.find_all("tr")
    if len(rows) < 2:
        return False
    nonempty = 0
    has_digit = False
    for tr in rows:
        for td in tr.find_all(["td", "th"]):
            t = _norm(td.get_text())
            if t:
                nonempty += 1
                if any(ch.isdigit() for ch in t):
                    has_digit = True
    return nonempty >= min_cells and has_digit


def _table_to_markdown(table) -> str | None:
    """Render a data table as a GitHub-flavored Markdown table."""
    rows = [_row_cells(tr) for tr in table.find_all("tr")]
    rows = [r for r in rows if r]            # drop fully-empty rows
    if not rows:
        return None
    ncols = max(len(r) for r in rows)
    rows = [r + [""] * (ncols - len(r)) for r in rows]   # pad ragged rows

    def fmt(row: list[str]) -> str:
        # escape pipes so a stray '|' in text doesn't break the Markdown table
        return "| " + " | ".join(c.replace("|", "\\|") for c in row) + " |"

    header, *body = rows
    lines = [fmt(header), "| " + " | ".join(["---"] * ncols) + " |"]
    lines += [fmt(r) for r in body]
    return "\n".join(lines)


# =============================================================================
# document-level detection
# =============================================================================
def _detect_company(path: Path, override: str | None) -> str:
    if override:
        return override
    return _COMPANY_BY_STEM.get(path.stem, path.stem)


def _detect_fiscal_year(soup: BeautifulSoup, override: int | None) -> int:
    if override:
        return override
    # dei:DocumentPeriodEndDate carries the period end, e.g. "2025-12-31".
    tag = soup.find(attrs={"name": "dei:DocumentPeriodEndDate"})
    if tag:
        m = re.search(r"(\d{4})", tag.get_text())
        if m:
            return int(m.group(1))
    return 2025  # corpus is all FY2025; safe fallback


# =============================================================================
# main entry point
# =============================================================================
def parse_document(
    path: str | Path,
    company: str | None = None,
    fiscal_year: int | None = None,
    *,
    table_open: str = DEFAULT_TABLE_OPEN,
    table_close: str = DEFAULT_TABLE_CLOSE,
    min_table_cells: int = 4,
) -> ParsedDocument:
    """
    Parse a 10-K HTML file into an ordered list of prose + table blocks.

    Strategy: convert each data table to Markdown and swap it for a placeholder
    token in the tree; unwrap non-data tables to plain text; then read the whole
    document as ordered text, tracking the current 'Item N' section, and split it
    back into blocks at the placeholders.
    """
    path = Path(path)
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "lxml")

    comp = _detect_company(path, company)
    fy = _detect_fiscal_year(soup, fiscal_year)

    # Only top-level tables (avoid double-processing nested ones).
    top_tables = [t for t in soup.find_all("table") if t.find_parent("table") is None]
    table_markdowns: list[str] = []
    for t in top_tables:
        if _is_data_table(t, min_table_cells):
            md = _table_to_markdown(t)
            if md:
                idx = len(table_markdowns)
                table_markdowns.append(md)
                t.replace_with(NavigableString(f"\n@@TABLE_{idx}@@\n"))
                continue
        # not a data table -> keep any text, drop the table structure
        t.replace_with(NavigableString(" " + t.get_text(" ") + " "))

    # Linearize the (now table-free) document, preserving order.
    raw_text = soup.get_text("\n")

    blocks: list[Block] = []
    buf: list[str] = []
    cur_section: str | None = None

    def flush() -> None:
        text = "\n".join(buf)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if text:
            blocks.append(Block("prose", cur_section, text))
        buf.clear()

    for line in raw_text.split("\n"):
        line = _norm(line)
        if not line:
            buf.append("")                       # preserve paragraph breaks
            continue
        ph = _PLACEHOLDER_RE.search(line)
        if ph:
            flush()
            md = table_markdowns[int(ph.group(1))]
            blocks.append(Block("table", cur_section, f"{table_open}\n{md}\n{table_close}"))
            continue
        sm = _SECTION_RE.match(line)
        if sm:
            flush()
            cur_section = f"Item {sm.group(1).upper()}"
        buf.append(line)
    flush()

    return ParsedDocument(
        company=comp,
        fiscal_year=fy,
        source_file=str(path),
        blocks=blocks,
    )


if __name__ == "__main__":
    doc = parse_document("data/verizon_10k.html")
    print("PARSED:", doc.stats(), "\n")
    # Show the first table that looks like an operating-revenues table.
    for b in doc.tables():
        if "operating revenues" in b.content.lower():
            print("=== A real financial table, preserved as Markdown ===")
            print(b.content[:900])
            break
