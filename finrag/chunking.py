"""
chunking.py — the two chunking strategies being compared.

Both consume a ParsedDocument (ordered prose + table blocks from parsing.py) and
emit a list of `Chunk`s with metadata. They differ in how they treat tables —
which is the whole experiment:

  - FIXED (protect_tables: false)  -> flattens everything, including table Markdown,
    and slices every 512 chars. Tables get shredded mid-row. The naive baseline.
  - SEMANTIC (protect_tables: true) -> keeps each table as one atomic chunk and cuts
    prose at topic boundaries. (built next — needs the Nebius embedder)

`protect_tables` is a per-strategy flag, so flipping it gives the "fair" comparison
(fixed WITH protection) for free — no code change.

Smoke test:  python -m finrag.chunking
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from finrag.config import (
    AppConfig,
    FixedChunkingConfig,
    SemanticChunkingConfig,
    SharedChunkingConfig,
    load_config,
)
from finrag.parsing import ParsedDocument, parse_document

if TYPE_CHECKING:
    from finrag.embedding import Embedder


# =============================================================================
# data model
# =============================================================================
@dataclass(frozen=True)
class Chunk:
    id: str                 # e.g. "verizon-0042" — what the LLM cites
    text: str               # content to embed + store
    company: str
    fiscal_year: int
    section: str | None     # 10-K section ("Item 7"), best-effort
    strategy: str           # "fixed" | "semantic"
    is_table: bool          # True if this chunk is a preserved table
    char_count: int

    def metadata(self) -> dict:
        """Flat metadata dict for the vector store (no `text` — stored separately)."""
        return {
            "company": self.company,
            "fiscal_year": self.fiscal_year,
            "section": self.section or "",
            "strategy": self.strategy,
            "is_table": self.is_table,
            "char_count": self.char_count,
        }


# =============================================================================
# shared helpers
# =============================================================================
def _alpha_ratio(text: str) -> float:
    """Fraction of non-space characters that are letters."""
    stripped = [c for c in text if not c.isspace()]
    if not stripped:
        return 0.0
    return sum(c.isalpha() for c in stripped) / len(stripped)


def _passes_filter(text: str, shared: SharedChunkingConfig) -> bool:
    """Drop junk: too short, or mostly digits/symbols (page numbers, stray fragments)."""
    return len(text) >= shared.min_chunk_chars and _alpha_ratio(text) >= shared.min_alpha_ratio


def _strip_markers(content: str, shared: SharedChunkingConfig) -> str:
    """Remove the parser's table markers — stored chunk text is clean Markdown."""
    return content.replace(shared.table_open_marker, "").replace(shared.table_close_marker, "").strip()


def _split_table_by_rows(md: str, max_chars: int) -> list[str]:
    """
    Secondary split for an oversized table (Pinecone metadata cap). Split by ROWS,
    repeating the header in each piece, so we never cut mid-row.
    """
    lines = md.split("\n")
    if len(md) <= max_chars or len(lines) < 3:
        return [md]
    header = lines[:2]                      # header row + the '| --- |' separator
    header_len = len("\n".join(header)) + 1
    parts: list[str] = []
    cur, cur_len = list(header), header_len
    for row in lines[2:]:
        if cur_len + len(row) + 1 > max_chars and len(cur) > 2:
            parts.append("\n".join(cur))
            cur, cur_len = list(header), header_len
        cur.append(row)
        cur_len += len(row) + 1
    if len(cur) > 2:
        parts.append("\n".join(cur))
    return parts


# =============================================================================
# fixed-size chunker
# =============================================================================
def chunk_fixed(
    doc: ParsedDocument,
    fixed: FixedChunkingConfig,
    shared: SharedChunkingConfig,
) -> list[Chunk]:
    """
    Slice the document into fixed-size pieces.

    With protect_tables=False (the baseline), table Markdown is flattened into the
    prose stream and sliced blindly — so tables get cut mid-row. With it True, tables
    are emitted atomically and only prose is sliced (the optional "fair" arm).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=fixed.chunk_size,
        chunk_overlap=fixed.chunk_overlap,
        separators=list(fixed.separators),
    )

    chunks: list[Chunk] = []
    n = 0
    prose_buf: list[str] = []
    cur_section: str | None = None

    def make(text: str, section: str | None, is_table: bool) -> None:
        nonlocal n
        text = text.strip()
        if not is_table and not _passes_filter(text, shared):
            return                          # tables are exempt from the junk filter
        n += 1
        chunks.append(Chunk(
            id=f"{doc.company}-{n:04d}",
            text=text, company=doc.company, fiscal_year=doc.fiscal_year,
            section=section, strategy="fixed", is_table=is_table, char_count=len(text),
        ))

    def flush_prose(section: str | None) -> None:
        if not prose_buf:
            return
        for piece in splitter.split_text("\n\n".join(prose_buf)):
            make(piece, section, is_table=False)
        prose_buf.clear()

    for b in doc.blocks:
        if b.section != cur_section:
            flush_prose(cur_section)
            cur_section = b.section
        if b.is_table and fixed.protect_tables:
            flush_prose(cur_section)
            md = _strip_markers(b.content, shared)
            for part in _split_table_by_rows(md, shared.max_chunk_chars):
                make(part, b.section, is_table=True)
        else:
            # prose, OR a table we're deliberately shredding (markers removed)
            prose_buf.append(_strip_markers(b.content, shared) if b.is_table else b.content)
    flush_prose(cur_section)
    return chunks


# =============================================================================
# semantic chunker
# =============================================================================
def _split_sentences(text: str) -> list[str]:
    """Lightweight regex sentence splitter (no nltk). Decimals like 3.8 don't split."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _enforce_max(text: str, max_chars: int) -> list[str]:
    """Keep a piece under max_chars by packing whole sentences; hard-split as last resort."""
    if len(text) <= max_chars:
        return [text]
    pieces, cur = [], ""
    for s in _split_sentences(text):
        if cur and len(cur) + len(s) + 1 > max_chars:
            pieces.append(cur)
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        pieces.append(cur)
    out: list[str] = []
    for p in pieces:
        out.extend([p] if len(p) <= max_chars else [p[i : i + max_chars] for i in range(0, len(p), max_chars)])
    return out


def _merge_small(groups: list[str], min_chars: int) -> list[str]:
    """Fold under-sized semantic groups into the previous chunk to avoid tiny fragments."""
    out: list[str] = []
    for g in groups:
        if out and len(g) < min_chars:
            out[-1] = out[-1] + " " + g
        else:
            out.append(g)
    return out


def chunk_semantic(
    doc: ParsedDocument,
    semantic: SemanticChunkingConfig,
    shared: SharedChunkingConfig,
    embedder: "Embedder",
    percentile: int | None = None,
) -> list[Chunk]:
    """
    Cut prose at topic boundaries; keep each table whole (protect_tables=True).

    Per prose run: embed each sentence, take the cosine DISTANCE between neighbors,
    and start a new chunk wherever that distance exceeds the `percentile` threshold.
    Embeddings are cached, so sweeping percentiles re-embeds nothing.
    """
    pct = percentile if percentile is not None else semantic.breakpoint_percentile
    chunks: list[Chunk] = []
    n = 0
    run_parts: list[str] = []
    run_section: str | None = None

    def make(text: str, section: str | None, is_table: bool) -> None:
        nonlocal n
        text = text.strip()
        if not is_table and not _passes_filter(text, shared):
            return
        n += 1
        chunks.append(Chunk(
            id=f"{doc.company}-{n:04d}",
            text=text, company=doc.company, fiscal_year=doc.fiscal_year,
            section=section, strategy="semantic", is_table=is_table, char_count=len(text),
        ))

    def flush_run(section: str | None) -> None:
        nonlocal run_parts
        if not run_parts:
            return
        run_text = "\n".join(run_parts)
        run_parts = []
        sents = _split_sentences(run_text)
        if len(sents) <= 1:                          # nothing to segment
            for piece in _enforce_max(run_text, shared.max_chunk_chars):
                make(piece, section, is_table=False)
            return
        vecs = embedder.embed_documents(sents)        # (n, dim), L2-normalized
        dists = 1.0 - np.sum(vecs[:-1] * vecs[1:], axis=1)   # cosine distance, neighbors
        threshold = float(np.percentile(dists, pct))

        groups: list[str] = []
        cur = [sents[0]]
        for i, d in enumerate(dists):
            if d > threshold:                         # topic boundary -> cut
                groups.append(" ".join(cur))
                cur = [sents[i + 1]]
            else:
                cur.append(sents[i + 1])
        groups.append(" ".join(cur))

        for g in _merge_small(groups, semantic.min_chunk_chars):
            for piece in _enforce_max(g, shared.max_chunk_chars):
                make(piece, section, is_table=False)

    for b in doc.blocks:
        if b.section != run_section:
            flush_run(run_section)
            run_section = b.section
        if b.is_table and semantic.protect_tables:
            flush_run(run_section)
            md = _strip_markers(b.content, shared)
            for part in _split_table_by_rows(md, shared.max_chunk_chars):
                make(part, b.section, is_table=True)
        else:
            run_parts.append(_strip_markers(b.content, shared) if b.is_table else b.content)
    flush_run(run_section)
    return chunks


def semantic_sweep(
    doc: ParsedDocument,
    semantic: SemanticChunkingConfig,
    shared: SharedChunkingConfig,
    embedder: "Embedder",
) -> dict[int, dict]:
    """Run chunk_semantic at every percentile in the sweep; return {percentile: stats}.
    Embeddings are cached after the first percentile, so this is cheap."""
    return {
        pct: chunk_stats(chunk_semantic(doc, semantic, shared, embedder, percentile=pct))
        for pct in semantic.breakpoint_sweep
    }


def chunk_stats(chunks: list[Chunk]) -> dict:
    if not chunks:
        return {"chunks": 0}
    lengths = [c.char_count for c in chunks]
    return {
        "chunks": len(chunks),
        "tables": sum(c.is_table for c in chunks),
        "avg_chars": sum(lengths) // len(lengths),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
    }


if __name__ == "__main__":
    from finrag.embedding import Embedder

    cfg: AppConfig = load_config()
    doc = parse_document("data/verizon_10k.html")

    fixed_chunks = chunk_fixed(doc, cfg.chunking.fixed, cfg.chunking.shared)
    print("FIXED   :", chunk_stats(fixed_chunks))

    embedder = Embedder(cfg.models.embedding)
    sem_chunks = chunk_semantic(doc, cfg.chunking.semantic, cfg.chunking.shared, embedder)
    print("SEMANTIC:", chunk_stats(sem_chunks))

    # KEY CHECK: the operating-revenues table must survive as ONE intact chunk.
    whole = [c for c in sem_chunks if c.is_table and "Consolidated Operating Revenues" in c.text]
    print(f"\nOperating-revenues table kept whole as a single chunk? {'YES' if whole else 'NO'}")
    if whole:
        print(f"--- {whole[0].id} (section {whole[0].section}, table, {whole[0].char_count} chars) ---")
        print(whole[0].text[:400])

    print("\nThreshold SWEEP (pick the winner on this corpus):")
    print(f"  {'pct':>4} {'chunks':>7} {'tables':>7} {'avg_chars':>10}")
    for pct, st in semantic_sweep(doc, cfg.chunking.semantic, cfg.chunking.shared, embedder).items():
        print(f"  {pct:>4} {st['chunks']:>7} {st['tables']:>7} {st['avg_chars']:>10}")
    embedder.close()
