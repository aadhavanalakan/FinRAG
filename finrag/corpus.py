"""
corpus.py — the corpus is DATA, not code: companies live in a registry
(data/companies.json), so adding one is a command, not an edit + full rebuild.

Responsibilities:
  - registry CRUD (load / save / hints for company auto-detection)
  - add_company(ticker)   : validate via EDGAR -> download 10-K -> register -> index
  - ingest_company(symbol): parse -> chunk (both) -> embed -> upsert + sync BM25 corpus
  - remove_company(symbol): delete that company's vectors + BM25 records + files + entry

The three seed telecoms are flagged `"eval": true` and are what the graded eval uses;
companies you add are purely additive (great for the chatbot) and never touch that set.

Vector deletes use the chunk IDs recorded in the BM25 corpus files (Pinecone serverless
has no delete-by-metadata), so those JSONL files are the source of truth for what's indexed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

# Heavy deps (openai / pinecone / requests / chunkers) are imported lazily inside the
# functions that need them, so `import finrag.corpus` stays light — the registry +
# hint helpers (and their unit tests) run with no model/vector-store dependency.
if TYPE_CHECKING:
    from finrag.chunking import Chunk
    from finrag.config import AppConfig
    from finrag.embedding import Embedder
    from finrag.vectorstore import VectorStore

REGISTRY_PATH = Path("data/companies.json")
CORPUS_DIR = Path(".cache/corpus")
STRATEGIES = ("fixed", "semantic")


# =============================================================================
# registry
# =============================================================================
def load_registry() -> dict[str, dict]:
    if not REGISTRY_PATH.exists():
        return {}
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def save_registry(reg: dict[str, dict]) -> None:
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2) + "\n", encoding="utf-8")


def symbols() -> list[str]:
    return sorted(load_registry())


def company_hints() -> dict[str, str]:
    """Lowercase phrase -> symbol, for question auto-detection. Includes the symbol,
    the ticker, and salient words from the legal name (drops Inc./Corp./common stops)."""
    stop = {"inc", "inc.", "corp", "corp.", "corporation", "co", "co.", "company",
            "communications", "us", "u.s.", "the", "ltd", "plc", "group", "holdings"}
    hints: dict[str, str] = {}
    for sym, meta in load_registry().items():
        hints[sym.lower()] = sym
        tk = (meta.get("ticker") or "").lower()
        if len(tk) >= 2:                 # skip 1-letter tickers (e.g. "T") — too ambiguous
            hints[tk] = sym
        for word in (meta.get("name", "")).lower().replace(",", " ").replace(".", " ").split():
            if word and word not in stop and len(word) > 2:
                hints.setdefault(word, sym)
    return hints


# =============================================================================
# BM25 corpus files (the id/text source of truth, per company)
# =============================================================================
def _corpus_path(strategy: str) -> Path:
    return CORPUS_DIR / f"{strategy}.jsonl"


def read_corpus(strategy: str) -> list[dict]:
    p = _corpus_path(strategy)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_corpus(strategy: str, records: list[dict]) -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    with _corpus_path(strategy).open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _records_for(chunks: list[Chunk]) -> list[dict]:
    return [{"id": c.id, "text": c.text, "company": c.company,
             "section": c.section or "", "is_table": c.is_table} for c in chunks]


def set_company_chunks(strategy: str, company: str, chunks: list[Chunk]) -> None:
    """Replace a company's records in the BM25 corpus, leaving other companies intact."""
    others = [r for r in read_corpus(strategy) if r.get("company") != company]
    write_corpus(strategy, others + _records_for(chunks))


def company_ids(strategy: str, company: str) -> list[str]:
    return [r["id"] for r in read_corpus(strategy) if r.get("company") == company]


# =============================================================================
# index operations
# =============================================================================
def ingest_company(
    symbol: str,
    cfg: AppConfig | None = None,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
) -> dict:
    """Parse -> chunk (fixed + semantic) -> embed -> upsert, and sync the BM25 corpus.
    Re-ingesting an existing company first deletes its old vectors (deterministic ids)."""
    from finrag.config import load_config
    cfg = cfg or load_config()
    reg = load_registry()
    if symbol not in reg:
        raise KeyError(f"'{symbol}' is not in the registry.")
    file = reg[symbol]["file"]

    from finrag.chunking import chunk_fixed, chunk_semantic, chunk_stats
    from finrag.embedding import Embedder
    from finrag.parsing import parse_document
    from finrag.vectorstore import VectorStore

    own_emb = embedder is None
    embedder = embedder or Embedder(cfg.models.embedding)
    store = store or VectorStore(cfg)
    store.ensure_index()

    doc = parse_document(file, company=symbol)
    out = {"symbol": symbol, "fiscal_year": doc.fiscal_year}
    chunkers = {
        "fixed": lambda d: chunk_fixed(d, cfg.chunking.fixed, cfg.chunking.shared),
        "semantic": lambda d: chunk_semantic(d, cfg.chunking.semantic, cfg.chunking.shared, embedder),
    }
    for strat, chunker in chunkers.items():
        old = company_ids(strat, symbol)
        if old:
            store.delete_ids(old, strat)                 # clear a previous version first
        chunks = chunker(doc)
        vecs = embedder.embed_documents([c.text for c in chunks])
        store.upsert_chunks(chunks, vecs, strat)
        set_company_chunks(strat, symbol, chunks)
        out[strat] = chunk_stats(chunks)

    # record the detected fiscal year back into the registry
    reg[symbol]["fiscal_year"] = doc.fiscal_year
    save_registry(reg)
    if own_emb:
        embedder.close()
    return out


def add_company(ticker: str, cfg: AppConfig | None = None) -> dict:
    """Validate a ticker against EDGAR, download its latest 10-K, register, and index it."""
    from finrag.config import load_config
    from finrag.edgar import fetch_latest_10k_html
    cfg = cfg or load_config()
    info = fetch_latest_10k_html(ticker)                 # raises EdgarError on bad ticker / no 10-K
    symbol = info["ticker"].lower()
    file = f"data/{symbol}_10k.html"
    Path(file).write_text(info["html"], encoding="utf-8")

    reg = load_registry()
    reg[symbol] = {
        "name": info["title"], "ticker": info["ticker"], "cik": info["cik"],
        "file": file, "filing_date": info["filing_date"], "eval": False,
    }
    save_registry(reg)
    stats = ingest_company(symbol, cfg=cfg)
    return {**stats, "name": info["title"], "filing_date": info["filing_date"], "file": file}


def remove_company(symbol: str, cfg: AppConfig | None = None) -> dict:
    """Delete a company's vectors (both namespaces), BM25 records, HTML file, and entry."""
    from finrag.config import load_config
    from finrag.vectorstore import VectorStore
    cfg = cfg or load_config()
    symbol = symbol.lower()
    reg = load_registry()
    if symbol not in reg:
        raise KeyError(f"'{symbol}' is not in the registry.")
    if reg[symbol].get("eval"):
        raise ValueError(
            f"'{symbol}' is part of the graded eval set and is protected from removal. "
            "Edit data/companies.json directly if you really mean to."
        )

    store = VectorStore(cfg)
    deleted = 0
    for strat in STRATEGIES:
        ids = company_ids(strat, symbol)
        if ids:
            store.delete_ids(ids, strat)
            deleted += len(ids)
        write_corpus(strat, [r for r in read_corpus(strat) if r.get("company") != symbol])

    file = reg[symbol].get("file")
    if file and Path(file).exists():
        Path(file).unlink()
    reg.pop(symbol)
    save_registry(reg)
    return {"symbol": symbol, "deleted_vectors": deleted}


if __name__ == "__main__":
    print("registry:", json.dumps(load_registry(), indent=2))
    print("hints sample:", dict(list(company_hints().items())[:8]))
