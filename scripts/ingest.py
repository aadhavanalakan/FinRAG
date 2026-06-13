"""
ingest.py — build the Pinecone index from the registered 10-K filings.

Pipeline (once): parse -> chunk (fixed AND semantic) -> embed -> upsert. Each strategy
goes to its own Pinecone namespace; the companies come from the registry
(data/companies.json), so this stays in sync with `add_company` / `remove_company`.

Usage (after PINECONE_API_KEY + NEBIUS_API_KEY are in .env):
  python -m scripts.ingest                 # all registered companies, both strategies
  python -m scripts.ingest --company verizon
  python -m scripts.ingest --keep          # don't clear namespaces first
"""

from __future__ import annotations

import argparse

from finrag.config import load_config
from finrag.corpus import ingest_company, load_registry
from finrag.embedding import Embedder
from finrag.vectorstore import VectorStore


def main() -> None:
    reg = load_registry()
    ap = argparse.ArgumentParser()
    ap.add_argument("--company", choices=list(reg), help="ingest just one company")
    ap.add_argument("--keep", action="store_true", help="do not clear namespaces first")
    args = ap.parse_args()

    cfg = load_config()
    print("RUN MANIFEST"); print(cfg.manifest.summary(), "\n")

    embedder = Embedder(cfg.models.embedding)
    store = VectorStore(cfg)
    store.ensure_index()
    print(f"Index '{cfg.pipeline.vectorstore.index_name}' ready "
          f"({cfg.models.embedding.dimension}-dim, {cfg.pipeline.vectorstore.metric}).")

    companies = [args.company] if args.company else list(reg)
    # A full rebuild clears both namespaces first; a single-company run does not
    # (ingest_company clears just that company's old ids).
    if not args.keep and not args.company:
        for strat in ("fixed", "semantic"):
            store.delete_namespace(strat)
        print("Cleared 'fixed' and 'semantic' namespaces for a clean rebuild.\n")

    for symbol in companies:
        stats = ingest_company(symbol, cfg=cfg, embedder=embedder, store=store)
        for strat in ("fixed", "semantic"):
            s = stats[strat]
            print(f"  {symbol:8} {strat:9} {s['chunks']:4} chunks  "
                  f"(tables={s['tables']}, avg_chars={s['avg_chars']})")

    print("\nIndex stats:")
    st = store.stats()
    print(f"  total vectors: {st.get('total_vector_count')}")
    for ns, info in (st.get("namespaces") or {}).items():
        print(f"    namespace '{ns}': {info.get('vector_count')} vectors")
    embedder.close()


if __name__ == "__main__":
    main()
