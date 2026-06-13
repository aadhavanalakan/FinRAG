"""
retrieval.py — hybrid retrieval: dense (Pinecone) + BM25 (keyword), fused via RRF.

Dense catches paraphrases ("capex" ~ "capital expenditure"); BM25 nails exact
tokens ("FY2025", a literal dollar figure). Reciprocal Rank Fusion merges the two
ranked lists without tuning score scales.

BM25 reads the chunk text written to .cache/corpus/<strategy>.jsonl at ingest, and
is built ONCE per strategy (cached), not per query.

Returns the fused top-k; reranking (P3 next) narrows it to the final few.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from finrag.config import AppConfig
from finrag.embedding import Embedder
from finrag.vectorstore import VectorStore

CORPUS_DIR = Path(".cache/corpus")
_TOKEN = re.compile(r"\b\w+\b")


def _tok(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class Retrieved:
    id: str
    text: str
    company: str
    section: str
    is_table: bool
    score: float                 # fused RRF score
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rerank_score: float | None = None   # set by the reranker (P3)


class _BM25Corpus:
    """BM25 over one strategy's chunk texts, loaded from the ingest-time corpus."""

    def __init__(self, strategy: str) -> None:
        path = CORPUS_DIR / f"{strategy}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"BM25 corpus missing: {path}. Run `python -m scripts.ingest` first."
            )
        self.records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.bm25 = BM25Okapi([_tok(r["text"]) for r in self.records])

    def search(self, query: str, top_k: int, company: str | None) -> list[dict]:
        scores = self.bm25.get_scores(_tok(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: list[dict] = []
        for i in order:
            r = self.records[i]
            if company and r["company"] != company:
                continue
            out.append(r)
            if len(out) >= top_k:
                break
        return out


class HybridRetriever:
    def __init__(self, cfg: AppConfig, embedder: Embedder, store: VectorStore) -> None:
        self.cfg = cfg
        self.r = cfg.pipeline.retrieval
        self.embedder = embedder
        self.store = store
        self._bm25_cache: dict[str, _BM25Corpus] = {}

    def _bm25(self, strategy: str) -> _BM25Corpus:
        if strategy not in self._bm25_cache:
            self._bm25_cache[strategy] = _BM25Corpus(strategy)
        return self._bm25_cache[strategy]

    def _dense(self, query: str, strategy: str, company: str | None) -> list[dict]:
        qv = self.embedder.embed_query(query)
        flt = {"company": company} if company else None
        return self.store.query(qv, top_k=self.r.dense_top_k, strategy=strategy, metadata_filter=flt)

    def retrieve(self, query: str, strategy: str, company: str | None = None) -> list[Retrieved]:
        dense = self._dense(query, strategy, company)
        bm25 = self._bm25(strategy).search(query, self.r.bm25_top_k, company)
        k = self.r.rrf_k

        fused: dict[str, dict] = {}

        def slot(cid: str) -> dict:
            return fused.setdefault(cid, {
                "score": 0.0, "text": None, "company": None,
                "section": "", "is_table": False, "dense_rank": None, "bm25_rank": None,
            })

        for rank, m in enumerate(dense):
            md = m["metadata"]
            e = slot(m["id"])
            e["score"] += 1.0 / (k + rank + 1)
            e["dense_rank"] = rank
            e.update(text=md["text"], company=md["company"],
                     section=md.get("section", ""), is_table=md.get("is_table", False))

        for rank, r in enumerate(bm25):
            e = slot(r["id"])
            e["score"] += 1.0 / (k + rank + 1)
            e["bm25_rank"] = rank
            if e["text"] is None:                     # only dense-missed docs need filling
                e.update(text=r["text"], company=r["company"],
                         section=r["section"], is_table=r["is_table"])

        ranked = sorted(fused.items(), key=lambda kv: kv[1]["score"], reverse=True)
        return [
            Retrieved(id=cid, text=e["text"], company=e["company"], section=e["section"],
                      is_table=e["is_table"], score=e["score"],
                      dense_rank=e["dense_rank"], bm25_rank=e["bm25_rank"])
            for cid, e in ranked[: self.r.fused_top_k]
        ]


if __name__ == "__main__":
    from finrag.config import load_config

    cfg = load_config()
    emb = Embedder(cfg.models.embedding)
    retr = HybridRetriever(cfg, emb, VectorStore(cfg))

    query = "What were Verizon's total consolidated operating revenues in 2025?"
    print(f"QUERY: {query}\n")
    for strat in ("fixed", "semantic"):
        print(f"=== {strat} (fused top 5) ===")
        for i, r in enumerate(retr.retrieve(query, strat, company="verizon")[:5]):
            src = f"dense#{r.dense_rank}" if r.dense_rank is not None else ""
            src += f" bm25#{r.bm25_rank}" if r.bm25_rank is not None else ""
            print(f"  {i+1}. {r.id} score={r.score:.4f} table={r.is_table} [{src.strip()}]")
            print(f"     {r.text[:110].replace(chr(10),' ')}")
        print()
    emb.close()
