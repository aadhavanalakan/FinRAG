"""
rerank_study.py — why the reranker hurts, and how to fix it.

Baseline finding: the MS-MARCO cross-encoder demotes table chunks, dropping answer
accuracy. This script tests mitigations on the SEMANTIC arm using the free retrieval
metrics (no LLM), so we can iterate cheaply:

  norerank         retrieval order only (the current best)
  msmarco          plain cross-encoder rerank (the baseline that hurts)
  msmarco+caption  prepend "<company> <section> financial table." to TABLE chunks when scoring
  msmarco+blend    blend normalized rerank score with the retrieval (RRF) score
  bge              a different reranker model (bge-reranker-base)
  bge+caption      bge with the table caption

Usage:  python -m eval.rerank_study
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from finrag.config import load_config
from finrag.embedding import Embedder
from finrag.retrieval import HybridRetriever, Retrieved
from finrag.vectorstore import VectorStore
from eval.metrics import aggregate, score_one

GOLDEN = Path("data/golden/golden.jsonl")
_MODELS: dict[str, object] = {}


def _model(name: str):
    if name not in _MODELS:
        from sentence_transformers import CrossEncoder
        _MODELS[name] = CrossEncoder(name)
    return _MODELS[name]


def _text_for(r: Retrieved, caption: bool) -> str:
    if caption and r.is_table:
        return f"{r.company} {r.section} financial table. {r.text}"
    return r.text


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def rerank(query, fused, model_name, *, caption=False, blend_alpha=1.0, top_n=5):
    cand = fused[:50]
    if not cand:
        return []
    rr = np.asarray(_model(model_name).predict([(query, _text_for(r, caption)) for r in cand]))
    if blend_alpha < 1.0:
        rrf = np.asarray([r.score for r in cand])
        final = blend_alpha * _minmax(rr) + (1.0 - blend_alpha) * _minmax(rrf)
    else:
        final = rr
    order = np.argsort(-final)
    return [cand[i] for i in order[:top_n]]


def main() -> None:
    cfg = load_config()
    gold = [json.loads(l) for l in GOLDEN.read_text().splitlines() if l.strip()]
    answerable = [g for g in gold if g["answerable"]]

    emb = Embedder(cfg.models.embedding)
    retriever = HybridRetriever(cfg, emb, VectorStore(cfg))

    # Fetch the fused candidate list ONCE per question (semantic arm), reuse for all variants.
    fused_by_q = {g["id"]: retriever.retrieve(g["question"], "semantic", g["company"]) for g in answerable}

    MS = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    BGE = "BAAI/bge-reranker-base"
    variants = {
        "norerank":        lambda q, f: f[:5],
        "msmarco":         lambda q, f: rerank(q, f, MS),
        "msmarco+caption": lambda q, f: rerank(q, f, MS, caption=True),
        "msmarco+blend":   lambda q, f: rerank(q, f, MS, blend_alpha=0.5),
        "bge":             lambda q, f: rerank(q, f, BGE),
        "bge+caption":     lambda q, f: rerank(q, f, BGE, caption=True),
    }

    cols = ["hit@1", "hit@3", "hit@5", "mrr", "ndcg@5"]
    print(f"SEMANTIC arm, {len(answerable)} answerable questions\n")
    print(f"{'variant':18} " + " ".join(f"{c:>8}" for c in cols))
    print("-" * (18 + 9 * len(cols)))
    for name, fn in variants.items():
        rows = []
        for g in answerable:
            top = fn(g["question"], fused_by_q[g["id"]])
            rows.append(score_one([r.text for r in top], g["answer_keys"], k=5))
        agg = aggregate(rows)
        print(f"{name:18} " + " ".join(f"{agg[c]:>8.3f}" for c in cols))
    emb.close()


if __name__ == "__main__":
    main()
