"""
run_eval.py — run the 3 experiment arms over the golden set and compare.

Arms (only chunking strategy + rerank change; everything else is identical):
  1. fixed     + rerank
  2. semantic  + rerank      <- expected winner
  3. semantic  (no rerank)   <- isolates the rerank lift

This module scores RETRIEVAL (recall/MRR/nDCG) — no LLM, no cost. Faithfulness +
citation coverage (LLM judge) are added in run_eval_generation (next).

Usage:  python -m eval.run_eval
"""

from __future__ import annotations

import json
from pathlib import Path

from finrag.config import load_config
from finrag.embedding import Embedder
from finrag.reranking import Reranker
from finrag.retrieval import HybridRetriever
from finrag.vectorstore import VectorStore
from eval.metrics import aggregate, score_one

GOLDEN = Path("data/golden/golden.jsonl")
RESULTS_DIR = Path("results")
ARMS = [
    ("fixed_norerank", "fixed", False),       # added: clean chunking comparison vs semantic_norerank
    ("fixed_rerank", "fixed", True),
    ("semantic_norerank", "semantic", False),
    ("semantic_rerank", "semantic", True),
]


def load_golden() -> list[dict]:
    return [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l.strip()]


def final_chunks(retriever, reranker, cfg, item, strategy, use_reranker):
    fused = retriever.retrieve(item["question"], strategy, item["company"])
    if use_reranker:
        return reranker.rerank(item["question"], fused)
    return fused[: cfg.pipeline.reranking.output_top_n]


def main() -> None:
    cfg = load_config()
    print("RUN MANIFEST"); print(cfg.manifest.summary(), "\n")

    gold = load_golden()
    answerable = [g for g in gold if g["answerable"]]
    print(f"Golden set: {len(gold)} items ({len(answerable)} answerable, "
          f"{len(gold) - len(answerable)} negative)\n")

    emb = Embedder(cfg.models.embedding)
    retriever = HybridRetriever(cfg, emb, VectorStore(cfg))
    reranker = Reranker(cfg)

    results = {}
    for arm_name, strategy, use_reranker in ARMS:
        rows = []
        for item in answerable:
            chunks = final_chunks(retriever, reranker, cfg, item, strategy, use_reranker)
            rows.append(score_one([c.text for c in chunks], item["answer_keys"], k=5))
        results[arm_name] = aggregate(rows)

    # comparison table
    cols = ["hit@1", "hit@3", "hit@5", "mrr", "ndcg@5"]
    print(f"{'arm':20} " + " ".join(f"{c:>8}" for c in cols))
    print("-" * (20 + 9 * len(cols)))
    for arm_name, _, _ in ARMS:
        agg = results[arm_name]
        print(f"{arm_name:20} " + " ".join(f"{agg[c]:>8.3f}" for c in cols))

    print("\nReading it:")
    print("  semantic_norerank vs fixed_norerank  -> the CHUNKING gain (clean)")
    print("  *_rerank vs *_norerank               -> the RERANK effect")

    # Persist version-stamped results for the report.
    RESULTS_DIR.mkdir(exist_ok=True)
    out = {
        "stage": "retrieval", "run_id": cfg.manifest.run_id,
        "versions": cfg.manifest.versions, "k": 5,
        "n_answerable": len(answerable),
        "reranker_model": cfg.models.reranker.model,
        "arms": results,
    }
    path = RESULTS_DIR / f"retrieval_{cfg.manifest.run_id}.json"
    path.write_text(json.dumps(out, indent=2))
    (RESULTS_DIR / "retrieval_latest.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {path}")
    emb.close()


if __name__ == "__main__":
    main()
