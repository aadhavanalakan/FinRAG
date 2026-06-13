"""
metrics.py — retrieval metrics with CONTENT-BASED relevance (no LLM).

A retrieved chunk is "relevant" to a question if its text contains one of the
question's verified answer keys (e.g. "138,191"). This is strategy-agnostic and
survives re-chunking, unlike pinning specific chunk ids.

Computed over the FINAL ranked chunks the LLM would see (top-n after rerank):
  - hit@k   : did a relevant chunk land in the top-k? (binary success rate)
  - MRR     : 1 / rank of the first relevant chunk
  - nDCG@k  : graded ordering quality (binary gains)
"""

from __future__ import annotations

import math
from typing import Sequence


def relevant_flags(chunk_texts: Sequence[str], answer_keys: Sequence[str]) -> list[bool]:
    """True at each rank where the chunk text contains any answer key."""
    return [any(k in t for k in answer_keys) for t in chunk_texts]


def hit_at_k(flags: Sequence[bool], k: int) -> float:
    return 1.0 if any(flags[:k]) else 0.0


def mrr(flags: Sequence[bool]) -> float:
    for i, f in enumerate(flags):
        if f:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(flags: Sequence[bool], k: int) -> float:
    """Binary-gain nDCG. Ideal = the same number of relevant items ranked at the top."""
    top = flags[:k]
    dcg = sum(1.0 / math.log2(i + 2) for i, f in enumerate(top) if f)
    num_rel = sum(top)
    if num_rel == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(num_rel))
    return dcg / idcg


def score_one(chunk_texts: Sequence[str], answer_keys: Sequence[str], k: int = 5) -> dict:
    flags = relevant_flags(chunk_texts, answer_keys)
    return {
        "hit@1": hit_at_k(flags, 1),
        "hit@3": hit_at_k(flags, 3),
        "hit@5": hit_at_k(flags, 5),
        "mrr": mrr(flags),
        f"ndcg@{k}": ndcg_at_k(flags, k),
    }


def aggregate(rows: list[dict]) -> dict:
    """Mean of each metric across questions."""
    if not rows:
        return {}
    keys = rows[0].keys()
    return {m: sum(r[m] for r in rows) / len(rows) for m in keys}
