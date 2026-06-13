"""
reranking.py — cross-encoder reranking of retrieved candidates.

Hybrid retrieval is fast but coarse (it scores query and chunk separately). A
cross-encoder reads the (query, chunk) pair JOINTLY and scores true relevance,
so it reorders far better — at the cost of being slow, which is why it runs only
on the handful of fused candidates, not the whole index.

Local model (ms-marco-MiniLM, from models.yaml). `enabled` is the knob that
experiment arm 3 turns off to isolate the reranking lift. Loaded lazily so importing
this module is cheap and the model downloads only when first used.
"""

from __future__ import annotations

from finrag.config import AppConfig
from finrag.retrieval import Retrieved


class Reranker:
    def __init__(self, cfg: AppConfig) -> None:
        self.rcfg = cfg.pipeline.reranking
        self.model_name = cfg.models.reranker.model
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, items: list[Retrieved]) -> list[Retrieved]:
        """
        Reorder candidates by cross-encoder relevance and return the top-n.

        If reranking is disabled (arm 3), pass through the fused order, just
        truncated to output_top_n — so downstream code is identical either way.
        """
        if not items:
            return []
        if not self.rcfg.enabled:
            return items[: self.rcfg.output_top_n]

        candidates = items[: self.rcfg.input_top_n]
        scores = self.model.predict([(query, c.text) for c in candidates])
        for c, s in zip(candidates, scores):
            c.rerank_score = float(s)
        candidates.sort(key=lambda c: c.rerank_score, reverse=True)
        return candidates[: self.rcfg.output_top_n]


if __name__ == "__main__":
    from finrag.config import load_config
    from finrag.embedding import Embedder
    from finrag.vectorstore import VectorStore
    from finrag.retrieval import HybridRetriever

    cfg = load_config()
    emb = Embedder(cfg.models.embedding)
    retr = HybridRetriever(cfg, emb, VectorStore(cfg))
    reranker = Reranker(cfg)

    query = "What were Verizon's total consolidated operating revenues in 2025?"
    print(f"QUERY: {query}\n")
    for strat in ("fixed", "semantic"):
        fused = retr.retrieve(query, strat, company="verizon")
        top = reranker.rerank(query, fused)
        print(f"=== {strat}: top {len(top)} after rerank ===")
        for i, r in enumerate(top):
            print(f"  {i+1}. {r.id}  rerank={r.rerank_score:.3f}  table={r.is_table}")
            print(f"     {r.text[:110].replace(chr(10), ' ')}")
        print()
    emb.close()
