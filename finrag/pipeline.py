"""
pipeline.py — end-to-end RAG: retrieve -> rerank -> generate.

One RAGPipeline instance is one experiment arm, set by two toggles:
  - strategy      : "fixed" | "semantic"   (which Pinecone namespace)
  - use_reranker  : True | False           (arm 3 turns this off)

The three arms from the brief:
  1. RAGPipeline("fixed",    use_reranker=True)
  2. RAGPipeline("semantic", use_reranker=True)   <- expected winner
  3. RAGPipeline("semantic", use_reranker=False)  <- isolates the rerank lift

ask() returns a RAGResult with the answer, the chunks actually used, and the run_id
so every answer is traceable to a config version.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from finrag.config import AppConfig, load_config
from finrag.embedding import Embedder
from finrag.generation import Generator
from finrag.reranking import Reranker
from finrag.retrieval import HybridRetriever, Retrieved
from finrag.vectorstore import VectorStore

# Light company detection from the question text (override via ask(company=...)).
# Static aliases for the seed telecoms (handle punctuation variants cleanly); these
# take precedence over registry-derived hints.
_COMPANY_HINTS = {
    "verizon": "verizon",
    "at&t": "att", "att": "att", "at and t": "att",
    "t-mobile": "tmobile", "tmobile": "tmobile", "t mobile": "tmobile",
}


def detect_companies(question: str) -> list[str]:
    """All companies named in the question, in order, de-duplicated. Drives both
    single-company filtering and multi-company (comparison) retrieval."""
    q = question.lower()
    found: list[str] = []

    def add(c: str) -> None:
        if c not in found:
            found.append(c)

    for hint, company in _COMPANY_HINTS.items():
        if hint in q:
            add(company)
    # Registry-derived hints (ticker + name words) cover companies added at runtime.
    # Word-boundary match so short hints don't fire on substrings (e.g. ticker inside a word).
    try:
        from finrag.corpus import company_hints
        for hint, company in company_hints().items():
            if re.search(rf"\b{re.escape(hint)}\b", q):
                add(company)
    except Exception:  # registry missing / unreadable — static hints are enough
        pass
    return found


def detect_company(question: str) -> str | None:
    """The first company named (back-compat single-company helper)."""
    companies = detect_companies(question)
    return companies[0] if companies else None


@dataclass
class RAGResult:
    question: str
    strategy: str
    reranked: bool
    company: str | None
    answer: str
    chunks: list[Retrieved]
    run_id: str
    citations: list[str] = field(default_factory=list)


class RAGPipeline:
    def __init__(
        self,
        cfg: AppConfig | None = None,
        strategy: str = "semantic",
        use_reranker: bool = False,   # eval (results/rerank_study.json): reranking HURTS on this corpus
    ) -> None:
        self.cfg = cfg or load_config()
        self.strategy = strategy
        self.use_reranker = use_reranker
        self.embedder = Embedder(self.cfg.models.embedding)
        self.store = VectorStore(self.cfg)
        self.retriever = HybridRetriever(self.cfg, self.embedder, self.store)
        self.reranker = Reranker(self.cfg)
        self.generator = Generator(self.cfg)

    def ask(self, question: str, company: str | None = None) -> RAGResult:
        company = company or detect_company(question)
        fused = self.retriever.retrieve(question, self.strategy, company)

        if self.use_reranker:
            final = self.reranker.rerank(question, fused)
        else:
            final = fused[: self.cfg.pipeline.reranking.output_top_n]

        answer = self.generator.generate(question, final)
        citations = sorted(set(re.findall(r"\[chunk:([^\]]+)\]", answer)))
        return RAGResult(
            question=question, strategy=self.strategy, reranked=self.use_reranker,
            company=company, answer=answer, chunks=final,
            run_id=self.cfg.manifest.run_id, citations=citations,
        )

    def close(self) -> None:
        self.embedder.close()
