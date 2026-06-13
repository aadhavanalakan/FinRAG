"""
graph.py — the query flow as a LangGraph StateGraph (the orchestration layer).

The pipeline is a compiled state machine with a conditional edge:

        ┌─ blocked ─────────────────────────────┐
  guard ┤                                        ├─▶ END
        └─ ok ─▶ retrieve ─▶ generate ─▶ audit ──┘

  - guard     : input guardrail (injection / advice / length). On a block it routes
                straight to END — no retrieval, no model call, zero cost.
  - retrieve  : company detect + hybrid retrieve (+ optional rerank)
  - generate  : stream the answer from the chosen model, capturing latency/TTFT/cost
  - audit     : output guardrail (citation audit) + disclaimer

Nodes are thin wrappers over ChatEngine, so the graph and the direct engine share the
exact same retrieval/generation/guardrail code — only the orchestration differs. The
eval pipeline keeps using the plain engine; this graph powers the chatbot.
"""

from __future__ import annotations

import re
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from finrag.chat import ChatEngine, ChatMetrics, ChatResult, _ZERO_METRICS
from finrag.config import AppConfig
from finrag.guardrails import audit_output, check_input


class GraphState(TypedDict, total=False):
    # inputs
    question: str
    model: Any            # AnswerModel
    strategy: str
    use_reranker: bool
    company: str | None
    # produced along the way
    verdict: Any          # InputVerdict
    blocked: bool
    input_category: str | None
    chunks: list
    answer: str
    metrics: ChatMetrics
    context: str
    citations: list
    audit: Any            # OutputAudit


class RAGGraph:
    """A compiled LangGraph over a ChatEngine. `.answer(...)` matches ChatEngine.answer."""

    def __init__(self, engine: ChatEngine | None = None, cfg: AppConfig | None = None) -> None:
        self.engine = engine or ChatEngine(cfg)
        self.app = self._build()

    # --- nodes ---------------------------------------------------------------
    def _guard(self, state: GraphState) -> GraphState:
        verdict = check_input(state["question"])
        return {"verdict": verdict, "blocked": verdict.blocked,
                "input_category": verdict.category}

    def _retrieve(self, state: GraphState) -> GraphState:
        companies = self.engine.resolve_companies(state["question"], state.get("company"))
        chunks = self.engine.retrieve(state["question"], state["strategy"],
                                      state["use_reranker"], companies)
        return {"company": "+".join(companies) if companies else None, "chunks": chunks}

    def _generate(self, state: GraphState) -> GraphState:
        answer, metrics, context = self.engine.generate(
            state["question"], state["chunks"], state["model"])
        return {"answer": answer, "metrics": metrics, "context": context}

    def _audit(self, state: GraphState) -> GraphState:
        answer = state["answer"]
        citations = sorted(set(re.findall(r"\[chunk:([^\]]+)\]", answer)))
        audit = audit_output(answer, {c.id for c in state["chunks"]})
        return {"citations": citations, "audit": audit}

    # --- routing -------------------------------------------------------------
    @staticmethod
    def _route_after_guard(state: GraphState) -> str:
        return "blocked" if state["blocked"] else "ok"

    def _build(self):
        g = StateGraph(GraphState)
        g.add_node("guard", self._guard)
        g.add_node("retrieve", self._retrieve)
        g.add_node("generate", self._generate)
        g.add_node("audit", self._audit)

        g.set_entry_point("guard")
        g.add_conditional_edges("guard", self._route_after_guard,
                                {"blocked": END, "ok": "retrieve"})
        g.add_edge("retrieve", "generate")
        g.add_edge("generate", "audit")
        g.add_edge("audit", END)
        return g.compile()

    # --- public API (same shape as ChatEngine.answer) ------------------------
    def answer(self, question: str, model, strategy: str = "semantic",
               use_reranker: bool = False, company: str | None = None) -> ChatResult:
        # Multi-company comparisons need per-company sub-lookups the linear graph can't
        # express — delegate to the engine (which still guards + audits).
        if not company and len(self.engine.resolve_companies(question, None)) >= 2:
            return self.engine.answer(question, model, strategy, use_reranker, company)
        final = self.app.invoke({
            "question": question, "model": model, "strategy": strategy,
            "use_reranker": use_reranker, "company": company,
        })
        if final.get("blocked"):
            return self.engine.blocked_result(
                question, model, strategy, use_reranker, final["verdict"])
        return ChatResult(
            question=question, answer=final["answer"], model_label=model.label,
            model_id=model.model, strategy=strategy, reranked=use_reranker,
            company=final.get("company"), chunks=final["chunks"],
            citations=final["citations"], metrics=final["metrics"],
            context=final.get("context", ""), audit=final["audit"],
        )

    def close(self) -> None:
        self.engine.close()


if __name__ == "__main__":
    from finrag.chat import available_models, default_model_id

    rg = RAGGraph()
    models = {m.model: m for m in available_models()}
    m = models.get(default_model_id()) or next(iter(models.values()))

    print("nodes:", list(rg.app.get_graph().nodes))
    for q in ["What were Verizon's total operating revenues in 2025?",
              "Should I buy Verizon stock?"]:
        r = rg.answer(q, m)
        print(f"\nQ: {q}\n  blocked={r.blocked}  answer={r.answer[:70]}")
        if not r.blocked:
            print("  metrics:", r.metrics.as_row(), "| audit flags:", r.audit.flags)
    rg.close()
