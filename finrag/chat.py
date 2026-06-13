"""
chat.py — the chatbot engine: retrieve -> generate (streaming, multi-provider) with
live per-query metrics (latency, time-to-first-token, tokens/sec, token counts, cost).

This is the UI-facing sibling of pipeline.py. It reuses the SAME retrieval, reranking
and context-assembly as the eval pipeline (so answers are identical in substance), but:
  - the answer model is SWITCHABLE per query (any OpenAI-compatible endpoint listed in
    config/answer_models.yaml — Nebius open models or OpenAI), and
  - generation STREAMS, so we can measure TTFT and throughput and price the call from
    the exact token usage the API returns (config/pricing.yaml).

Nothing here touches the strict AppConfig/manifest, so eval + CI stay unchanged.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from openai import OpenAI

from finrag.config import AppConfig, load_config, resolve_api_key
from finrag.embedding import Embedder
from finrag.generation import Generator
from finrag.guardrails import OutputAudit, audit_output, check_input
from finrag.pipeline import detect_companies
from finrag.reranking import Reranker
from finrag.retrieval import HybridRetriever, Retrieved
from finrag.vectorstore import VectorStore

_REGISTRY_PATH = Path("config/answer_models.yaml")
_PRICING_PATH = Path("config/pricing.yaml")

# Comparison / superlative cues that imply "look across companies" even when none is named.
_COMPARE_CUES = (
    "compare", "comparison", "versus", " vs ", " vs.", "which company", "which carrier",
    "which of", "highest", "lowest", "biggest", "largest", "smallest", "most", "least",
    "better", "higher", "lower", "rank", "carriers", "companies", "all three", "among",
)

# A query that reliably pins a company's CONSOLIDATED INCOME STATEMENT (one table that
# holds revenue, operating income, net income, and EPS) at the top of retrieval.
_INCOME_STMT_QUERY = ("consolidated statements of income total operating revenues "
                      "operating income net income diluted earnings per share")

# Metric detection: (trigger words, retrieval query, human label, is_income_statement_line).
# Income-statement lines are all answered from the single income-statement table above;
# others (capex, service revenue) use their own canonical query. Order: specific first.
_METRICS = [
    (("operating income", "income from operations", "operating profit"), "operating income", "operating income", True),
    (("net income", "net earnings", "bottom line", "profit"), "net income", "net income", True),
    (("earnings per share", "eps", "per share"), "diluted earnings per share", "diluted earnings per share", True),
    (("capital expenditure", "capex", "capital spending"), "capital expenditures", "capital expenditures", False),
    (("service revenue", "wireless service"), "total service revenues", "total service revenues", False),
    (("operating expense", "operating costs"), "total operating expenses", "total operating expenses", True),
    (("revenue", "revenues", "sales", "top line"), "total operating revenues net sales", "total operating revenues", True),
]


def _metric_for(question: str):
    """(retrieval_query, human_label, is_income_statement_line) for a metric, or (None, None, False)."""
    q = question.lower()
    for keys, rq, label, is_is in _METRICS:
        if any(k in q for k in keys):
            return rq, label, is_is
    return None, None, False


def _target_year(question: str) -> str:
    """Resolve which fiscal year a comparison is about. Explicit year wins; otherwise
    'last fiscal year' / 'latest' map to the corpus's most recent year (2025)."""
    yrs = re.findall(r"\b(20\d{2})\b", question)
    return yrs[-1] if yrs else "2025"


# =============================================================================
# model registry + pricing (read from YAML, independent of the strict AppConfig)
# =============================================================================
@dataclass(frozen=True)
class AnswerModel:
    label: str
    provider: str
    model: str
    base_url: str
    api_key_env: str

    def key_present(self) -> bool:
        import os
        return bool((os.getenv(self.api_key_env) or "").strip())


def load_answer_models() -> list[AnswerModel]:
    data = yaml.safe_load(_REGISTRY_PATH.read_text(encoding="utf-8"))
    return [AnswerModel(**m) for m in data["models"]]


def default_model_id() -> str:
    return yaml.safe_load(_REGISTRY_PATH.read_text(encoding="utf-8"))["default"]


def load_prices() -> dict[str, dict]:
    if not _PRICING_PATH.exists():
        return {}
    return yaml.safe_load(_PRICING_PATH.read_text(encoding="utf-8")).get("prices", {})


def available_models() -> list[AnswerModel]:
    """Only models whose API key is actually set in the environment."""
    return [m for m in load_answer_models() if m.key_present()]


# =============================================================================
# result + metrics
# =============================================================================
@dataclass
class ChatMetrics:
    latency_s: float            # total wall-clock for generation
    ttft_s: float               # time to first token
    prompt_tokens: int
    completion_tokens: int
    tokens_per_s: float
    cost_usd: float | None      # None if the model isn't priced in pricing.yaml

    def as_row(self) -> dict:
        return {
            "latency_s": round(self.latency_s, 3),
            "ttft_s": round(self.ttft_s, 3),
            "tok/s": round(self.tokens_per_s, 1),
            "in_tok": self.prompt_tokens,
            "out_tok": self.completion_tokens,
            "cost_usd": (round(self.cost_usd, 6) if self.cost_usd is not None else None),
        }


_ZERO_METRICS = ChatMetrics(0.0, 0.0, 0, 0, 0.0, 0.0)


@dataclass
class ChatResult:
    question: str
    answer: str
    model_label: str
    model_id: str
    strategy: str
    reranked: bool
    company: str | None
    chunks: list[Retrieved]
    citations: list[str]
    metrics: ChatMetrics
    context: str = field(default="", repr=False)
    blocked: bool = False                 # input guardrail tripped → no model call
    input_category: str | None = None     # "injection" | "advice" | "too_long"
    audit: OutputAudit | None = None       # output citation audit (None when blocked)


# =============================================================================
# engine
# =============================================================================
class ChatEngine:
    def __init__(self, cfg: AppConfig | None = None) -> None:
        self.cfg = cfg or load_config()
        self.embedder = Embedder(self.cfg.models.embedding)
        self.store = VectorStore(self.cfg)
        self.retriever = HybridRetriever(self.cfg, self.embedder, self.store)
        self.reranker = Reranker(self.cfg)
        self.generator = Generator(self.cfg)   # reused only for context assembly + prompt
        self.prices = load_prices()
        self._clients: dict[str, OpenAI] = {}

    def _client(self, m: AnswerModel) -> OpenAI:
        if m.model not in self._clients:
            self._clients[m.model] = OpenAI(base_url=m.base_url, api_key=resolve_api_key(m.api_key_env))
        return self._clients[m.model]

    def _cost(self, model_id: str, prompt_tokens: int, completion_tokens: int) -> float | None:
        p = self.prices.get(model_id)
        if not p:
            return None
        return (prompt_tokens / 1_000_000) * p["input"] + (completion_tokens / 1_000_000) * p["output"]

    def _income_statement(self, strategy: str, company: str, year: str):
        """The company's consolidated income-statement table (revenue/op income/net income/EPS)."""
        mf = self.retriever.retrieve(f"{_INCOME_STMT_QUERY} {year}", strategy, company)
        tables = [c for c in mf[:10] if c.is_table]
        full = [c for c in tables
                if "net income" in c.text.lower() and "operating income" in c.text.lower()]
        return (full or tables or [None])[0]

    def _focused(self, question: str, strategy: str, use_reranker: bool, comp: str) -> list[Retrieved]:
        """A small, sharp single-company set (top-n + the metric table) — used by the
        comparison planner, where each sub-answer should be clean, not broad."""
        fused = self.retriever.retrieve(question, strategy, comp)
        n = self.cfg.pipeline.reranking.output_top_n
        base = self.reranker.rerank(question, fused) if use_reranker else fused[:n]
        mt = self._metric_table(strategy, comp, question)
        if mt and mt.id not in {c.id for c in base}:
            base = base[:1] + [mt] + base[1:]
        return base

    def _metric_table(self, strategy: str, company: str, question: str):
        """The table guaranteed to hold the asked-for metric: the income statement for an
        income-statement line, else the metric's own canonical table. None if no metric."""
        rq, _, is_is = _metric_for(question)
        if not rq:
            return None
        year = _target_year(question)
        if is_is:
            return self._income_statement(strategy, company, year)
        mf = self.retriever.retrieve(f"{rq} {year}", strategy, company)
        return next((c for c in mf[:8] if c.is_table), None)

    def resolve_companies(self, question: str, company: str | None) -> list[str]:
        """Which companies to retrieve for. Explicit filter wins; else every company
        named; else — if the question is clearly a comparison/superlative but names
        nobody ("which carrier...", "compare...") — all registered companies."""
        if company:
            return [company]
        detected = detect_companies(question)
        if len(detected) >= 2:
            return detected
        if any(c in question.lower() for c in _COMPARE_CUES):
            try:
                from finrag.corpus import symbols
                allc = symbols()
                if len(allc) >= 2:
                    return allc
            except Exception:
                pass
        return detected

    def retrieve(self, question: str, strategy: str, use_reranker: bool,
                 companies: list[str]) -> list[Retrieved]:
        top_n = self.cfg.pipeline.reranking.output_top_n
        # 0 or 1 company → ordinary single-namespace retrieval, PLUS a metric guarantee:
        # if the question asks for a financial metric, also pull that metric's table via a
        # canonical query and inject it, so a figure lookup never misses the income-
        # statement line just because the conversational phrasing ranked it low.
        if len(companies) <= 1:
            comp = companies[0] if companies else None
            fused = self.retriever.retrieve(question, strategy, comp)
            # Coverage mode (default): feed many ranked chunks so a retrievable fact isn't
            # left out. Reranking mode stays selective (top-n) for precision experiments.
            base = self.reranker.rerank(question, fused) if use_reranker else list(fused[:20])
            if comp:
                mt = self._metric_table(strategy, comp, question)
                if mt and mt.id not in {c.id for c in base}:
                    base = [mt] + base                  # guarantee the income-statement table
            return base
        # Multiple companies (a comparison) → retrieve each company with a CANONICAL
        # metric query so its income-statement line is pinned (a fuzzy conversational
        # query can surface a segment sub-total instead — the classic wrong-comparison
        # bug). Take its top chunks and ensure its best table is present; tables first so
        # the answer-bearing rows survive the context-char budget.
        rq, _, _ = _metric_for(question)
        mq = f"{rq or question} {_target_year(question)}".strip()
        out: list[Retrieved] = []
        for comp in companies:
            fused = self.retriever.retrieve(mq, strategy, comp)
            ranked = self.reranker.rerank(mq, fused) if use_reranker else fused
            # Keep it CLEAN: just the best table (the consolidated statement) + the best
            # overall chunk per company. Fewer, sharper tables → the model doesn't pick a
            # segment sub-total or the prior-year column by mistake.
            picks: list[Retrieved] = []
            best_table = next((c for c in ranked[:12] if c.is_table), None)
            if best_table:
                picks.append(best_table)
            for c in ranked[:3]:
                if c.id not in {p.id for p in picks}:
                    picks.append(c)
                    break
            out.extend(picks)
        out.sort(key=lambda c: 0 if c.is_table else 1)   # tables first
        return out

    # Modern models have huge context windows — for a small corpus, feeding MORE context
    # (not 5 stingy chunks) is the single biggest reliability win. Budget in characters,
    # comfortably within every supported model's window (~30k tokens).
    _CTX_BUDGET = 120_000

    def _assemble(self, chunks: list[Retrieved], max_chars: int) -> str:
        """Label chunks with their ids and pack them, in relevance order, up to a char
        budget — so a retrievable fact is rarely left out of the model's context."""
        out, used = [], 0
        for c in chunks:
            block = f"[chunk:{c.id}]\n{c.text}"
            if out and used + len(block) > max_chars:
                break
            out.append(block)
            used += len(block)
        return "\n\n".join(out)

    def generate(self, question: str, final: list[Retrieved], model: AnswerModel
                 ) -> tuple[str, ChatMetrics, str]:
        """Assemble context from chunks, stream the answer, and measure cost/latency."""
        context = self._assemble(final, self._CTX_BUDGET)
        answer, metrics = self._stream(question, context, model)
        return answer, metrics, context

    def _stream(self, question: str, context: str, model: AnswerModel
                ) -> tuple[str, ChatMetrics]:
        """Stream an answer for a question against a ready-made context string, measuring
        latency / TTFT / throughput / cost. Used by generate() AND compare()."""
        user = self.cfg.prompts.rag_answer.user.format(context=context, question=question)
        messages = [
            {"role": "system", "content": self.cfg.prompts.rag_answer.system},
            {"role": "user", "content": user},
        ]

        client = self._client(model)
        g = self.cfg.models.generation
        start = time.perf_counter()
        first_t: float | None = None
        parts: list[str] = []
        usage = None
        stream = client.chat.completions.create(
            model=model.model, temperature=g.temperature, max_tokens=g.max_tokens,
            stream=True, stream_options={"include_usage": True}, messages=messages,
        )
        for ev in stream:
            if ev.choices and ev.choices[0].delta and ev.choices[0].delta.content:
                if first_t is None:
                    first_t = time.perf_counter()
                parts.append(ev.choices[0].delta.content)
            if getattr(ev, "usage", None):
                usage = ev.usage
        end = time.perf_counter()

        answer = "".join(parts).strip()
        prompt_tokens = usage.prompt_tokens if usage else len(user) // 4
        completion_tokens = usage.completion_tokens if usage else max(1, len(answer) // 4)
        ttft = (first_t - start) if first_t else (end - start)
        gen_time = max(1e-6, end - (first_t or start))
        metrics = ChatMetrics(
            latency_s=end - start, ttft_s=ttft,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            tokens_per_s=completion_tokens / gen_time,
            cost_usd=self._cost(model.model, prompt_tokens, completion_tokens),
        )
        return answer, metrics

    def compare(self, question: str, model: AnswerModel, strategy: str,
                use_reranker: bool, companies: list[str]):
        """Reliable multi-company comparison via query planning: resolve EACH company's
        figure with its own single-company lookup (those are accurate), then synthesize
        the comparison from those clean resolved answers — so the model never has to pick
        the right number/year out of a pile of similar tables."""
        from finrag.corpus import load_registry
        reg = load_registry()
        _, label, _ = _metric_for(question)
        year = _target_year(question)

        blocks, all_chunks, summed_cost = [], [], 0.0
        for comp in companies:
            name = reg.get(comp, {}).get("name", comp)
            sub_q = f"What were {name}'s {label} for fiscal year {year}?" if label \
                else f"{question} (for {name})"
            # Focused per-company set (clean, not broad) — keeps each sub-answer sharp.
            chunks = self._focused(sub_q, strategy, use_reranker, comp)
            sub_ans, sub_m, _ = self.generate(sub_q, chunks, model)
            summed_cost += (sub_m.cost_usd or 0.0)
            blocks.append(f"=== {name} — fiscal year {year} ===\n{sub_ans}")
            all_chunks.extend(chunks)

        # Synthesize from the resolved single-company answers (each already a cited figure).
        context = "\n\n".join(blocks)
        answer, metrics = self._stream(question, context, model)
        metrics.cost_usd = (metrics.cost_usd or 0.0) + summed_cost     # full planning cost
        return answer, metrics, all_chunks, context

    def blocked_result(self, question: str, model: AnswerModel, strategy: str,
                       use_reranker: bool, verdict) -> ChatResult:
        """Build the zero-cost ChatResult for an input the guardrail blocked."""
        return ChatResult(
            question=question, answer=verdict.message, model_label=model.label,
            model_id=model.model, strategy=strategy, reranked=use_reranker, company=None,
            chunks=[], citations=[], metrics=_ZERO_METRICS,
            blocked=True, input_category=verdict.category,
        )

    def answer(
        self,
        question: str,
        model: AnswerModel,
        strategy: str = "semantic",
        use_reranker: bool = False,
        company: str | None = None,
    ) -> ChatResult:
        # INPUT guardrail — block/deflect before any retrieval or model call (zero cost).
        verdict = check_input(question)
        if verdict.blocked:
            return self.blocked_result(question, model, strategy, use_reranker, verdict)

        companies = self.resolve_companies(question, company)
        if len(companies) >= 2:
            answer, metrics, final, context = self.compare(
                question, model, strategy, use_reranker, companies)
        else:
            final = self.retrieve(question, strategy, use_reranker, companies)
            answer, metrics, context = self.generate(question, final, model)
        citations = sorted(set(re.findall(r"\[chunk:([^\]]+)\]", answer)))
        # OUTPUT guardrail — audit citations against the chunks we actually retrieved.
        audit = audit_output(answer, {c.id for c in final})
        return ChatResult(
            question=question, answer=answer, model_label=model.label, model_id=model.model,
            strategy=strategy, reranked=use_reranker,
            company="+".join(companies) if companies else None,
            chunks=final, citations=citations, metrics=metrics, context=context,
            audit=audit,
        )

    def close(self) -> None:
        self.embedder.close()


if __name__ == "__main__":
    # Smoke test (no Streamlit): one answer with the default model + its metrics.
    eng = ChatEngine()
    models = {m.model: m for m in available_models()}
    chosen = models.get(default_model_id()) or next(iter(models.values()))
    res = eng.answer("What were Verizon's total operating revenues in 2025?", chosen)
    print(f"model={res.model_label}")
    print("ANSWER:", res.answer)
    print("metrics:", res.metrics.as_row())
    print("citations:", res.citations)
    eng.close()
