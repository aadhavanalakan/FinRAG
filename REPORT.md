# Financial Document Intelligence Pipeline — Comparison Report

A Retrieval-Augmented Generation (RAG) system that answers natural-language questions
about SEC 10-K filings with grounded, cited answers. This report covers the two
required analyses — **chunking-strategy comparison** and **reranking-impact analysis** —
plus a hosted-model benchmark, all backed by a reproducible eval harness.

All numbers below are produced by the eval harness and persisted under `results/`,
each stamped with the config version that generated it.

---

## 1. Corpus

Three telecom 10-K filings, all fiscal year **ended December 31, 2025** (clean
apples-to-apples for cross-company questions):

| Company | Source | Parsed prose blocks | Parsed data tables |
|---|---|---|---|
| Verizon | SEC EDGAR (iXBRL HTML) | 140 | 118 |
| AT&T | SEC EDGAR (iXBRL HTML) | 117 | 104 |
| T-Mobile | SEC EDGAR (iXBRL HTML) | 131 | 108 |

The filings are Workiva-generated **inline-XBRL HTML**. The parser converts each
`<table>` to a Markdown table and wraps it in protective markers so the chunkers can
treat it as an atomic unit — the single most important design choice in the project.

## 2. Stack

| Layer | Choice |
|---|---|
| Generation | switchable per query: Nebius open models (default `Qwen3-30B-A3B`, temp 0) or OpenAI `gpt-4o`/`gpt-4o-mini` |
| Embeddings | `Qwen/Qwen3-Embedding-8B` via Nebius — **4096-dim**, cosine, disk-cached |
| Vector store | Pinecone serverless (one index, namespaces `fixed` / `semantic`) |
| Retrieval | hybrid dense + BM25, fused with Reciprocal Rank Fusion (k=60) |
| Reranker | `ms-marco-MiniLM-L-12-v2` cross-encoder (local; off by default) |
| Orchestration | LangGraph `StateGraph` — guard → retrieve → generate → audit |
| Guardrails | deterministic rule-based (input injection/advice/length + output citation audit) |
| UI | Streamlit chatbot — model switch, live latency/cost/TTFT metrics, sources, manage-corpus panel |
| Config | versioned YAML + typed Pydantic loader + per-run manifest |

### Component reference

| Component | Type | LLM calls | Role |
|---|---|---|---|
| Ingestion (`edgar` / `download_seed_filings`) | Pure Python | 0 | Pull 10-K HTML from SEC EDGAR |
| Parser (`parsing`) | Pure Python | 0 | Table-aware HTML → ordered prose + atomic Markdown tables |
| Fixed chunker (`chunking.chunk_fixed`) | Pure Python | 0 | 512-char `RecursiveCharacterTextSplitter` (tables shredded) |
| Semantic chunker (`chunking.chunk_semantic`) | Embedding-based | 0 chat | Sentence-embedding percentile breakpoints; tables kept whole |
| Embedder (`embedding`) | Hosted (Nebius) | 0 chat | Qwen3-Embedding-8B, 4096-d, disk-cached |
| Vector store (`vectorstore`) | Managed DB (Pinecone) | 0 | 2 namespaces (`fixed`/`semantic`), cosine |
| Hybrid retriever (`retrieval`) | Dense + BM25 + RRF | 0 | Fuse vector + keyword rankings |
| Reranker (`reranking`) | Local model | 0 | ms-marco cross-encoder (off by default — measured to hurt) |
| Guardrails (`guardrails`) | Pure Python | 0 | Input injection/advice/length + output citation audit |
| Generator (`generation` / `chat`) | LLM | **1** | Cited answer (Nebius or OpenAI), streamed |
| Orchestrator (`graph`) | LangGraph | 0 | guard → retrieve → generate → audit state machine |
| Faithfulness judge (`eval/faithfulness`) | LLM judge | ~2 / Q | Atomic-claim extraction + support check |
| RAGAS (`eval/run_ragas`) | LLM judge | ~3 / Q | context precision / recall / faithfulness |

Per **query**: exactly **1** LLM call (generation). Retrieval, reranking, and guardrails
are local/hosted-embedding (no chat call). Eval adds judge calls offline.

## 3. Method

**Two chunking strategies** over the same parsed documents:

| | Chunks (3 docs) | Intact tables | Avg chars |
|---|---|---|---|
| **Fixed** (512 chars, blind) | 4,317 | **0** (tables shredded) | ~380 |
| **Semantic** (topic boundaries) | 1,013 | **330** (kept whole) | ~1,600 |

Fixed-size chunking splits a financial table mid-row, separating a figure like
`$138,191` from its `Total Operating Revenues` label. Semantic chunking cuts at
sentence-embedding topic boundaries and keeps each table atomic.

**Eval harness.** A golden set of **18 manually-verified Q/A pairs** (16 answerable
lookups/comparisons + 2 negative "should-refuse" cases) across the three companies.
Relevance is **content-based**: a retrieved chunk is relevant if it contains the
verified answer figure (e.g. `138,191`) — strategy-agnostic and survives re-chunking.
Scored in two stages:

- **Retrieval** (no LLM): hit@k, MRR, nDCG@5
- **Generation** (LLM-judge): answer correctness, false-refusal rate, citation
  coverage, faithfulness (atomic-claim verification), refusal accuracy

**Experiment arms** — only the chunking strategy and the reranker toggle change;
everything downstream is identical.

---

## 4. Result A — Chunking comparison

### Retrieval (the cleanest comparison is *without* the reranker confound)

| arm | hit@1 | hit@3 | hit@5 | MRR | nDCG@5 |
|---|---|---|---|---|---|
| fixed_norerank | 0.250 | 0.375 | 0.562 | 0.353 | 0.404 |
| **semantic_norerank** | **0.375** | **0.562** | **0.688** | **0.483** | **0.523** |

Semantic chunking wins on **every metric** — +0.126 hit@5, +0.130 MRR, +0.119 nDCG.

### Generation

_(answer prompt v2 — see note below)_

| arm | correct | false-refuse | citation cov. | **faithfulness** | refuse acc. |
|---|---|---|---|---|---|
| fixed_norerank | 0.562 | 0.438 | 0.562 | 0.904 | 1.000 |
| **semantic_norerank** | **0.625** | **0.312** | **0.750** | **0.974** | 1.000 |

Semantic chunking wins on **every** generation metric: higher correctness (0.625 vs
0.562), fewer false-refusals (0.312 vs 0.438), better citation coverage (0.750 vs 0.562),
and higher faithfulness (0.974 vs 0.904). Fixed chunking's shredded tables feed the model
header-less number fragments, so it occasionally asserts a claim the context doesn't fully
support. Refusal accuracy stays **1.000** for both — when retrieval misses, the model
**refuses rather than fabricates**.

> **Prompt note (v1 → v2):** the answer prompt was softened (v2) so it answers synthesis/
> comparison questions instead of over-refusing, and made company-agnostic for the dynamic
> corpus. This lowered false-refusals and raised correctness; under the stricter v1 prompt
> faithfulness was a perfect 1.000 vs 0.778, but v1 refused too many answerable questions to
> be a usable product. Retrieval metrics above are unaffected (they don't use the prompt).

> **Conclusion A:** Semantic, table-aware chunking measurably improves both retrieval
> and answer faithfulness. The mechanism is the preserved table structure, not a
> generic "bigger chunks" effect.

---

## 5. Result B — Reranking impact

The textbook expectation is that a cross-encoder reranker improves results. On this
corpus it **does the opposite**, consistently:

| arm | hit@5 | MRR | nDCG@5 | faithfulness |
|---|---|---|---|---|
| semantic_norerank | **0.688** | **0.483** | **0.523** | **0.974** |
| semantic_rerank | 0.375 | 0.158 | 0.211 | 0.861 |
| fixed_norerank | 0.562 | 0.353 | 0.404 | 0.904 |
| fixed_rerank | 0.438 | 0.190 | 0.251 | 0.786 |

**Cause:** the MS-MARCO cross-encoder was trained on natural-language passage ranking
and systematically **demotes table chunks** — but the answer figures live in those
tables, so reranking pushes the right chunk out of the top-5.

### Can it be fixed? (mitigation study, semantic arm)

| variant | hit@5 | nDCG@5 |
|---|---|---|
| **no rerank** | **0.688** | **0.523** |
| ms-marco (baseline) | 0.375 | 0.211 |
| ms-marco + table caption | 0.625 | 0.429 |
| ms-marco + score blend | 0.625 | 0.372 |
| bge-reranker-base | 0.625 | 0.435 |
| bge-reranker-base + caption | 0.625 | 0.454 |

Every mitigation (captioning tables for the scorer, blending in the retrieval score, a
stronger reranker) **recovers most of the damage but none beats no-reranking**. Hybrid
retrieval already ranks the answer-bearing tables well; the cross-encoder can only
break ties it gets wrong.

> **Conclusion B:** For table-heavy financial RAG, a cross-encoder reranker is not
> worth it — invest in chunking instead. Reranking's value is corpus-dependent, not
> universal. The production default is therefore **reranking OFF**, and CI gates on
> `semantic_norerank`.

### Cross-check with RAGAS (the industry-standard metrics)

The same arms scored with **RAGAS-style metrics** (faithfulness, context precision,
context recall), judged by `gpt-4o-mini` — computed in-house (`eval/run_ragas.py`) to
avoid the `ragas` package's langchain/langgraph dependency clash on Python 3.14:

| arm | faithfulness | context_precision | context_recall |
|---|---|---|---|
| fixed_norerank | 0.439 | 0.382 | 0.521 |
| **semantic_norerank** | **0.681** | **0.436** | **0.583** |
| semantic_rerank | 0.573 | 0.218 | 0.375 |

RAGAS **independently confirms both findings**: semantic beats fixed on every metric, and
**reranking hurts** (semantic+rerank has the lowest context precision, 0.218). This is the
notable contrast with single-company kits that report reranking *helping* — those use
*pure dense* retrieval, where a cross-encoder cleans up a noisy ranking; our **hybrid
(dense + BM25 + RRF)** already ranks the answer tables well, so the reranker only demotes
them. (Absolute values are lower than single-company kits because this corpus is harder —
cross-company questions — and `gpt-4o-mini` elaborates beyond the context, which
faithfulness penalizes.)

---

## 6. Result C — Hosted-model benchmark

10 standardized prompts × 4 Nebius models, streaming (200 max tokens):

| model | TTFT p50 (s) | tok/s | latency p50 (s) | latency p95 (s) |
|---|---|---|---|---|
| **Qwen3-30B-A3B** (pipeline) | **0.592** | 95.0 | 2.55 | 3.17 |
| gemma-3-27b-it | 0.662 | 66.4 | 3.65 | 14.96 |
| Llama-3.3-70B | 0.742 | 19.7 | 11.15 | 22.97 |
| gpt-oss-120b | 0.817 | **690.7** | **1.06** | **1.32** |

`gpt-oss-120b` is dramatically the fastest on Nebius despite being largest (heavily
optimized MoE serving); the dense Llama-70B is the *slowest*. Our choice, Qwen3-30B,
is a strong balance (lowest TTFT, stable p95). Quality is measured separately by the
eval harness above. (Cost per request depends on Nebius per-token pricing — supply
`--price-file` to populate a cost column.)

---

## 7. Result D — Adversarial hard set & failure analysis

Beyond the golden set (which measures the system where it's meant to work), a **15-item
hard set** probes where it breaks: exact line-item names, wrong-year bait, cross-company
comparisons, a multi-hop ratio, ambiguous queries, refusal traps, and prompt-injection /
advice attempts. `python -m eval.run_hard` runs them, classifies each outcome, and writes
**[eval/FAILURE_ANALYSIS.md](eval/FAILURE_ANALYSIS.md)** from the real run.

| Outcome | Count | Notes |
|---|---|---|
| Met expected behavior | **14 / 15** | incl. all refusal traps + injection/advice blocks |
| Genuine miss | 1 | AT&T operating income — the figure wasn't retrieved, so the model correctly refused (a **retrieval** miss, not a generation one) |
| Acceptable refusals | 4 | cross-company / multi-hop questions decline rather than guess |

> **Conclusion D:** Refusal discipline and guardrails hold under adversarial probing.
> The one structural weakness is **multi-document synthesis** — comparisons and computed
> ratios need per-company sub-queries the single-pass retriever doesn't issue, so it
> honestly refuses. That points the next investment at query planning, not the model.

## 8. Engineering & product layer

What separates this from a notebook prototype:

- **A real chatbot.** A Streamlit UI ([app.py](app.py)) over the pipeline with per-query
  model switching and a live metrics strip (latency, TTFT, tokens/sec, token counts,
  cost) plus a session cost meter — every answer cites its sources.
- **Deterministic guardrails** ([finrag/guardrails.py](finrag/guardrails.py)) — input
  injection/advice/length gating and an output citation audit, with **no extra LLM call**.
- **LangGraph orchestration** ([finrag/graph.py](finrag/graph.py)) — the query flow is a
  compiled state machine with a conditional edge that short-circuits a blocked input.
- **Data-not-code corpus** — add/remove any US-listed company at runtime by ticker,
  validated against SEC EDGAR ([finrag/corpus.py](finrag/corpus.py), `data/companies.json`).
- **Versioned config + run manifest.** Every prompt/parameter lives in git-tracked YAML
  with a `version`; each run records the version + content hash of every config.
- **Two-stage offline eval** with a fixed golden set and an LLM-as-judge faithfulness
  scorer (no hard RAGAS dependency), plus the adversarial hard set above.
- **Embedding cache** (disk-backed) — re-running ingest re-embeds only changed text.
- **CI regression gating** — `python -m eval.gate` checks the gated arm against
  `thresholds.yaml` and fails the build on regression; the test suite asserts the
  table-preservation invariant and the guardrail rules.

### Key design decisions

| Decision | Rejected alternative | Rationale |
|---|---|---|
| Table-aware parsing (tables → atomic Markdown) | Strip all HTML to plain text | Keeps each figure beside its row label — semantic faithfulness **0.974 vs 0.904** |
| Hybrid dense + BM25 + RRF | Pure dense retrieval | BM25 nails exact tokens (tickers, literal `$` figures) dense search misses |
| Reranking **off** by default | Always-on cross-encoder | Measured to **hurt** on this hybrid + table-aware corpus (demotes table chunks) |
| Multi-provider generation (Nebius + OpenAI) | Single provider | Cheap open models by default; OpenAI parity, switchable per query with a live cost meter |
| In-house faithfulness + RAGAS-style metrics | Hard `ragas` package dependency | `ragas` pins clash with LangGraph on Python 3.14; an in-house judge gives the same metrics with zero dependency risk |
| Versioned YAML + run manifest | Hardcoded parameters | Every metric traceable to one config version (content-hash stamped) |
| LangGraph orchestration | Plain Python pipeline | guard → retrieve → generate → audit as a state machine; clean guardrail short-circuit |
| Registry-backed corpus (add/remove by ticker) | Fixed corpus in code | Corpus is data, not code — add any US-listed company at runtime via SEC EDGAR |

## 9. Limitations & future work

- Golden set is 18 items; scaling to 50–200 would tighten the metric estimates.
- Hosted temp-0 generation is not perfectly deterministic (observed ±0.06 on
  correctness between runs) — averaging over more items/seeds would reduce variance.
- Content-based relevance can over-credit a fixed-size fragment that contains a figure
  without its label; a stricter "figure + label co-occurrence" relevance would widen
  the measured chunking gap further.
- The reranking finding motivates trying a table-native reranker or an LLM reranker.
- **Multi-document synthesis is the main capability gap** (from the hard set): a single
  retrieval pass can't gather every company's answer table at once, so cross-company
  comparisons and computed ratios refuse. A query-planning / per-company sub-retrieval
  step would turn those refusals into answers.
- Prices in `config/pricing.yaml` are estimates; set real per-token values for an exact
  cost meter.

---

## Reproduce

```bash
pip install -r requirements.txt
# set NEBIUS_API_KEY and PINECONE_API_KEY in .env (OPENAI_API_KEY optional)
python -m scripts.ingest          # build the index (parse -> chunk -> embed -> upsert)
streamlit run app.py              # the chatbot UI
python -m scripts.ask "What were Verizon's total operating revenues in 2025?"
python -m scripts.add_company TSLA   # add a company at runtime (SEC EDGAR)

python -m eval.run_eval           # retrieval metrics (free)
python -m eval.run_eval_gen       # generation metrics (LLM)
python -m eval.rerank_study       # reranking mitigation study
python -m eval.run_hard           # adversarial hard set -> eval/FAILURE_ANALYSIS.md
python -m benchmark.run_benchmark # model speed comparison
```
