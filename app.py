"""
app.py — FinRAG chatbot UI (Streamlit).

A chat interface over the RAG pipeline with everything visible:
  - pick the answer model (Nebius open models, or OpenAI if a key + billing are set)
  - see / add / remove the corpus companies live (validated against SEC EDGAR)
  - toggle chunking strategy / reranker / orchestration / company filter
  - every answer shows cited sources AND a live metrics strip:
      latency · time-to-first-token · tokens/sec · in/out tokens · cost
  - the sidebar tracks cumulative session cost + a per-query history table

Run:  streamlit run app.py
"""

from __future__ import annotations

import os

import streamlit as st

from finrag.chat import ChatEngine, available_models, default_model_id
from finrag.corpus import add_company, load_registry, remove_company, symbols
from finrag.edgar import EdgarError
from finrag.graph import RAGGraph

st.set_page_config(page_title="FinRAG — AI 10-K Analyst", page_icon="🦉", layout="wide")

# --- sidebar theme: red + wider, so session totals / cost are easy to read ----------
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #b3122b;
        width: 380px !important;
        min-width: 380px !important;
    }
    /* default sidebar text (headers, labels, captions, metrics) -> white */
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    /* but widgets sit on light surfaces -> keep their text dark + readable */
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] [data-baseweb="select"] *,
    [data-testid="stSidebar"] [data-testid="stDataFrame"] *,
    [data-testid="stSidebar"] button * { color: #1a1a1a !important; }
    /* make the Session metrics pop */
    [data-testid="stSidebar"] [data-testid="stMetricValue"] { color: #ffffff !important; font-weight: 700; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.35); }
    </style>
    """,
    unsafe_allow_html=True,
)

# On Streamlit Cloud, API keys come from st.secrets — mirror them into env vars so the
# existing os.getenv-based key resolution works (locally, .env handles this instead).
try:
    for _k in ("NEBIUS_API_KEY", "PINECONE_API_KEY", "OPENAI_API_KEY",
               "ANTHROPIC_API_KEY", "APP_PASSWORD"):
        if _k in st.secrets and not os.getenv(_k):
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass  # no secrets.toml locally — .env handles it

# Fail with a clear setup message (not a redacted crash) if required keys are missing.
_missing = [k for k in ("NEBIUS_API_KEY", "PINECONE_API_KEY")
            if not (os.getenv(k) or "").strip()]
if _missing:
    st.title("🦉 FinRAG — setup needed")
    st.error("Missing required secret(s): **" + ", ".join(_missing) + "**")
    try:
        _seen = list(st.secrets.keys())
    except Exception:
        _seen = []
    st.caption(f"Secrets the app can currently see (names only): {_seen or '(none)'}")
    st.markdown("On Streamlit Cloud, add them in **Manage app → ⋮ → Settings → Secrets** "
                "(TOML, no `[section]` header), then **Save** — the app reruns automatically:")
    st.code('NEBIUS_API_KEY   = "your-key"\nPINECONE_API_KEY = "your-key"\n'
            'OPENAI_API_KEY   = "your-key"\nAPP_PASSWORD     = "anything"', language="toml")
    st.stop()


# Optional password gate — protects your API keys on a public deploy. Active only when
# APP_PASSWORD is set (so local dev stays open). Set it in Streamlit Cloud → Secrets.
def _require_password() -> None:
    expected = (os.getenv("APP_PASSWORD") or "").strip()
    if not expected or st.session_state.get("authed"):
        return
    st.title("🦉 FinRAG")
    st.caption("This demo is password-protected to control API usage.")
    pw = st.text_input("Access password", type="password")
    if pw and pw == expected:
        st.session_state["authed"] = True
        st.rerun()
    elif pw:
        st.error("Incorrect password.")
    st.stop()


_require_password()

# Curated demo prompts — every one is verified to answer (no awkward refusals on stage),
# spanning lookups, a comparison, the auditor, narrative, capex, EPS, and year-over-year.
EXAMPLES = [
    "What were Verizon's total operating revenues in 2025?",
    "What was AT&T's operating income in 2025?",
    "What was T-Mobile's net income in 2025?",
    "Did AT&T or Verizon have higher operating income in 2025?",
    "Who audited T-Mobile's financial statements?",
    "What does AT&T describe as its 2025 revenue drivers?",
    "How much did Verizon spend on capital expenditures in 2025?",
    "How did Verizon's total operating revenues change from 2024 to 2025?",
]


# --- engine (built once, cached across reruns) -------------------------------
@st.cache_resource(show_spinner="Loading retrieval index + models…")
def get_engine() -> ChatEngine:
    return ChatEngine()


@st.cache_resource(show_spinner=False)
def get_graph() -> RAGGraph:
    return RAGGraph(engine=get_engine())


def init_state() -> None:
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("total_cost", 0.0)
    st.session_state.setdefault("total_queries", 0)


init_state()
engine = get_engine()
models = available_models()
model_by_label = {m.label: m for m in models}


# =============================================================================
# render helpers
# =============================================================================
def render_metrics_strip(m) -> None:
    row = m.as_row()
    cols = st.columns(6)
    cols[0].metric("Latency", f"{row['latency_s']:.2f}s")
    cols[1].metric("TTFT", f"{row['ttft_s']:.2f}s")
    cols[2].metric("Throughput", f"{row['tok/s']:.0f} tok/s")
    cols[3].metric("Input tok", f"{row['in_tok']:,}")
    cols[4].metric("Output tok", f"{row['out_tok']:,}")
    cols[5].metric("Cost", f"${row['cost_usd']:.5f}" if row["cost_usd"] is not None else "n/a")


def render_sources(r) -> None:
    cited = set(r.citations)
    with st.expander(f"📎 Sources — {len(r.chunks)} retrieved, {len(cited)} cited"):
        for c in r.chunks:
            tag = "🟦 TABLE" if c.is_table else "⬜ prose"
            star = " ✅ cited" if c.id in cited else ""
            st.markdown(f"**[{c.id}]** · {tag} · _{c.section or '—'}_{star}")
            st.caption(c.text[:300].replace("\n", " ") + ("…" if len(c.text) > 300 else ""))


def render_assistant(r) -> None:
    st.markdown(r.answer)
    if r.blocked:
        st.warning(f"🛡️ Guardrail blocked this input — category: **{r.input_category}**. "
                   "No model was called (zero cost).")
        return
    st.caption(f"model: {r.model_label} · strategy: {r.strategy} · "
               f"reranked: {r.reranked} · company: {r.company or 'all'}")
    render_metrics_strip(r.metrics)
    if r.audit and r.audit.flags:
        st.warning("🛡️ Output audit: " + "; ".join(r.audit.flags))
    render_sources(r)
    if r.audit:
        st.caption(r.audit.disclaimer)


def friendly_error(e: Exception) -> None:
    msg = str(e)
    name = type(e).__name__
    if "insufficient_quota" in msg or "RateLimit" in name or "429" in msg:
        st.error("⚠️ This model's provider rejected the request for **quota / billing** "
                 "reasons. Add billing/credits for that provider (e.g. OpenAI), or switch "
                 "to a **Nebius** model in the sidebar — those are working.")
    elif "AuthenticationError" in name or "401" in msg:
        st.error("⚠️ Authentication failed for this provider — check the API key in `.env`.")
    else:
        st.error(f"⚠️ Generation failed: {msg[:300]}")


# =============================================================================
# sidebar — model · settings · corpus · session
# =============================================================================
with st.sidebar:
    st.header("⚙️ Controls")

    if not models:
        st.error("No answer models available — set NEBIUS_API_KEY (and optionally "
                 "OPENAI_API_KEY) in .env.")
        st.stop()

    # --- model ---
    labels = list(model_by_label)
    default_label = next((m.label for m in models if m.model == default_model_id()), labels[0])
    model_label = st.selectbox("🧠 Answer model", labels, index=labels.index(default_label))
    model = model_by_label[model_label]
    if not any(m.provider == "openai" for m in models):
        st.caption("💡 Add `OPENAI_API_KEY` to `.env` to enable GPT-4o / GPT-4o-mini.")

    # --- retrieval settings ---
    with st.expander("🔧 Retrieval settings", expanded=False):
        strategy = st.radio("Chunking strategy", ["semantic", "fixed"], index=0,
                            help="semantic keeps tables whole (recommended); fixed is the naive baseline")
        use_reranker = st.checkbox("Use reranker", value=False,
                                   help="cross-encoder rerank — measured to HURT on this corpus; off by default")
        orchestration = st.radio("Orchestration", ["LangGraph", "Direct"], index=0, horizontal=True,
                                 help="LangGraph runs the compiled guard→retrieve→generate→audit state machine.")
        company = st.selectbox("Company filter", ["(auto-detect)"] + symbols(), index=0)
        company_arg = None if company == "(auto-detect)" else company

    st.divider()

    # --- corpus ---
    reg = load_registry()
    st.subheader(f"📚 Corpus · {len(reg)} companies")
    for sym, meta in reg.items():
        lock = " 🔒" if meta.get("eval") else ""
        st.markdown(f"&nbsp;&nbsp;**{meta.get('ticker', sym.upper())}** · {meta.get('name', sym)}{lock}",
                    unsafe_allow_html=True)
    st.caption("🔒 = protected (graded eval set)")

    with st.expander("➕ Add / remove a company"):
        st.caption("Any US-listed company with a 10-K — validated against SEC EDGAR, then "
                   "downloaded, chunked, embedded and indexed. The rest is untouched.")
        new_ticker = st.text_input("Ticker to add", placeholder="e.g. AAPL").strip()
        if st.button("Add company", disabled=not new_ticker, use_container_width=True):
            try:
                with st.spinner(f"Validating + indexing {new_ticker.upper()} "
                                "(download → chunk → embed)…"):
                    r = add_company(new_ticker)
                st.success(f"Added {r['name']} — {r['semantic']['chunks']} semantic chunks.")
                get_engine.clear()
                st.rerun()
            except (EdgarError, ValueError) as e:
                st.error(str(e))

        removable = [s for s, m in reg.items() if not m.get("eval")]
        if removable:
            rem = st.selectbox("Remove an added company", removable)
            if st.button("Remove company", use_container_width=True):
                r = remove_company(rem)
                st.success(f"Removed {rem} ({r['deleted_vectors']} vectors).")
                get_engine.clear()
                st.rerun()
        else:
            st.caption("No removable companies yet (the 3 seed telecoms are protected).")

    st.divider()

    # --- session ---
    st.subheader("💰 Session")
    c1, c2 = st.columns(2)
    c1.metric("Queries", st.session_state.total_queries)
    c2.metric("Cost", f"${st.session_state.total_cost:.4f}")
    if st.session_state.history:
        st.caption("Per-query metrics")
        st.dataframe(
            [{"model": r.model_id.split("/")[-1], **r.metrics.as_row()} for r in st.session_state.history],
            use_container_width=True, hide_index=True)
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.session_state.total_cost = 0.0
        st.session_state.total_queries = 0
        st.rerun()


# =============================================================================
# main — title · examples · conversation
# =============================================================================
st.title("🦉 FinRAG — Your AI 10-K Analyst")
st.caption(f"Grounded, cited answers over {len(reg)} companies' SEC 10-K filings · "
           "switch models · live metrics & cost")

# resolve the next question: chat box, or a clicked example
prompt = st.chat_input("Ask about a company's 10-K…") or st.session_state.pop("pending", None)

# empty-state example prompts
if not st.session_state.history and prompt is None:
    st.markdown("#### 💬 Try asking")
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 2].button(ex, key=f"ex{i}", use_container_width=True):
            st.session_state.pending = ex
            st.rerun()

# replay the conversation
for r in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(r.question)
    with st.chat_message("assistant"):
        render_assistant(r)

# handle a new question
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        executor = get_graph() if orchestration == "LangGraph" else engine
        result = None
        try:
            with st.spinner(f"Retrieving + generating with {model.label} · {orchestration}…"):
                result = executor.answer(prompt, model=model, strategy=strategy,
                                         use_reranker=use_reranker, company=company_arg)
        except Exception as e:  # noqa: BLE001 — surface any provider error cleanly
            friendly_error(e)
        if result is not None:
            render_assistant(result)

    if result is not None:
        st.session_state.history.append(result)
        st.session_state.total_queries += 1
        if result.metrics.cost_usd:
            st.session_state.total_cost += result.metrics.cost_usd
        st.rerun()
