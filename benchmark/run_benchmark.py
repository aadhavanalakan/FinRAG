"""
run_benchmark.py — speed/cost benchmark across Nebius models.

The brief's "model + quantization study" was local-first; since we run hosted on
Nebius, this is the realistic equivalent: compare candidate generation models on
the metrics you CAN measure over an API —

  - TTFT       time to first token (streaming)
  - tok/s      output throughput  (output_tokens / generation_time)
  - latency    p50 / p95 total wall-clock
  - out_toks   mean output tokens
  - cost       output_tokens x price  (price per 1M from --price-file, if given)

Output quality is measured separately by the eval harness (run_eval*), not here.

Usage:
  python -m benchmark.run_benchmark
  python -m benchmark.run_benchmark --models Qwen/Qwen3-30B-A3B-Instruct-2507 google/gemma-3-27b-it
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from openai import OpenAI

from finrag.config import load_config, resolve_api_key

RESULTS_DIR = Path("results")

# Self-contained generation prompts (no retrieval) — a fixed, standardized set.
PROMPTS = [
    "Explain the difference between operating income and net income in one paragraph.",
    "What is free cash flow and why do telecom investors watch it?",
    "Summarize what a 10-K filing is for someone new to finance.",
    "Define capital expenditures and give a telecom example.",
    "In two sentences, explain what EBITDA measures.",
    "What does 'net income attributable to noncontrolling interests' mean?",
    "Explain why revenue can grow while net income falls.",
    "What is the difference between service revenue and equipment revenue for a carrier?",
    "Briefly explain what a cross-encoder reranker does in a RAG system.",
    "What is the purpose of segment reporting in a 10-K?",
]

DEFAULT_MODELS = [
    "Qwen/Qwen3-30B-A3B-Instruct-2507",   # the pipeline's choice
    "google/gemma-3-27b-it",
    "meta-llama/Llama-3.3-70B-Instruct",
    "openai/gpt-oss-120b",
]


def bench_call(client: OpenAI, model: str, prompt: str, max_tokens: int) -> dict:
    start = time.perf_counter()
    first_t = None
    text_parts: list[str] = []
    usage = None
    stream = client.chat.completions.create(
        model=model, temperature=0, max_tokens=max_tokens, stream=True,
        stream_options={"include_usage": True},
        messages=[{"role": "user", "content": prompt}],
    )
    for ev in stream:
        if ev.choices and ev.choices[0].delta and ev.choices[0].delta.content:
            if first_t is None:
                first_t = time.perf_counter()
            text_parts.append(ev.choices[0].delta.content)
        if getattr(ev, "usage", None):
            usage = ev.usage
    end = time.perf_counter()

    out_toks = usage.completion_tokens if usage else max(1, len("".join(text_parts)) // 4)
    ttft = (first_t - start) if first_t else (end - start)
    gen_time = max(1e-6, end - (first_t or start))
    return {"ttft": ttft, "latency": end - start, "out_toks": out_toks, "tok_s": out_toks / gen_time}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--price-file", help="JSON {model: usd_per_1M_output_tokens} for cost column")
    args = ap.parse_args()

    cfg = load_config()
    client = OpenAI(base_url=cfg.models.generation.base_url, api_key=resolve_api_key("NEBIUS_API_KEY"))
    prices = json.loads(Path(args.price_file).read_text()) if args.price_file else {}

    print(f"Benchmark: {len(PROMPTS)} prompts x {len(args.models)} models, max_tokens={args.max_tokens}\n")
    cols = ["ttft_p50", "ttft_p95", "tok/s", "lat_p50", "lat_p95", "out_toks"]
    print(f"{'model':40} " + " ".join(f"{c:>9}" for c in cols) + f" {'cost$':>8}")
    print("-" * (40 + 10 * len(cols) + 9))

    results = {}
    for model in args.models:
        calls = []
        for p in PROMPTS:
            try:
                calls.append(bench_call(client, model, p, args.max_tokens))
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] {model} failed on a prompt: {str(e)[:60]}")
        if not calls:
            continue
        def pct(key, q):
            return statistics.quantiles([c[key] for c in calls], n=100)[q - 1] if len(calls) > 1 else calls[0][key]
        total_out = sum(c["out_toks"] for c in calls)
        agg = {
            "ttft_p50": statistics.median(c["ttft"] for c in calls),
            "ttft_p95": pct("ttft", 95),
            "tok_s": statistics.median(c["tok_s"] for c in calls),
            "lat_p50": statistics.median(c["latency"] for c in calls),
            "lat_p95": pct("latency", 95),
            "out_toks": total_out / len(calls),
            "total_out_toks": total_out,
        }
        cost = (total_out / 1_000_000) * prices[model] if model in prices else None
        agg["cost_usd"] = cost
        results[model] = agg
        cost_s = f"{cost:>8.4f}" if cost is not None else f"{'n/a':>8}"
        print(f"{model:40} {agg['ttft_p50']:>9.3f} {agg['ttft_p95']:>9.3f} {agg['tok_s']:>9.1f} "
              f"{agg['lat_p50']:>9.3f} {agg['lat_p95']:>9.3f} {agg['out_toks']:>9.0f} {cost_s}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = {"stage": "benchmark", "n_prompts": len(PROMPTS), "max_tokens": args.max_tokens,
           "base_url": cfg.models.generation.base_url, "models": results}
    (RESULTS_DIR / "benchmark.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {RESULTS_DIR / 'benchmark.json'}")
    print("Cost column is blank unless --price-file gives USD/1M output tokens (from your Nebius pricing page).")


if __name__ == "__main__":
    main()
