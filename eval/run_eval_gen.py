"""
run_eval_gen.py — generation-stage eval: correctness, citations, faithfulness, refusal.

For each arm and each golden item: retrieve (+rerank) -> assemble context -> generate.
Then score:
  - answer_correctness  (answerable): does the answer contain the verified figure?
  - false_refusal_rate  (answerable): did it wrongly refuse an answerable question?
  - citation_coverage   (answerable): fraction of answers with >=1 VALID chunk citation
  - faithfulness        (answerable): LLM-judge supported-claims ratio
  - refusal_accuracy    (negative):   did it correctly refuse the unanswerable ones?

LLM-heavy (generation + judge). Use --arms / --limit to scope cost while iterating.

Usage:
  python -m eval.run_eval_gen --arms semantic_norerank
  python -m eval.run_eval_gen                      # all 4 arms, all items
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from finrag.config import load_config
from finrag.embedding import Embedder
from finrag.generation import Generator
from finrag.reranking import Reranker
from finrag.retrieval import HybridRetriever
from finrag.vectorstore import VectorStore
from eval.faithfulness import FaithfulnessJudge

GOLDEN = Path("data/golden/golden.jsonl")
RESULTS_DIR = Path("results")
ARMS = {
    "fixed_norerank": ("fixed", False),
    "fixed_rerank": ("fixed", True),
    "semantic_norerank": ("semantic", False),
    "semantic_rerank": ("semantic", True),
}
REFUSAL_MARK = "does not contain enough information"


def is_refusal(answer: str) -> bool:
    return REFUSAL_MARK in answer.lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    ap.add_argument("--limit", type=int, help="only the first N golden items (quick runs)")
    ap.add_argument("--no-faithfulness", action="store_true", help="skip the LLM judge (faster)")
    args = ap.parse_args()

    cfg = load_config()
    gold = [json.loads(l) for l in GOLDEN.read_text().splitlines() if l.strip()]
    if args.limit:
        gold = gold[: args.limit]
    answerable = [g for g in gold if g["answerable"]]
    negatives = [g for g in gold if not g["answerable"]]

    emb = Embedder(cfg.models.embedding)
    retriever = HybridRetriever(cfg, emb, VectorStore(cfg))
    reranker = Reranker(cfg)
    generator = Generator(cfg)
    judge = None if args.no_faithfulness else FaithfulnessJudge(cfg)

    print("RUN MANIFEST"); print(cfg.manifest.summary(), "\n")
    cols = ["correct", "false_refuse", "cite_cov", "faithful", "refuse_acc"]
    print(f"{'arm':20} " + " ".join(f"{c:>12}" for c in cols))
    print("-" * (20 + 13 * len(cols)))

    results: dict[str, dict] = {}
    for arm in args.arms:
        strategy, use_rr = ARMS[arm]
        correct, false_ref, cite_cov, faiths = [], [], [], []
        for item in answerable:
            fused = retriever.retrieve(item["question"], strategy, item["company"])
            final = reranker.rerank(item["question"], fused) if use_rr \
                else fused[: cfg.pipeline.reranking.output_top_n]
            context = generator.assemble_context(final)
            answer = generator.generate(item["question"], final)

            refused = is_refusal(answer)
            false_ref.append(1.0 if refused else 0.0)
            correct.append(1.0 if (not refused and any(k in answer for k in item["answer_keys"])) else 0.0)
            cited = set(re.findall(r"\[chunk:([^\]]+)\]", answer))
            chunk_ids = {c.id for c in final}
            cite_cov.append(1.0 if (cited and cited <= chunk_ids) else 0.0)
            if judge and not refused:
                f = judge.score(answer, context)["faithfulness"]
                if f is not None:
                    faiths.append(f)

        refuse_acc = []
        for item in negatives:
            fused = retriever.retrieve(item["question"], strategy, item["company"])
            final = reranker.rerank(item["question"], fused) if use_rr \
                else fused[: cfg.pipeline.reranking.output_top_n]
            answer = generator.generate(item["question"], final)
            refuse_acc.append(1.0 if is_refusal(answer) else 0.0)

        def mean(xs):
            return sum(xs) / len(xs) if xs else float("nan")

        row = [mean(correct), mean(false_ref), mean(cite_cov), mean(faiths), mean(refuse_acc)]
        results[arm] = dict(zip(cols, row))
        print(f"{arm:20} " + " ".join(f"{v:>12.3f}" for v in row))

    # Persist (only when the full arm set + all items ran, so saved files are comparable).
    if set(args.arms) == set(ARMS) and not args.limit:
        RESULTS_DIR.mkdir(exist_ok=True)
        out = {
            "stage": "generation", "run_id": cfg.manifest.run_id,
            "versions": cfg.manifest.versions,
            "reranker_model": cfg.models.reranker.model,
            "judge_model": cfg.models.generation.model,
            "n_answerable": len(answerable), "n_negative": len(negatives),
            "arms": results,
        }
        path = RESULTS_DIR / f"generation_{cfg.manifest.run_id}.json"
        path.write_text(json.dumps(out, indent=2))
        (RESULTS_DIR / "generation_latest.json").write_text(json.dumps(out, indent=2))
        print(f"\nSaved: {path}")

    emb.close()


if __name__ == "__main__":
    main()
