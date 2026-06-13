"""
run_ragas.py — RAGAS-style RAG metrics, computed in-house with an OpenAI judge.

Reports the three metrics most RAG projects publish, using RAGAS's published
definitions so the numbers are directly comparable across teams — but WITHOUT the
heavy/conflicting `ragas` package (it pins langchain versions that clash with our
langgraph stack on Python 3.14). The judge is gpt-4o-mini, matching the reference kit.

  - faithfulness      : supported answer-claims / total answer-claims
  - context_recall    : reference-claims supported by the retrieved context / total
  - context_precision : RAGAS weighted precision@k over the retrieved chunks'
                        relevance to the question + reference answer

Three configurations (fixed / semantic / semantic+rerank) over the answerable golden
questions; answers generated with gpt-4o-mini.

Needs only OPENAI_API_KEY (uses the core `openai` client — no extra install).
Usage:  python -m eval.run_ragas            # all 3 arms
        python -m eval.run_ragas --arms semantic_norerank
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from openai import OpenAI

from finrag.chat import ChatEngine, available_models

GOLDEN = Path("data/golden/golden.jsonl")
OUT = Path("results/ragas.json")
ANSWER_MODEL_ID = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o-mini"
ARMS = {
    "fixed_norerank": ("fixed", False),
    "semantic_norerank": ("semantic", False),
    "semantic_rerank": ("semantic", True),
}
METRIC_COLS = ["faithfulness", "context_precision", "context_recall"]


def _json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


class Judge:
    """gpt-4o-mini judge for the RAGAS-style metric primitives."""

    def __init__(self) -> None:
        self.client = OpenAI()           # OpenAI default base_url + OPENAI_API_KEY
        self.model = JUDGE_MODEL

    def _ask(self, system: str, user: str) -> str:
        r = self.client.chat.completions.create(
            model=self.model, temperature=0, max_tokens=512,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return r.choices[0].message.content.strip()

    def claims(self, text: str) -> list[str]:
        out = self._ask(
            "Break the text into a list of standalone, individually-checkable factual claims.",
            f'Text:\n"""{text}"""\nReturn JSON: {{"claims": ["...", "..."]}}')
        return _json(out).get("claims", [])

    def supported(self, context: str, claim: str) -> bool:
        out = self._ask(
            "You verify whether a claim can be inferred from a context. Answer strictly.",
            f'Context:\n"""{context}"""\n\nClaim: "{claim}"\n'
            'Can the claim be directly inferred from the context? JSON: {"verdict":"yes"|"no"}')
        return _json(out).get("verdict", "no").lower() == "yes"

    def relevant(self, question: str, reference: str, passage: str) -> bool:
        out = self._ask(
            "You judge whether a retrieved passage is useful for answering a question.",
            f'Question: "{question}"\nReference answer: "{reference}"\n'
            f'Passage:\n"""{passage}"""\n'
            'Is this passage useful for arriving at the reference answer? JSON: {"verdict":"yes"|"no"}')
        return _json(out).get("verdict", "no").lower() == "yes"


def faithfulness(j: Judge, answer: str, context: str) -> float | None:
    claims = j.claims(answer)
    if not claims:
        return None                       # a refusal makes no claims → excluded
    return sum(j.supported(context, c) for c in claims) / len(claims)


def context_recall(j: Judge, reference: str, context: str) -> float | None:
    claims = j.claims(reference)
    if not claims:
        return None
    return sum(j.supported(context, c) for c in claims) / len(claims)


def context_precision(j: Judge, question: str, reference: str, contexts: list[str]) -> float:
    rels = [j.relevant(question, reference, c) for c in contexts]
    if not any(rels):
        return 0.0
    # RAGAS weighted precision@k: mean of precision@k taken at each relevant rank.
    num, hits = 0.0, 0
    for k, r in enumerate(rels, start=1):
        if r:
            hits += 1
            num += hits / k
    return num / sum(rels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    args = ap.parse_args()
    if not (os.getenv("OPENAI_API_KEY") or "").strip():
        raise SystemExit("RAGAS run needs OPENAI_API_KEY (answer + judge are gpt-4o-mini).")

    eng = ChatEngine()
    model = next((m for m in available_models() if m.model == ANSWER_MODEL_ID), None)
    if model is None:
        raise SystemExit(f"{ANSWER_MODEL_ID} not available — set OPENAI_API_KEY in .env.")
    judge = Judge()

    gold = [json.loads(l) for l in GOLDEN.read_text().splitlines() if l.strip()]
    gold = [g for g in gold if g.get("answerable")]
    print(f"RAGAS-style · {len(gold)} answerable questions · answer+judge={ANSWER_MODEL_ID}\n")
    print(f"{'arm':20} " + " ".join(f"{c:>18}" for c in METRIC_COLS))
    print("-" * (20 + 19 * len(METRIC_COLS)))

    results = {}
    for arm in args.arms:
        strategy, use_rr = ARMS[arm]
        faiths, recalls, precs = [], [], []
        for g in gold:
            res = eng.answer(g["question"], model, strategy=strategy,
                             use_reranker=use_rr, company=g.get("company"))
            ctx = "\n\n".join(c.text for c in res.chunks)
            ctx_list = [c.text for c in res.chunks] or [""]
            ref = g.get("expected_answer") or (g.get("answer_keys") or [""])[0]
            f = faithfulness(judge, res.answer, ctx)
            if f is not None:
                faiths.append(f)
            recalls.append(context_recall(judge, ref, ctx) or 0.0)
            precs.append(context_precision(judge, g["question"], ref, ctx_list))

        def mean(xs):
            return round(sum(xs) / len(xs), 3) if xs else float("nan")

        scores = {"faithfulness": mean(faiths), "context_precision": mean(precs),
                  "context_recall": mean(recalls)}
        results[arm] = scores
        print(f"{arm:20} " + " ".join(f"{scores[c]:>18.3f}" for c in METRIC_COLS))
    eng.close()

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(
        {"stage": "ragas", "note": "RAGAS-style metrics computed in-house (gpt-4o-mini judge)",
         "answer_model": ANSWER_MODEL_ID, "judge_model": JUDGE_MODEL,
         "n_questions": len(gold), "metrics": METRIC_COLS, "arms": results}, indent=2))
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
