"""
ask.py — ask the RAG pipeline a question from the command line.

Usage:
  python -m scripts.ask "What were Verizon's total operating revenues in 2025?"
  python -m scripts.ask "Compare AT&T and Verizon capex" --strategy fixed
  python -m scripts.ask "..." --no-rerank --company verizon
"""

from __future__ import annotations

import argparse

from finrag.pipeline import RAGPipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--strategy", choices=["fixed", "semantic"], default="semantic")
    ap.add_argument("--rerank", action="store_true",
                    help="enable the cross-encoder reranker (off by default — it hurts on this corpus)")
    ap.add_argument("--company", choices=["verizon", "att", "tmobile"], help="force a company filter")
    args = ap.parse_args()

    pipe = RAGPipeline(strategy=args.strategy, use_reranker=args.rerank)
    result = pipe.ask(args.question, company=args.company)

    print(f"\nstrategy={result.strategy}  reranked={result.reranked}  "
          f"company={result.company or 'all'}  run={result.run_id}")
    print("\n" + "=" * 70 + "\nANSWER\n" + "=" * 70)
    print(result.answer)
    print("\n" + "=" * 70 + f"\nSOURCES ({len(result.chunks)} chunks used)\n" + "=" * 70)
    for r in result.chunks:
        tag = "TABLE" if r.is_table else "prose"
        print(f"  [{r.id}] {tag} · {r.section} · {r.text[:80].replace(chr(10), ' ')}…")
    print(f"\nCitations in answer: {result.citations or '(none)'}")
    pipe.close()


if __name__ == "__main__":
    main()
