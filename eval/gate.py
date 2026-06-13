"""
gate.py — regression gate for CI.

Loads the persisted eval results (results/*.json) for the gated arm and checks
them against config/thresholds.yaml. Exits non-zero if any gated metric falls
below its floor (minus the tolerance band) when gating.mode == "hard".

This gates on the COMMITTED eval numbers, so the workflow is: change the pipeline
-> re-run the eval -> commit the new results/. If the change regressed a metric,
this gate fails the PR. (A scheduled/manual workflow can re-run the live eval.)

Usage:  python -m eval.gate
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from finrag.config import load_config

RESULTS = Path("results")


def _load(*names: str) -> dict:
    for n in names:
        p = RESULTS / n
        if p.exists():
            return json.loads(p.read_text())
    raise FileNotFoundError(f"none of {names} found in {RESULTS}/ — run the eval first")


def main() -> int:
    cfg = load_config()
    th = cfg.thresholds
    arm = th.gated_config

    retr = _load("retrieval_latest.json")["arms"]
    gen = _load("generation_latest.json", "generation_baseline.json")["arms"]
    if arm not in retr or arm not in gen:
        print(f"GATE ERROR: gated arm '{arm}' missing from results."); return 2

    r, g = retr[arm], gen[arm]
    checks = [
        ("recall@k (hit@5)", r["hit@5"], th.retrieval.min_recall_at_k),
        ("mrr", r["mrr"], th.retrieval.min_mrr),
        ("ndcg@k", r["ndcg@5"], th.retrieval.min_ndcg_at_k),
        ("faithfulness", g["faithful"], th.generation.min_faithfulness),
        ("citation_coverage", g["cite_cov"], th.generation.min_citation_coverage),
    ]
    tol = th.gating.tolerance

    print(f"REGRESSION GATE  (arm={arm}, mode={th.gating.mode}, tolerance={tol})")
    print(f"  thresholds v{th.version}  |  run {cfg.manifest.run_id}\n")
    print(f"  {'metric':20} {'value':>8} {'floor':>8}  status")
    print("  " + "-" * 48)
    failures = []
    for name, value, floor in checks:
        ok = value >= floor - tol
        if not ok:
            failures.append(name)
        print(f"  {name:20} {value:>8.3f} {floor:>8.3f}  {'PASS' if ok else 'FAIL <—'}")

    if failures and th.gating.mode == "hard":
        print(f"\nGATE FAILED: {len(failures)} metric(s) below floor: {', '.join(failures)}")
        return 1
    if failures:
        print(f"\nGATE WARN: {len(failures)} below floor (mode=warn, not failing): {', '.join(failures)}")
    else:
        print("\nGATE PASSED: all gated metrics meet their floors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
