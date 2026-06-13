"""
remove_company.py — remove companies you added (vectors + BM25 records + files + entry).

The three seed telecoms are flagged "eval": true and are protected — removing them would
break the graded eval, so it refuses.

Usage:
  python -m scripts.remove_company TSLA
  python -m scripts.remove_company tsla meta
"""

from __future__ import annotations

import sys

from finrag.config import load_config
from finrag.corpus import remove_company


def main() -> None:
    symbols = sys.argv[1:]
    if not symbols:
        print("usage: python -m scripts.remove_company SYMBOL [SYMBOL ...]")
        raise SystemExit(2)

    cfg = load_config()
    for s in symbols:
        try:
            print(f"\n=== removing {s.lower()} ===")
            r = remove_company(s, cfg=cfg)
            print(f"  removed ✓  ({r['deleted_vectors']} vectors deleted across both namespaces)")
        except (KeyError, ValueError) as e:
            print(f"  ✗ {e}")


if __name__ == "__main__":
    main()
