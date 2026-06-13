"""
add_company.py — add one or more companies to the corpus at runtime.

For each ticker: validate against SEC EDGAR -> download its latest 10-K -> register it
-> chunk + embed + index just that ticker (the rest of the corpus is untouched).

Usage:
  python -m scripts.add_company TSLA
  python -m scripts.add_company TSLA META NFLX
"""

from __future__ import annotations

import sys

from finrag.config import load_config
from finrag.corpus import add_company
from finrag.edgar import EdgarError


def main() -> None:
    tickers = sys.argv[1:]
    if not tickers:
        print("usage: python -m scripts.add_company TICKER [TICKER ...]")
        raise SystemExit(2)

    cfg = load_config()
    for t in tickers:
        try:
            print(f"\n=== adding {t.upper()} ===")
            r = add_company(t, cfg=cfg)
            print(f"  {r['name']}  (10-K filed {r['filing_date']}, FY{r.get('fiscal_year')})")
            for strat in ("fixed", "semantic"):
                s = r[strat]
                print(f"  {strat:9}: {s['chunks']} chunks (tables={s['tables']}, avg_chars={s['avg_chars']})")
            print(f"  indexed ✓  -> ask about it now (symbol: {r['file'].split('/')[-1].split('_')[0]})")
        except (EdgarError, ValueError) as e:
            print(f"  ✗ skipped {t.upper()}: {e}")


if __name__ == "__main__":
    main()
