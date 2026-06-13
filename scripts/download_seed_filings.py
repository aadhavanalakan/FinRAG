"""
download_seed_filings.py — (re)fetch the seed telecom 10-K HTML into data/.

The three seed filings (Verizon, AT&T, T-Mobile) ship in data/, so you normally don't
need this. Run it to refresh them from SEC EDGAR — it reuses the same client that
`add_company` uses and pulls the companies flagged `"eval": true` in data/companies.json.

Usage:  python -m scripts.download_seed_filings
"""

from __future__ import annotations

from pathlib import Path

from finrag.corpus import load_registry
from finrag.edgar import fetch_latest_10k_html


def main() -> None:
    reg = load_registry()
    seed = {s: m for s, m in reg.items() if m.get("eval")}
    print(f"Fetching {len(seed)} seed 10-K filing(s) from SEC EDGAR…\n")
    for sym, meta in seed.items():
        info = fetch_latest_10k_html(meta["ticker"])
        Path(meta["file"]).write_text(info["html"], encoding="utf-8")
        print(f"  {sym:8} {meta['ticker']:5} filed {info['filing_date']}  "
              f"-> {meta['file']} ({len(info['html']):,} chars)")
    print("\nDone. Next: python -m scripts.ingest  (parse -> chunk -> embed -> upsert).")


if __name__ == "__main__":
    main()
