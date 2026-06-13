"""
edgar.py — minimal SEC EDGAR client: resolve a ticker, find its latest 10-K, download it.

Used by the corpus-management layer (add a company at runtime). Pure HTTP + JSON; no
scraping. SEC requires a descriptive User-Agent with a contact email or it blocks you —
set SEC_USER_AGENT in the environment to override the default.

Three calls:
  resolve_ticker("AAPL")  -> {cik, ticker, title}     (via the public ticker map)
  latest_10k(cik)         -> {accession, primary_doc, filing_date}
  download_filing(cik, accession, primary_doc) -> html text
"""

from __future__ import annotations

import os

import requests

USER_AGENT = os.getenv("SEC_USER_AGENT", "FinRAG research (aadhavanalakan97@gmail.com)")
HEADERS = {"User-Agent": USER_AGENT}
TIMEOUT = 30

_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

_ticker_cache: dict[str, dict] | None = None


class EdgarError(Exception):
    """A clear, user-facing failure (unknown ticker, no 10-K, network)."""


def _get(url: str) -> requests.Response:
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        return r
    except requests.RequestException as e:  # noqa: BLE001
        raise EdgarError(f"SEC request failed ({url}): {e}") from e


def _ticker_map() -> dict[str, dict]:
    """Lazy-load + cache the public ticker→CIK map ({TICKER: {cik, title}})."""
    global _ticker_cache
    if _ticker_cache is None:
        data = _get(_TICKER_MAP_URL).json()
        _ticker_cache = {
            row["ticker"].upper(): {"cik": str(row["cik_str"]).zfill(10), "title": row["title"]}
            for row in data.values()
        }
    return _ticker_cache


def resolve_ticker(ticker: str) -> dict:
    """Validate a ticker against EDGAR; return {ticker, cik, title} or raise EdgarError."""
    t = (ticker or "").strip().upper()
    if not t:
        raise EdgarError("Empty ticker.")
    hit = _ticker_map().get(t)
    if not hit:
        raise EdgarError(
            f"'{t}' is not a recognized US-listed ticker in SEC's registry. "
            "Check the symbol (foreign filers without a US listing won't appear)."
        )
    return {"ticker": t, "cik": hit["cik"], "title": hit["title"]}


def latest_10k(cik: str) -> dict:
    """Find the most recent 10-K for a (zero-padded) CIK. Raises if the filer has none."""
    cik_padded = str(cik).zfill(10)
    data = _get(_SUBMISSIONS_URL.format(cik=cik_padded)).json()
    recent = data["filings"]["recent"]
    for i, form in enumerate(recent["form"]):
        if form == "10-K":
            return {
                "accession": recent["accessionNumber"][i],
                "primary_doc": recent["primaryDocument"][i],
                "filing_date": recent["filingDate"][i],
            }
    raise EdgarError(
        f"No 10-K found for CIK {cik_padded}. Foreign private issuers file 20-F instead "
        "and are not supported."
    )


def download_filing(cik: str, accession: str, primary_doc: str) -> str:
    """Download the primary 10-K HTML document and return its text."""
    accession_nodash = accession.replace("-", "")
    url = (f"https://www.sec.gov/Archives/edgar/data/"
           f"{int(cik)}/{accession_nodash}/{primary_doc}")
    return _get(url).text


def fetch_latest_10k_html(ticker: str) -> dict:
    """Convenience: ticker -> {ticker, cik, title, filing_date, html}. One call, validated."""
    info = resolve_ticker(ticker)
    filing = latest_10k(info["cik"])
    html = download_filing(info["cik"], filing["accession"], filing["primary_doc"])
    return {**info, "filing_date": filing["filing_date"], "html": html}


if __name__ == "__main__":
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    info = resolve_ticker(sym)
    filing = latest_10k(info["cik"])
    print(f"{info['ticker']}  CIK {info['cik']}  {info['title']}")
    print(f"  latest 10-K: {filing['filing_date']}  doc={filing['primary_doc']}")
