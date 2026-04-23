"""
Extract holdings of the iShares MSCI World UCITS ETF (ISIN: IE00B0M62Q58)
as of 31 March 2026 from Financial Modeling Prep (FMP).

FMP's ETF endpoints expect a ticker symbol, not an ISIN. IE00B0M62Q58 is
the USD-distributing iShares MSCI World UCITS ETF; on FMP it is typically
exposed under one of its listing tickers. We resolve the ISIN -> symbol
first via FMP's search endpoint, then fetch holdings for the requested
portfolio date.

Usage:
    export FMP_API_KEY="your_key_here"
    python fmp_ishares_msci_world_holdings.py

Outputs:
    holdings_IE00B0M62Q58_2026-03-31.csv
"""

from __future__ import annotations

import csv
import os
import sys
from datetime import date
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

ISIN = "IE00B0M62Q58"           # iShares MSCI World UCITS ETF (Dist)
PORTFOLIO_DATE = date(2026, 3, 31)
FALLBACK_SYMBOLS = ["IWRD.L", "IWRD.AS", "IWRD.SW", "IWRD.MI", "IWRD"]
BASE = "https://financialmodelingprep.com"
TIMEOUT = 30


def _get(url: str, params: dict[str, Any]) -> Any:
    """GET helper that raises on HTTP error and returns parsed JSON."""
    r = requests.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def resolve_symbol(isin: str, api_key: str) -> str:
    """Look up the FMP ticker for a given ISIN, falling back to known tickers."""
    try:
        hits = _get(
            f"{BASE}/api/v3/search",
            {"query": isin, "limit": 10, "apikey": api_key},
        )
        if isinstance(hits, list) and hits:
            # Prefer exact ISIN match if FMP returns it, otherwise first hit.
            for h in hits:
                if h.get("isin") == isin and h.get("symbol"):
                    return h["symbol"]
            if hits[0].get("symbol"):
                return hits[0]["symbol"]
    except requests.HTTPError as exc:
        print(f"[warn] ISIN search failed: {exc}", file=sys.stderr)

    # Fallback: try well-known listings of this ETF.
    for sym in FALLBACK_SYMBOLS:
        try:
            quote = _get(f"{BASE}/api/v3/quote/{sym}", {"apikey": api_key})
            if isinstance(quote, list) and quote:
                return sym
        except requests.HTTPError:
            continue

    raise RuntimeError(f"Could not resolve a tradable FMP symbol for ISIN {isin}")


def list_portfolio_dates(symbol: str, api_key: str) -> list[str]:
    """Return all portfolio dates FMP has on file for the given ETF symbol."""
    data = _get(
        f"{BASE}/api/v4/etf-holdings/portfolio-date",
        {"symbol": symbol, "apikey": api_key},
    )
    if not isinstance(data, list):
        return []
    return [row["date"] for row in data if "date" in row]


def fetch_holdings(symbol: str, as_of: date, api_key: str) -> list[dict[str, Any]]:
    """Fetch holdings via v4 historical endpoint, falling back to v3 current holdings."""
    try:
        rows = _get(
            f"{BASE}/api/v4/etf-holdings",
            {"date": as_of.isoformat(), "symbol": symbol, "apikey": api_key},
        )
        if isinstance(rows, list) and rows:
            return rows
    except requests.HTTPError as exc:
        print(f"[warn] v4 holdings failed: {exc}", file=sys.stderr)

    print("[info] falling back to v3 etf-holder endpoint (current holdings, no date filter)")
    rows = _get(f"{BASE}/api/v3/etf-holder/{symbol}", {"apikey": api_key})
    return rows if isinstance(rows, list) else []


def enrich_with_profiles(holdings: list[dict[str, Any]], api_key: str, batch_size: int = 50) -> list[dict[str, Any]]:
    """Add sector, industry, and country to each holding via batched profile calls."""
    symbols = [h["asset"] for h in holdings if h.get("asset")]
    profile_map: dict[str, dict] = {}

    for i in range(0, len(symbols), batch_size):
        batch = ",".join(symbols[i : i + batch_size])
        try:
            results = _get(f"{BASE}/api/v3/profile/{batch}", {"apikey": api_key})
            if isinstance(results, list):
                for p in results:
                    profile_map[p["symbol"]] = p
        except requests.HTTPError as exc:
            print(f"[warn] profile batch failed at offset {i}: {exc}", file=sys.stderr)
        if i % 500 == 0 and i > 0:
            print(f"[info] enriched {i}/{len(symbols)} holdings...")

    for h in holdings:
        p = profile_map.get(h.get("asset", ""), {})
        h["sector"] = p.get("sector", "")
        h["industry"] = p.get("industry", "")
        h["country"] = p.get("country", "")

    return holdings


def write_csv(rows: list[dict[str, Any]], path: str) -> None:
    if not rows:
        print("[warn] no holdings returned; CSV will be empty", file=sys.stderr)
        with open(path, "w") as f:
            f.write("")
        return
    # Union of keys preserves any extra fields FMP adds over time.
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    api_key = os.environ.get("FMP_API_KEY")
    if not api_key:
        print("ERROR: set FMP_API_KEY in your environment.", file=sys.stderr)
        return 2

    symbol = resolve_symbol(ISIN, api_key)
    print(f"Resolved {ISIN} -> {symbol}")

    available = list_portfolio_dates(symbol, api_key)
    target = PORTFOLIO_DATE.isoformat()
    if available and target not in available:
        # Snap to the closest available date on/before the target.
        earlier = sorted(d for d in available if d <= target)
        if not earlier:
            print(
                f"ERROR: no portfolio dates on or before {target} for {symbol}. "
                f"Available sample: {available[:5]}",
                file=sys.stderr,
            )
            return 1
        snapped = earlier[-1]
        print(f"[info] {target} not published; using nearest available {snapped}")
        as_of = date.fromisoformat(snapped)
    else:
        as_of = PORTFOLIO_DATE

    holdings = fetch_holdings(symbol, as_of, api_key)
    print(f"Retrieved {len(holdings)} holdings as of {as_of.isoformat()}")
    print("Enriching with sector/industry/country...")
    holdings = enrich_with_profiles(holdings, api_key)

    out_path = f"holdings_{ISIN}_{as_of.isoformat()}.csv"
    write_csv(holdings, out_path)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())