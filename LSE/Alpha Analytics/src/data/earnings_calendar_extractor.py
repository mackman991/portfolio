"""Historical earnings-calendar extractor (Financial Modeling Prep).

Hits ``/api/v3/historical/earning_calendar/{ticker}`` which — unlike the
``earnings-surprises`` endpoint used by :mod:`src.data.eps_extractor` — carries
an explicit ``time`` field tagging each release as ``bmo`` (before market open),
``amc`` (after market close), or ``dmh`` (during market hours).

This extractor is used by :mod:`src.analysis.verify_eps_dates` to validate the
``eps_date`` column we feed into the strategies, and to flag events whose
current entry-price (eps_date close) sits on the *wrong side* of the
announcement (an AMC release should really enter the next trading day).
"""
from __future__ import annotations

import logging
import time
from typing import Iterable, Optional, Union

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class EarningsCalendarExtractor:
    """Fetch FMP's historical earnings calendar (with bmo/amc timing)."""

    ENDPOINT = "https://financialmodelingprep.com/api/v3/historical/earning_calendar/{ticker}"

    # Canonical normalisation of the ``time`` field. FMP is occasionally
    # inconsistent (capitalisation, spacing, the very occasional "--").
    TIME_MAP = {
        "bmo": "bmo",
        "amc": "amc",
        "dmh": "dmh",
        "--": None,
        "": None,
    }

    def __init__(
        self,
        fmp_api_key: str,
        rate_limit_seconds: float = 0.25,
        request_timeout: int = 30,
    ):
        if not fmp_api_key:
            raise ValueError("FMP API key is required.")
        self.fmp_api_key = fmp_api_key
        self.rate_limit_seconds = rate_limit_seconds
        self.request_timeout = request_timeout

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _fetch_one(self, ticker: str) -> pd.DataFrame:
        url = f"{self.ENDPOINT.format(ticker=ticker)}?apikey={self.fmp_api_key}"
        resp = requests.get(url, timeout=self.request_timeout)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            raise ValueError(f"Unexpected FMP response for {ticker}: {data}")
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df.columns = [c.strip().lower() for c in df.columns]
        df["ticker"] = ticker
        return df

    @classmethod
    def _normalise(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Coerce dtypes and normalise the ``time`` field."""
        if df.empty:
            return df

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        if "fiscaldateending" in df.columns:
            df["fiscaldateending"] = pd.to_datetime(
                df["fiscaldateending"], errors="coerce"
            ).dt.normalize()
        if "updatedfromdate" in df.columns:
            df["updatedfromdate"] = pd.to_datetime(
                df["updatedfromdate"], errors="coerce"
            ).dt.normalize()

        if "time" in df.columns:
            df["time_raw"] = df["time"]
            df["time"] = (
                df["time"].astype(str).str.strip().str.lower()
                .map(cls.TIME_MAP)
            )
        else:
            df["time"] = None

        # Convenient boolean helpers.
        df["is_amc"] = df["time"].eq("amc")
        df["is_bmo"] = df["time"].eq("bmo")

        return df

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def extract(
        self,
        tickers: Union[str, Iterable[str]],
        save_csv: Optional[str] = None,
    ) -> pd.DataFrame:
        """Pull the historical earnings calendar for each ticker.

        Parameters
        ----------
        tickers : str or iterable of str
            One or more ticker symbols (FMP format — e.g. ``"BRK-B"``).
        save_csv : optional path
            If provided, write the merged long-form frame to this CSV.

        Returns
        -------
        pd.DataFrame
            Long-form, keyed on (ticker, date). Columns include
            ``eps``, ``epsEstimated``, ``revenue``, ``revenueEstimated``,
            ``time`` (normalised to bmo/amc/dmh/None), ``is_amc``, ``is_bmo``.
        """
        if isinstance(tickers, str):
            tickers = [tickers]

        frames: list[pd.DataFrame] = []
        for ticker in tickers:
            logger.info("Extracting earnings calendar for %s", ticker)
            try:
                df = self._fetch_one(ticker)
            except Exception as exc:  # noqa: BLE001 - per-ticker resilience
                logger.error("  failed %s: %s", ticker, exc)
                time.sleep(self.rate_limit_seconds)
                continue

            if df.empty:
                logger.warning("  empty payload for %s", ticker)
                time.sleep(self.rate_limit_seconds)
                continue

            df = self._normalise(df)
            logger.info(
                "  rows=%d (bmo=%d, amc=%d, dmh=%d, unknown=%d) for %s",
                len(df),
                int(df["time"].eq("bmo").sum()),
                int(df["time"].eq("amc").sum()),
                int(df["time"].eq("dmh").sum()),
                int(df["time"].isna().sum()),
                ticker,
            )
            frames.append(df)
            time.sleep(self.rate_limit_seconds)

        if not frames:
            logger.error("No earnings-calendar rows fetched for any ticker.")
            return pd.DataFrame()

        out = (
            pd.concat(frames, ignore_index=True)
            .sort_values(["ticker", "date"])
            .reset_index(drop=True)
        )

        if save_csv:
            out.to_csv(save_csv, index=False)
            logger.info("Saved earnings calendar to %s (rows=%d)", save_csv, len(out))

        return out


# ---------------------------------------------------------------------- #
# CLI entry point
# ---------------------------------------------------------------------- #
def main() -> None:  # pragma: no cover - thin wrapper
    import argparse
    import os

    from dotenv import load_dotenv

    from src.config import SP100_TICKERS

    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="data/earnings_calendar_sp100.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Override ticker list (default: S&P 100 from config).",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.25,
        help="Seconds between requests (FMP Starter: 300 req/min → 0.25s).",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    api_key = os.environ.get("FMP_API_KEY")
    if not api_key:
        raise SystemExit("FMP_API_KEY not set in environment (.env).")

    tickers = args.tickers if args.tickers else list(SP100_TICKERS)

    extractor = EarningsCalendarExtractor(
        fmp_api_key=api_key, rate_limit_seconds=args.rate_limit
    )
    extractor.extract(tickers=tickers, save_csv=args.out)


if __name__ == "__main__":  # pragma: no cover
    main()
