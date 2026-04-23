"""FMP technical-indicator extractor.

Hits the `/api/v3/technical_indicator/{timeframe}/{ticker}` endpoint for each
(ticker, indicator_type, period) triple and merges the results into a single
long-form DataFrame keyed on (date, ticker).

Allowed indicator types (per FMP docs, as of 2026-04):
    sma, ema, wma, dema, tema, williams, rsi, adx, standardDeviation

MACD is NOT served directly by this endpoint — compute it locally from the
12- and 26-period EMAs that this extractor returns.
"""
from __future__ import annotations

import logging
import time
from typing import Iterable, List, Optional, Tuple, Union

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class TechnicalIndicatorExtractor:
    """Fetch FMP-computed technical indicators (RSI, SMA, EMA, ...)."""

    ENDPOINT = "https://financialmodelingprep.com/api/v3/technical_indicator/{timeframe}/{ticker}"
    ALLOWED_TYPES = {
        "sma", "ema", "wma", "dema", "tema",
        "williams", "rsi", "adx", "standardDeviation",
    }

    def __init__(
        self,
        fmp_api_key: str,
        timeframe: str = "daily",
        rate_limit_seconds: float = 0.25,
        request_timeout: int = 30,
    ):
        if not fmp_api_key:
            raise ValueError("FMP API key is required.")
        self.fmp_api_key = fmp_api_key
        self.timeframe = timeframe
        self.rate_limit_seconds = rate_limit_seconds
        self.request_timeout = request_timeout

    # --------------------------------------------------------------------- #
    # Internals
    # --------------------------------------------------------------------- #
    def _fetch_one(self, ticker: str, indicator_type: str, period: int) -> pd.DataFrame:
        """Fetch one (ticker, type, period) and return a tidy 3-col frame:
        [date, ticker, <type>_<period>]."""
        url = (
            f"{self.ENDPOINT.format(timeframe=self.timeframe, ticker=ticker)}"
            f"?type={indicator_type}&period={period}&apikey={self.fmp_api_key}"
        )
        resp = requests.get(url, timeout=self.request_timeout)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            raise ValueError(
                f"Unexpected FMP response for {ticker}/{indicator_type}({period}): {data}"
            )

        df = pd.DataFrame(data)
        col_suffix = f"{indicator_type.lower()}_{period}"
        if df.empty:
            return df

        df.columns = [c.strip().lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["ticker"] = ticker

        # FMP returns the indicator in a column whose name equals the type
        # (e.g. "rsi", "sma"). Rename to include the period so multiple
        # lookbacks (e.g. sma_20, sma_50) coexist after merging.
        indicator_col = indicator_type.lower()
        if indicator_col in df.columns:
            df = df.rename(columns={indicator_col: col_suffix})
        elif col_suffix not in df.columns:
            logger.warning(
                "Indicator column missing in FMP payload for %s/%s(%d). "
                "Columns returned: %s", ticker, indicator_type, period, list(df.columns),
            )
            return pd.DataFrame()

        return df[["date", "ticker", col_suffix]]

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def extract(
        self,
        tickers: Union[str, Iterable[str]],
        indicators: List[Tuple[str, int]],
        save_csv: Optional[str] = None,
    ) -> pd.DataFrame:
        """Pull the supplied indicators for each ticker and merge on (date, ticker).

        Parameters
        ----------
        tickers : str or iterable of str
            One or more ticker symbols (FMP format — e.g. use "BRK-B" not "BRK.B").
        indicators : list[tuple[str, int]]
            Pairs of (indicator_type, period). Types must be in `ALLOWED_TYPES`.
        save_csv : optional path
            If provided, the merged long-form frame is written to this CSV.

        Returns
        -------
        pd.DataFrame
            Long-form on (date, ticker). One column per requested indicator,
            named `{type}_{period}` (e.g. `rsi_14`, `sma_50`, `ema_12`).
        """
        if isinstance(tickers, str):
            tickers = [tickers]

        for itype, _ in indicators:
            if itype not in self.ALLOWED_TYPES:
                raise ValueError(
                    f"Unsupported indicator type: {itype!r}. "
                    f"Allowed: {sorted(self.ALLOWED_TYPES)}"
                )

        per_ticker_frames: list[pd.DataFrame] = []
        for ticker in tickers:
            logger.info("Extracting indicators for %s", ticker)
            merged: Optional[pd.DataFrame] = None
            for itype, period in indicators:
                try:
                    df = self._fetch_one(ticker, itype, period)
                except Exception as exc:  # noqa: BLE001 - per-call resilience
                    logger.error("  failed %s/%s(%d): %s", ticker, itype, period, exc)
                    time.sleep(self.rate_limit_seconds)
                    continue

                if df.empty:
                    logger.warning("  empty %s/%s(%d)", ticker, itype, period)
                    time.sleep(self.rate_limit_seconds)
                    continue

                merged = (
                    df if merged is None
                    else merged.merge(df, on=["date", "ticker"], how="outer")
                )
                time.sleep(self.rate_limit_seconds)

            if merged is not None and not merged.empty:
                logger.info("  rows=%d cols=%s", len(merged), list(merged.columns))
                per_ticker_frames.append(merged)

        if not per_ticker_frames:
            logger.error("No indicators fetched for any ticker.")
            return pd.DataFrame()

        out = (
            pd.concat(per_ticker_frames, ignore_index=True)
            .sort_values(["ticker", "date"])
            .reset_index(drop=True)
        )

        if save_csv:
            out.to_csv(save_csv, index=False)
            logger.info("Saved indicators to %s (rows=%d)", save_csv, len(out))

        return out


# --------------------------------------------------------------------------- #
# Helpers for derived indicators not served by the FMP endpoint
# --------------------------------------------------------------------------- #
def add_macd(
    df: pd.DataFrame,
    fast_col: str = "ema_12",
    slow_col: str = "ema_26",
    signal_period: int = 9,
    ticker_col: str = "ticker",
    date_col: str = "date",
) -> pd.DataFrame:
    """Append MACD columns to a long-form (date, ticker, ema_12, ema_26) frame.

    Produces:
      macd         = ema_12 - ema_26
      macd_signal  = 9-period EMA of macd (within each ticker)
      macd_hist    = macd - macd_signal

    MACD is computed locally because FMP's /technical_indicator endpoint does
    not ship a MACD type.
    """
    if fast_col not in df.columns or slow_col not in df.columns:
        raise KeyError(
            f"add_macd requires columns {fast_col!r} and {slow_col!r}; "
            f"got {list(df.columns)}"
        )

    out = df.sort_values([ticker_col, date_col]).copy()
    out["macd"] = out[fast_col] - out[slow_col]
    out["macd_signal"] = (
        out.groupby(ticker_col)["macd"]
        .transform(lambda s: s.ewm(span=signal_period, adjust=False).mean())
    )
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    return out
