"""Earnings-surprise extractor (Financial Modeling Prep)."""
from __future__ import annotations

import logging
import time
from typing import Iterable, Optional, Union

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class EpsExtractor:
    """Fetch historical EPS estimates and actuals via the FMP earnings-surprises endpoint."""

    ENDPOINT = "https://financialmodelingprep.com/api/v3/earnings-surprises/{ticker}"

    def __init__(self, fmp_api_key: str, rate_limit_seconds: float = 0.25):
        if not fmp_api_key:
            raise ValueError("FMP API key is required.")
        self.fmp_api_key = fmp_api_key
        self.rate_limit_seconds = rate_limit_seconds

    def extract(
        self,
        tickers: Union[str, Iterable[str]],
        save_csv: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return a long-form DataFrame of earnings surprises across the supplied tickers."""
        if isinstance(tickers, str):
            tickers = [tickers]

        rows: list[dict] = []
        for ticker in tickers:
            logger.info("Extracting EPS for %s", ticker)
            url = f"{self.ENDPOINT.format(ticker=ticker)}?apikey={self.fmp_api_key}&limit=10000"
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                data = response.json()
                if isinstance(data, dict):
                    raise ValueError(f"Unexpected response for {ticker}: {data}")
                for rec in data:
                    rec["ticker"] = ticker
                rows.extend(data)
                logger.info("  records=%d for %s", len(data), ticker)
            except Exception as exc:  # noqa: BLE001 - want resilience per ticker
                logger.error("Failed fetching EPS for %s: %s", ticker, exc)
            time.sleep(self.rate_limit_seconds)

        df = pd.DataFrame(rows)
        if not df.empty:
            df.columns = [c.strip().lower() for c in df.columns]
            for col in ("date", "fiscaldateending", "reporteddate"):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")

        if save_csv:
            df.to_csv(save_csv, index=False)
            logger.info("Saved EPS data to %s", save_csv)
        return df
