"""Daily OHLCV price extractor (yfinance)."""
from __future__ import annotations

import logging
from typing import Iterable, Optional, Union

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class StockPriceExtractor:
    """Pull daily OHLCV history from Yahoo Finance for one or more tickers."""

    def __init__(self, start_date: str, end_date: str):
        self.start_date = start_date
        self.end_date = end_date

    def extract(
        self,
        tickers: Union[str, Iterable[str]],
        save_csv: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return a tidy long-form DataFrame with columns: date, open/high/low/close/volume, ticker."""
        if isinstance(tickers, str):
            tickers = [tickers]

        frames = []
        for ticker in tickers:
            logger.info("Extracting prices for %s", ticker)
            hist = yf.Ticker(ticker).history(
                start=self.start_date,
                end=self.end_date,
                interval="1d",
            )
            if hist.empty:
                logger.warning("No price data returned for %s", ticker)
                continue
            hist = hist.rename(columns=str.lower).reset_index()
            if "Date" in hist.columns:
                hist = hist.rename(columns={"Date": "date"})
            hist["ticker"] = ticker
            frames.append(hist)
            logger.info("  rows=%d for %s", len(hist), ticker)

        if not frames:
            logger.error("No historical data fetched for any ticker.")
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        if save_csv:
            df.to_csv(save_csv, index=False)
            logger.info("Saved price data to %s", save_csv)
        return df
