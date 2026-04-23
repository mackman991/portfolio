"""Pull S&P 100 OHLCV + EPS surprises + FMP technical indicators.

Writes three CSVs into ``config.DATA_DIR`` (default: ./data):

  - price_data_sp100.csv       — daily OHLCV via yfinance (free, unlimited)
  - eps_sp100.csv              — FMP /api/v3/earnings-surprises
  - indicators_sp100.csv       — FMP /api/v3/technical_indicator/daily
                                 (RSI14, SMA20/50/200, EMA12/26) + locally
                                 derived MACD

Run:

    # from the repo root, with the venv activated and .env populated:
    python -m src.pull_sp100

    # smoke-test on the first 5 tickers:
    python -m src.pull_sp100 --limit 5

    # skip steps that already have outputs:
    python -m src.pull_sp100 --skip prices --skip eps

FMP rate-limit notes
--------------------
Per full run (~104 tickers), this script issues roughly:
  - 104 EPS requests
  - 624 technical-indicator requests (6 indicators × 104 tickers)

At the default 0.25s spacing (240/min) that's ~3 minutes. Starter tier
(300 req/min) has ~20% headroom. --rate-limit controls the spacing.

Robust writes
-------------
Each step writes to ``<name>.csv.tmp`` and then ``os.replace``s into the final
path. If Windows blocks the final rename because the file is open elsewhere
(Excel, a notebook preview, etc.), the script falls back to a timestamped
filename like ``indicators_sp100.20260417T091200.csv`` and logs a WARNING.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time as _time
from typing import List

import pandas as pd

from src import config
from src.data import (
    EpsExtractor,
    StockPriceExtractor,
    TechnicalIndicatorExtractor,
    add_macd,
)

logger = logging.getLogger(__name__)


def _save_csv_safely(df: pd.DataFrame, path: str) -> str:
    """Write df to path atomically. If the final rename is blocked (file open
    in Excel/etc.), fall back to a timestamped sibling filename so the pull is
    never lost. Returns the path actually written."""
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    try:
        os.replace(tmp, path)
        return path
    except PermissionError as exc:
        ts = _time.strftime("%Y%m%dT%H%M%S")
        base, ext = os.path.splitext(path)
        fallback = f"{base}.{ts}{ext}"
        os.replace(tmp, fallback)
        logger.warning(
            "Could not overwrite %s (%s). Wrote to %s instead. "
            "Close any program holding the original file (Excel, a notebook, "
            "an IDE preview pane) and rename manually if you want.",
            path, exc, fallback,
        )
        return fallback


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pull S&P 100 price/EPS/technical data.")
    p.add_argument(
        "--limit", type=int, default=None,
        help="If set, only pull the first N tickers (useful for smoke-testing).",
    )
    p.add_argument(
        "--skip", action="append",
        choices=["prices", "eps", "indicators"], default=[],
        help="Skip one or more steps (may be passed multiple times).",
    )
    p.add_argument(
        "--rate-limit", type=float, default=0.25,
        help="Seconds to sleep between FMP requests. Raise if you hit 429s.",
    )
    p.add_argument(
        "--out-prefix", default="sp100",
        help="Filename suffix for outputs, e.g. 'sp100' -> price_data_sp100.csv.",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    tickers = list(config.SP100_TICKERS)
    if args.limit:
        tickers = tickers[: args.limit]

    data_dir = config.DATA_DIR
    os.makedirs(data_dir, exist_ok=True)

    prefix = args.out_prefix
    price_csv = os.path.join(data_dir, f"price_data_{prefix}.csv")
    eps_csv = os.path.join(data_dir, f"eps_{prefix}.csv")
    ind_csv = os.path.join(data_dir, f"indicators_{prefix}.csv")

    logger.info("Universe: %d tickers (skip=%s)", len(tickers), args.skip or "none")

    # ---- 1. Prices (yfinance) ------------------------------------------------
    if "prices" not in args.skip:
        logger.info("=== Prices (yfinance) -> %s ===", price_csv)
        df = StockPriceExtractor(config.DATES.start, config.DATES.end).extract(tickers)
        if not df.empty:
            out = _save_csv_safely(df, price_csv)
            logger.info("  wrote %d rows to %s", len(df), out)
    else:
        logger.info("Skipping prices")

    # ---- 2. EPS surprises (FMP) ---------------------------------------------
    if "eps" not in args.skip:
        logger.info("=== EPS surprises (FMP) -> %s ===", eps_csv)
        df = EpsExtractor(
            config.fmp_api_key(),
            rate_limit_seconds=args.rate_limit,
        ).extract(tickers)
        if not df.empty:
            out = _save_csv_safely(df, eps_csv)
            logger.info("  wrote %d rows to %s", len(df), out)
    else:
        logger.info("Skipping EPS")

    # ---- 3. Technical indicators (FMP) + local MACD --------------------------
    if "indicators" not in args.skip:
        logger.info("=== Technical indicators (FMP) -> %s ===", ind_csv)
        ti = TechnicalIndicatorExtractor(
            config.fmp_api_key(),
            rate_limit_seconds=args.rate_limit,
        )
        indicators_df = ti.extract(
            tickers, indicators=config.DEFAULT_INDICATORS, save_csv=None,
        )
        if indicators_df.empty:
            logger.error("Indicator pull returned no rows; nothing to write.")
        else:
            if {"ema_12", "ema_26"}.issubset(indicators_df.columns):
                logger.info("Computing MACD locally from ema_12/ema_26 ...")
                indicators_df = add_macd(indicators_df)
            else:
                logger.warning(
                    "Skipping MACD: need both ema_12 and ema_26; got %s",
                    list(indicators_df.columns),
                )
            out = _save_csv_safely(indicators_df, ind_csv)
            logger.info(
                "  wrote %d rows to %s (tickers=%d, cols=%s)",
                len(indicators_df), out, indicators_df.ticker.nunique(),
                list(indicators_df.columns),
            )
    else:
        logger.info("Skipping indicators")

    logger.info("Done. Outputs in %s/", data_dir)


if __name__ == "__main__":  # pragma: no cover
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
