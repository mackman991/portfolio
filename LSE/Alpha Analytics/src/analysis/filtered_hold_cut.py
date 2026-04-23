"""Filtered Hold/Cut backtests - tests 3 regime filters against the unfiltered
baseline and a passive Buy-and-Hold benchmark on the same event dates.

See module docstring in the repository version. Emitted outputs:
  filtered_hold_cut_summary.csv
  equity_curve_hold_cut_unfiltered.csv
  equity_curve_hold_cut_macd_bullish.csv
  equity_curve_hold_cut_above_sma50.csv
  equity_curve_hold_cut_not_rsi_oversold.csv
  equity_curve_buyhold_benchmark.csv
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import Callable, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_HOLD_HORIZON = 11  # trading days: entry + up to 10 hold


def _equity_from_returns(per_event: pd.DataFrame) -> pd.DataFrame:
    if per_event.empty:
        return pd.DataFrame(columns=["eps_date", "ret", "equity"])
    daily = (
        per_event.dropna(subset=["eps_date", "ret"])
        .groupby("eps_date", as_index=False)["ret"]
        .mean()
        .sort_values("eps_date")
        .reset_index(drop=True)
    )
    daily["equity"] = (1.0 + daily["ret"]).cumprod()
    return daily


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return np.nan
    rm = equity.cummax()
    return float((equity / rm - 1.0).min())


def _span_years(eps_dates: pd.Series) -> float:
    if eps_dates.empty:
        return np.nan
    d = pd.to_datetime(eps_dates)
    return (d.max() - d.min()).days / 365.25


def _summarise(per_event: pd.DataFrame, label: str, total_universe_n: int) -> Dict:
    r = per_event["ret"].dropna()
    n = int(r.shape[0])
    if n == 0:
        return {"variant": label, "n": 0, "coverage": 0.0, "mean": np.nan,
                "median": np.nan, "hit_rate": np.nan, "std": np.nan,
                "t_stat": np.nan, "info_ratio": np.nan,
                "total_return": np.nan, "cagr": np.nan,
                "max_drawdown": np.nan, "car_mdd": np.nan}

    mean = float(r.mean()); median = float(r.median())
    hit = float((r > 0).mean())
    std = float(r.std(ddof=1)) if n > 1 else np.nan
    se = std / np.sqrt(n) if (n > 1 and std > 0) else np.nan
    t_stat = mean / se if (se and se > 0) else np.nan
    info_ratio = mean / std if (std and std > 0) else np.nan

    eq = _equity_from_returns(per_event)
    total_ret = float(eq["equity"].iloc[-1] - 1.0) if not eq.empty else np.nan
    years = _span_years(eq["eps_date"])
    cagr = (1.0 + total_ret) ** (1.0 / years) - 1.0 if (years and years > 0 and total_ret > -1) else np.nan
    mdd = _max_drawdown(eq["equity"])
    car_mdd = cagr / abs(mdd) if (mdd and mdd < 0 and cagr == cagr) else np.nan

    return {"variant": label, "n": n,
            "coverage": n / total_universe_n if total_universe_n else np.nan,
            "mean": mean, "median": median, "hit_rate": hit, "std": std,
            "t_stat": t_stat, "info_ratio": info_ratio,
            "total_return": total_ret, "cagr": cagr,
            "max_drawdown": mdd, "car_mdd": car_mdd}


def _buyhold_returns(per_event: pd.DataFrame, prices: pd.DataFrame,
                     horizon_days: int = DEFAULT_HOLD_HORIZON) -> pd.DataFrame:
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    prices = prices.dropna(subset=["date"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    by_tic = {t: g.reset_index(drop=True) for t, g in prices.groupby("ticker")}

    out_rows = []
    for row in per_event[["ticker", "eps_date"]].itertuples(index=False):
        tic = row.ticker
        ed = pd.to_datetime(row.eps_date).normalize()
        g = by_tic.get(tic)
        if g is None:
            out_rows.append({"ticker": tic, "eps_date": row.eps_date, "ret": np.nan}); continue
        idx_entry = g.index[g["date"] >= ed]
        if len(idx_entry) == 0:
            out_rows.append({"ticker": tic, "eps_date": row.eps_date, "ret": np.nan}); continue
        i0 = int(idx_entry[0])
        i1 = min(i0 + horizon_days, len(g) - 1)
        p0 = g.loc[i0, "close"]; p1 = g.loc[i1, "close"]
        if p0 and not np.isnan(p0) and p1 and not np.isnan(p1):
            out_rows.append({"ticker": tic, "eps_date": row.eps_date, "ret": float(p1 / p0 - 1.0)})
        else:
            out_rows.append({"ticker": tic, "eps_date": row.eps_date, "ret": np.nan})
    return pd.DataFrame(out_rows)


FilterFn = Callable[[pd.DataFrame], pd.Series]

FILTERS: Dict[str, FilterFn] = {
    "macd_bullish":     lambda df: df["macd_bullish"] == True,
    "above_sma50":      lambda df: df["above_sma_50"] == True,
    "not_rsi_oversold": lambda df: df["rsi_bucket"] != "oversold_<30",
}


def run(returns_csv: str = "data/strategy_returns_regime_hold_cut_sp100.csv",
        prices_csv:  str = "data/price_data_sp100.csv",
        out_dir:     str = "data") -> pd.DataFrame:
    logger.info("Loading %s", returns_csv)
    df = pd.read_csv(returns_csv)
    logger.info("  %d events (%d tickers)", len(df), df["ticker"].nunique())
    total_n = len(df)
    summaries = []

    s = _summarise(df, "unfiltered_hold_cut", total_n); summaries.append(s)
    _equity_from_returns(df).to_csv(os.path.join(out_dir, "equity_curve_hold_cut_unfiltered.csv"), index=False)
    logger.info("  unfiltered: n=%d mean=%+.2f%% IR=%.2f mdd=%+.2f%% cagr=%+.2f%%",
                s["n"], s["mean"]*100, s["info_ratio"], s["max_drawdown"]*100, (s["cagr"] or 0)*100)

    for name, fn in FILTERS.items():
        sub = df[fn(df)].copy()
        s = _summarise(sub, f"hold_cut_{name}", total_n); summaries.append(s)
        _equity_from_returns(sub).to_csv(os.path.join(out_dir, f"equity_curve_hold_cut_{name}.csv"), index=False)
        logger.info("  %s: n=%d cov=%.1f%% mean=%+.2f%% IR=%.2f mdd=%+.2f%% cagr=%+.2f%%",
                    name, s["n"], s["coverage"]*100, s["mean"]*100, s["info_ratio"],
                    s["max_drawdown"]*100, (s["cagr"] or 0)*100)

    logger.info("Computing Buy-and-Hold benchmark from %s", prices_csv)
    prices = pd.read_csv(prices_csv)
    bh = _buyhold_returns(df[["ticker", "eps_date"]], prices)
    s = _summarise(bh, "buyhold_benchmark", total_n); summaries.append(s)
    _equity_from_returns(bh).to_csv(os.path.join(out_dir, "equity_curve_buyhold_benchmark.csv"), index=False)
    logger.info("  buyhold: n=%d mean=%+.2f%% IR=%.2f mdd=%+.2f%% cagr=%+.2f%%",
                s["n"], s["mean"]*100, s["info_ratio"], s["max_drawdown"]*100, (s["cagr"] or 0)*100)

    summary = pd.DataFrame(summaries)
    summary_path = os.path.join(out_dir, "filtered_hold_cut_summary.csv")
    summary.to_csv(summary_path, index=False)
    logger.info("Wrote %s", summary_path)
    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--returns", default="data/strategy_returns_regime_hold_cut_sp100.csv")
    p.add_argument("--prices",  default="data/price_data_sp100.csv")
    p.add_argument("--out-dir", default="data")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(returns_csv=args.returns, prices_csv=args.prices, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
