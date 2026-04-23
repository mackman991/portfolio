"""Technical-regime × earnings interaction analysis.

Given an event-window panel and the FMP-computed indicator panel, this module:

  1. Snapshots the indicator + close values for each event at a nominated
     `rel_day` (look-ahead-safe: use -11 for a pre-earnings -10 entry; use 0
     for a post-earnings +1 entry — indicator values at day 0's close are
     observable before the next day's open).
  2. Derives a set of categorical/binary regime features (RSI bucket,
     price-vs-SMA flags, MACD sign, etc.).
  3. Computes per-event returns for the main strategies directly from the
     event window (so we can slice them by regime).
  4. Runs a (strategy × regime feature) grid and returns per-bucket summary
     statistics alongside the unconditional baseline.

Everything here operates at the **per-event** level — one row per
(ticker, eps_date).
"""
from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# 1. Merge indicators onto events at a safe rel_day
# --------------------------------------------------------------------------- #
_IND_COLS = [
    "rsi_14", "sma_20", "sma_50", "sma_200",
    "ema_12", "ema_26",
    "macd", "macd_signal", "macd_hist",
]


def _normalise_date(s: pd.Series) -> pd.Series:
    """Strip tz/time-of-day so event dates align with daily-indicator dates."""
    s = pd.to_datetime(s, errors="coerce")
    try:
        if getattr(s.dt, "tz", None) is not None:
            s = s.dt.tz_localize(None)
    except (AttributeError, TypeError):
        pass
    return s.dt.normalize()


def snapshot_at_rel_day(
    event_windows: pd.DataFrame,
    indicators: pd.DataFrame,
    at_rel_day: int,
) -> pd.DataFrame:
    """Return one row per event with indicator values at the given rel_day.

    Columns returned: ticker, eps_date, snapshot_rel_day, snapshot_date,
    snapshot_close, rsi_14, sma_20, sma_50, sma_200, ema_12, ema_26,
    macd, macd_signal, macd_hist.

    Events with no indicator match at the snapshot date are kept with NaNs
    in the indicator columns so the caller can see coverage loss.
    """
    required_ev = {"ticker", "eps_date", "rel_day", "price_date", "close"}
    missing_ev = required_ev - set(event_windows.columns)
    if missing_ev:
        raise KeyError(f"event_windows missing required columns: {missing_ev}")

    required_ind = {"ticker", "date"} | {c for c in _IND_COLS if c}
    missing_ind = required_ind - set(indicators.columns)
    if missing_ind:
        raise KeyError(f"indicators missing required columns: {missing_ind}")

    snap = (
        event_windows[event_windows["rel_day"] == at_rel_day]
        [["ticker", "eps_date", "price_date", "close"]]
        .copy()
        .rename(columns={"price_date": "snapshot_date", "close": "snapshot_close"})
    )
    snap["snapshot_rel_day"] = at_rel_day
    snap["date_key"] = _normalise_date(snap["snapshot_date"])

    ind = indicators.copy()
    ind["date_key"] = _normalise_date(ind["date"])

    merged = snap.merge(
        ind[["ticker", "date_key"] + _IND_COLS],
        on=["ticker", "date_key"],
        how="left",
    ).drop(columns=["date_key"])
    return merged


# --------------------------------------------------------------------------- #
# 2. Derive regime features
# --------------------------------------------------------------------------- #
def add_regime_features(snap: pd.DataFrame) -> pd.DataFrame:
    """Append binary/categorical regime flags derived from raw indicator values."""
    out = snap.copy()

    # RSI regimes
    out["rsi_bucket"] = pd.cut(
        out["rsi_14"],
        bins=[-np.inf, 30, 50, 70, np.inf],
        labels=["oversold_<30", "weak_30-50", "strong_50-70", "overbought_>70"],
    )

    # Price vs moving averages
    c = out["snapshot_close"]
    for ma in ("sma_20", "sma_50", "sma_200"):
        out[f"above_{ma}"] = (c > out[ma]).where(out[ma].notna())

    # Long-term trend: 50-day SMA above 200-day SMA (golden-cross regime)
    out["trend_up_50v200"] = (out["sma_50"] > out["sma_200"]).where(
        out["sma_50"].notna() & out["sma_200"].notna()
    )

    # MACD regimes
    out["macd_bullish"] = (out["macd"] > out["macd_signal"]).where(out["macd"].notna())
    out["macd_above_zero"] = (out["macd"] > 0).where(out["macd"].notna())
    out["macd_hist_positive"] = (out["macd_hist"] > 0).where(out["macd_hist"].notna())

    # Continuous: % distance from SMA-50 (useful for finer splits later)
    out["pct_vs_sma_50"] = (c / out["sma_50"]) - 1.0

    return out


REGIME_COLS = [
    "rsi_bucket",
    "above_sma_20", "above_sma_50", "above_sma_200",
    "trend_up_50v200",
    "macd_bullish", "macd_above_zero", "macd_hist_positive",
]


# --------------------------------------------------------------------------- #
# 3. Per-event strategy returns from an event-window panel
# --------------------------------------------------------------------------- #
def _compound(group: pd.DataFrame, lo: int, hi: int) -> float:
    r = pd.to_numeric(
        group.loc[(group["rel_day"] >= lo) & (group["rel_day"] <= hi), "ret"],
        errors="coerce",
    ).dropna().values
    if r.size == 0:
        return np.nan
    return float(np.prod(1 + r) - 1)


def per_event_returns(
    event_windows: pd.DataFrame,
    *,
    strategy: str,
    H: int = 6,
    P: int = 10,
    hold_cut_threshold: float = 0.05,
    hold_cut_extension: int = 10,
    amc_set: Optional[set] = None,
) -> pd.DataFrame:
    """Compute one return per event for a named strategy.

    strategy options:
      - "momentum_1_H"       : long beats / short misses, rel_day +1..+H (default H=6)
      - "pre_runup_P_1"      : long, rel_day -P..-1 (default P=10)
      - "hold_cut"           : long -P..exit, where exit shifts +1 for AMC events

    amc_set: optional set of (ticker, eps_date_str) pairs for after-market-close
             events. For hold_cut, the cut/hold exit is shifted by +1 trading day
             for these events so that the exit price reflects post-announcement
             information rather than the pre-announcement close.
    """
    gb = event_windows.groupby(["ticker", "eps_date"], sort=False)
    rows: list[dict] = []

    if strategy == "momentum_1_H":
        for (tic, ed), g in gb:
            gross = _compound(g, 1, H)
            if pd.isna(gross):
                continue
            sign = 1.0 if bool(g["beat"].iloc[0]) else -1.0
            rows.append({"ticker": tic, "eps_date": ed, "ret": sign * gross})

    elif strategy == "pre_runup_P_1":
        for (tic, ed), g in gb:
            gross = _compound(g, -P, -1)
            if pd.isna(gross):
                continue
            rows.append({"ticker": tic, "eps_date": ed, "ret": gross})

    elif strategy == "hold_cut":
        for (tic, ed), g in gb:
            surprise = pd.to_numeric(g["surprise"], errors="coerce").dropna()
            if surprise.empty:
                continue
            abs_surp = abs(float(surprise.iloc[0]))
            base_exit = hold_cut_extension if abs_surp > hold_cut_threshold else 0
            # AMC correction: announcement after close means day-0 price precedes
            # the print; shift exit by +1 so the exit captures post-announcement prices.
            amc_shift = 0
            if amc_set is not None:
                ed_key = str(pd.Timestamp(ed).date())
                if (tic, ed_key) in amc_set:
                    amc_shift = 1
            exit_day = base_exit + amc_shift
            gross = _compound(g, -P, exit_day)
            if pd.isna(gross):
                continue
            rows.append({
                "ticker": tic, "eps_date": ed, "ret": gross,
                "decision": "hold" if base_exit == hold_cut_extension else "cut",
                "amc": bool(amc_shift),
            })

    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 4. Summary + grid runner
# --------------------------------------------------------------------------- #
def summarise(returns: pd.Series) -> dict:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    n = int(r.size)
    mean = float(r.mean()) if n else np.nan
    med = float(r.median()) if n else np.nan
    hit = float((r > 0).mean()) if n else np.nan
    std = float(r.std(ddof=1)) if n > 1 else np.nan
    se = (std / np.sqrt(n)) if (n > 1 and std and std > 0) else np.nan
    t = (mean / se) if (se and se > 0) else np.nan
    return {"n": n, "mean": mean, "median": med, "hit_rate": hit, "std": std, "t_stat": t}


def grid_splits(
    per_event: pd.DataFrame,
    regime_cols: Iterable[str] = REGIME_COLS,
    min_n: int = 50,
    baseline_label: str = "__baseline__",
) -> pd.DataFrame:
    """Produce a long-form table of (regime, bucket) -> summary stats.

    The first row is the unconditional baseline so downstream plots can use
    it as a reference. Rows with n < min_n get small_n=True flagged.
    """
    rows = [{"regime": baseline_label, "bucket": "all",
             **summarise(per_event["ret"]), "small_n": False}]

    for col in regime_cols:
        if col not in per_event.columns:
            continue
        gb = per_event.groupby(col, dropna=False, observed=False)
        for bucket, g in gb:
            stats = summarise(g["ret"])
            rows.append({
                "regime": col,
                "bucket": "NaN" if pd.isna(bucket) else str(bucket),
                **stats,
                "small_n": stats["n"] < min_n,
            })
    return pd.DataFrame(rows)


def uplift_table(grid: pd.DataFrame, baseline_label: str = "__baseline__") -> pd.DataFrame:
    """Sort buckets by mean-return uplift vs the unconditional baseline."""
    base_mean = float(grid.loc[grid["regime"] == baseline_label, "mean"].iloc[0])
    base_hit  = float(grid.loc[grid["regime"] == baseline_label, "hit_rate"].iloc[0])
    out = grid.loc[grid["regime"] != baseline_label].copy()
    out["mean_uplift"]     = out["mean"] - base_mean
    out["hit_rate_uplift"] = out["hit_rate"] - base_hit
    return out.sort_values(["mean_uplift", "t_stat"], ascending=[False, False]).reset_index(drop=True)
