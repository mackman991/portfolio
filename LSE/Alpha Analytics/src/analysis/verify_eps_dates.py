"""Verify the ``eps_date`` column used by the Hold/Cut backtest.

Three cross-checks, each independently useful:

    1. **Calendar coverage**
       Left-join ``data/eps_sp100.csv`` (FMP earnings-surprises endpoint) with
       ``data/earnings_calendar_sp100.csv`` (FMP historical-calendar endpoint
       with bmo/amc timing). Report how many events match, how many are only
       in one source, and the bmo/amc/dmh time distribution.

    2. **SEC cross-check**
       Pull ``earnings_release`` (8-K Item 2.02 acceptance timestamp) from
       ``data/sec_financials.csv`` and compare against the FMP calendar date.
       Match within ±3 calendar days on (ticker), report |Δ| distribution and
       flag any event where |Δ| > 1 day.

    3. **AMC entry-shift impact**
       The current Hold/Cut backtest exits at day 0 close (cut) or day +10
       close (hold). For events released *after* market close (AMC), the
       day-0 close prints **before** the surprise is revealed — so a "cut at
       day 0" decision embeds look-ahead. Re-run the strategy with exit
       shifted by one trading day for AMC events (cut → +1, hold → +11) and
       report the net impact on mean return, hit rate, and t-stat, both
       unfiltered and under the MACD-bullish filter.

Outputs (under ``data/``):
    verify_eps_coverage_summary.csv     one-row per source/status
    verify_eps_sec_crosscheck.csv       per-event Δdays table
    verify_eps_amc_shift_summary.csv    before/after headline stats
    verify_eps_amc_shift_events.csv     per-event, AMC-tagged, both returns

Run locally once ``earnings_calendar_sp100.csv`` has been extracted:

    python -m src.data.earnings_calendar_extractor
    python -m src.analysis.verify_eps_dates
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Hold/Cut parameters, as used in run_technical_regime.py.
HOLD_CUT_ENTRY_DAY = -10        # rel_day of entry
HOLD_CUT_CUT_EXIT = 0           # rel_day of exit on "cut"
HOLD_CUT_HOLD_EXIT = 10         # rel_day of exit on "hold"

# AMC correction: shift the exit by one trading day for events released
# after the close. Entry remains at -10 (well before the release) either way.
AMC_SHIFT = 1


# --------------------------------------------------------------------------- #
# 1. Calendar coverage
# --------------------------------------------------------------------------- #
def verify_calendar_coverage(
    eps_df: pd.DataFrame,
    cal_df: pd.DataFrame,
) -> pd.DataFrame:
    """Outer-join surprises-based EPS dates against the FMP calendar.

    Returns a long-form summary table keyed on `source`/`bucket`.
    """
    e = eps_df[["ticker", "date"]].dropna().copy()
    e["date"] = pd.to_datetime(e["date"], errors="coerce").dt.normalize()
    e = e.drop_duplicates(["ticker", "date"]).assign(_in_eps=True)

    c = cal_df[["ticker", "date", "time"]].dropna(subset=["ticker", "date"]).copy()
    c["date"] = pd.to_datetime(c["date"], errors="coerce").dt.normalize()
    c = c.drop_duplicates(["ticker", "date"]).assign(_in_cal=True)

    merged = e.merge(c, on=["ticker", "date"], how="outer")

    n_both = int((merged["_in_eps"] & merged["_in_cal"]).fillna(False).sum())
    n_only_eps = int((merged["_in_eps"].fillna(False) & ~merged["_in_cal"].fillna(False)).sum())
    n_only_cal = int((~merged["_in_eps"].fillna(False) & merged["_in_cal"].fillna(False)).sum())

    # Time distribution over matched + cal-only events.
    with_time = merged[merged["_in_cal"].fillna(False)]
    time_dist = with_time["time"].fillna("unknown").value_counts().to_dict()

    rows: List[Dict] = [
        {"source": "both",         "bucket": "n_matched",    "count": n_both},
        {"source": "eps_only",     "bucket": "n_only_in_surprises", "count": n_only_eps},
        {"source": "calendar_only","bucket": "n_only_in_calendar",  "count": n_only_cal},
    ]
    for k, v in time_dist.items():
        rows.append({"source": "calendar_time", "bucket": f"time_{k}", "count": int(v)})

    summary = pd.DataFrame(rows)
    logger.info("Calendar coverage: matched=%d eps_only=%d cal_only=%d",
                n_both, n_only_eps, n_only_cal)
    logger.info("Calendar time distribution: %s", time_dist)
    return summary


# --------------------------------------------------------------------------- #
# 2. SEC cross-check
# --------------------------------------------------------------------------- #
def verify_sec_crosscheck(
    cal_df: pd.DataFrame,
    sec_df: pd.DataFrame,
    max_match_days: int = 3,
) -> pd.DataFrame:
    """For each 8-K Item 2.02 acceptance timestamp, find the nearest FMP
    calendar date within ``max_match_days`` and report the signed delta.
    """
    if "earnings_release" not in sec_df.columns:
        logger.warning("sec_financials missing earnings_release; skipping SEC cross-check.")
        return pd.DataFrame()

    sec = sec_df[["ticker", "earnings_release"]].copy()
    sec["earnings_release"] = pd.to_datetime(sec["earnings_release"], errors="coerce", utc=True)
    sec = sec.dropna(subset=["ticker", "earnings_release"])
    if sec.empty:
        logger.warning("sec_financials has no non-null earnings_release rows.")
        return pd.DataFrame()

    # Strip timezone for date arithmetic.
    sec["sec_date"] = sec["earnings_release"].dt.tz_convert("US/Eastern").dt.normalize().dt.tz_localize(None)
    sec = sec.drop_duplicates(["ticker", "sec_date"])

    cal = cal_df[["ticker", "date", "time"]].copy()
    cal["date"] = pd.to_datetime(cal["date"], errors="coerce").dt.normalize()
    cal = cal.dropna(subset=["ticker", "date"])

    out_rows: List[Dict] = []
    cal_by_ticker = {t: g.sort_values("date") for t, g in cal.groupby("ticker")}

    for _, row in sec.iterrows():
        tic = row["ticker"]
        sd = row["sec_date"]
        g = cal_by_ticker.get(tic)
        if g is None or g.empty:
            out_rows.append({
                "ticker": tic, "sec_date": sd, "calendar_date": pd.NaT,
                "calendar_time": None, "delta_days": np.nan, "match": False,
            })
            continue

        deltas = (g["date"] - sd).dt.days
        idx = deltas.abs().idxmin()
        best = g.loc[idx]
        delta = int((best["date"] - sd).days)
        out_rows.append({
            "ticker": tic,
            "sec_date": sd,
            "calendar_date": best["date"],
            "calendar_time": best.get("time"),
            "delta_days": delta,
            "match": abs(delta) <= max_match_days,
        })

    df = pd.DataFrame(out_rows)
    if df.empty:
        return df

    matched = df[df["match"]]
    n_exact = int((matched["delta_days"] == 0).sum())
    n_within_1 = int((matched["delta_days"].abs() <= 1).sum())
    n_off = int((matched["delta_days"].abs() > 1).sum())
    logger.info(
        "SEC cross-check: %d events matched within %dd — exact=%d, |Δ|≤1=%d, |Δ|>1=%d",
        len(matched), max_match_days, n_exact, n_within_1, n_off,
    )
    return df


# --------------------------------------------------------------------------- #
# 3. AMC entry-shift impact on Hold/Cut returns
# --------------------------------------------------------------------------- #
def _prepare_prices(price_df: pd.DataFrame) -> pd.DataFrame:
    """Normalise the price panel so date is a midnight-UTC-free pd.Timestamp."""
    p = price_df[["ticker", "date", "close"]].copy()
    p["date"] = (
        pd.to_datetime(p["date"], utc=True, errors="coerce")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    p["close"] = pd.to_numeric(p["close"], errors="coerce")
    p = p.dropna(subset=["ticker", "date", "close"]).drop_duplicates(["ticker", "date"])
    return p.sort_values(["ticker", "date"]).reset_index(drop=True)


def _close_at_offset(
    price_by_ticker: Dict[str, pd.DataFrame],
    ticker: str,
    eps_date: pd.Timestamp,
    trading_day_offset: int,
) -> Tuple[Optional[float], Optional[pd.Timestamp]]:
    """Return (close, actual_date) at ``eps_date + trading_day_offset`` using
    the ticker's own trading-day calendar. Returns (None, None) if out of
    range or the eps_date isn't in the calendar (non-trading day).
    """
    g = price_by_ticker.get(ticker)
    if g is None or g.empty:
        return None, None

    dates = g["date"].values
    # Find insertion point; eps_date may fall on a non-trading day if a holiday
    # was reported as the release date. Use the trading day on or after eps_date
    # as the anchor, then apply offset from there.
    idx = np.searchsorted(dates, np.datetime64(eps_date))
    if idx >= len(dates):
        return None, None

    anchor_idx = idx
    target_idx = anchor_idx + trading_day_offset
    if target_idx < 0 or target_idx >= len(g):
        return None, None

    row = g.iloc[target_idx]
    return float(row["close"]), row["date"]


def verify_amc_shift(
    strategy_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    price_df: pd.DataFrame,
    filter_col: Optional[str] = None,
    filter_val=True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute Hold/Cut per-event returns with the AMC exit-shift correction.

    For every event in ``strategy_df`` we:
      * join in the FMP calendar's `time` field (bmo/amc/dmh/None)
      * compute entry_close = close at rel_day -10
      * compute current exit_close using the existing `decision` (cut → 0, hold → +10)
      * compute shifted exit_close for AMC events (cut → +1, hold → +11)
      * report per-event recomputed_ret, delta_vs_current, and the
        mean/t-stat shift for the cohort as a whole.

    Parameters
    ----------
    filter_col : str or None
        Column in ``strategy_df`` to use as a regime filter (e.g. ``macd_bullish``).
        If None, runs on all events (unfiltered).
    filter_val : any
        Value of ``filter_col`` to keep. Default True.
    """
    df = strategy_df.copy()
    df["eps_date"] = pd.to_datetime(df["eps_date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["ticker", "eps_date", "ret", "decision"])

    if filter_col is not None and filter_col in df.columns:
        df = df[df[filter_col] == filter_val].copy()

    cal = cal_df[["ticker", "date", "time"]].copy()
    cal["date"] = pd.to_datetime(cal["date"], errors="coerce").dt.normalize()
    cal = cal.dropna(subset=["ticker", "date"]).drop_duplicates(["ticker", "date"])

    df = df.merge(
        cal, left_on=["ticker", "eps_date"], right_on=["ticker", "date"], how="left"
    ).drop(columns=["date"])
    df["time"] = df["time"].fillna("unknown")

    prices = _prepare_prices(price_df)
    price_by_ticker = {t: g.reset_index(drop=True) for t, g in prices.groupby("ticker")}

    # Build per-event recomputation table.
    records: List[Dict] = []
    for _, row in df.iterrows():
        tic = row["ticker"]
        ed = row["eps_date"]
        decision = str(row["decision"])
        time_tag = row["time"]

        entry_close, entry_date = _close_at_offset(
            price_by_ticker, tic, ed, HOLD_CUT_ENTRY_DAY
        )

        current_exit_offset = HOLD_CUT_HOLD_EXIT if decision == "hold" else HOLD_CUT_CUT_EXIT
        shifted_exit_offset = (
            current_exit_offset + AMC_SHIFT if time_tag == "amc" else current_exit_offset
        )

        cur_close, cur_date = _close_at_offset(
            price_by_ticker, tic, ed, current_exit_offset
        )
        shift_close, shift_date = _close_at_offset(
            price_by_ticker, tic, ed, shifted_exit_offset
        )

        cur_ret = (cur_close / entry_close - 1.0) if (entry_close and cur_close) else np.nan
        shift_ret = (shift_close / entry_close - 1.0) if (entry_close and shift_close) else np.nan

        records.append({
            "ticker": tic,
            "eps_date": ed,
            "time": time_tag,
            "decision": decision,
            "is_amc": time_tag == "amc",
            "entry_date": entry_date,
            "current_exit_date": cur_date,
            "shifted_exit_date": shift_date,
            "entry_close": entry_close,
            "current_exit_close": cur_close,
            "shifted_exit_close": shift_close,
            "current_ret_recomputed": cur_ret,
            "shifted_ret": shift_ret,
            "original_ret": float(row["ret"]),
            "delta_original_vs_recomputed": (cur_ret - float(row["ret"])) if cur_ret == cur_ret else np.nan,
            "delta_shifted_vs_current": (shift_ret - cur_ret) if (shift_ret == shift_ret and cur_ret == cur_ret) else np.nan,
        })

    events = pd.DataFrame(records)
    if events.empty:
        return events, pd.DataFrame()

    summary = _summarise_before_after(events)
    return events, summary


def _summary_row(r: pd.Series, label: str) -> Dict:
    r = pd.to_numeric(r, errors="coerce").dropna()
    n = int(r.size)
    mean = float(r.mean()) if n else np.nan
    hit = float((r > 0).mean()) if n else np.nan
    std = float(r.std(ddof=1)) if n > 1 else np.nan
    se = std / np.sqrt(n) if (n > 1 and std > 0) else np.nan
    t = mean / se if (se and se > 0) else np.nan
    return {"variant": label, "n": n, "mean": mean, "hit_rate": hit,
            "std": std, "t_stat": t}


def _summarise_before_after(events: pd.DataFrame) -> pd.DataFrame:
    """Three rows per cohort: original reported, recomputed current, AMC-shifted."""
    rows: List[Dict] = []

    # All events.
    rows.append(_summary_row(events["original_ret"], "all_events__original_reported"))
    rows.append(_summary_row(events["current_ret_recomputed"], "all_events__recomputed_current"))
    rows.append(_summary_row(events["shifted_ret"], "all_events__amc_shifted"))

    # AMC-only cohort.
    amc = events[events["is_amc"]]
    rows.append(_summary_row(amc["original_ret"], "amc_only__original_reported"))
    rows.append(_summary_row(amc["current_ret_recomputed"], "amc_only__recomputed_current"))
    rows.append(_summary_row(amc["shifted_ret"], "amc_only__amc_shifted"))

    # Non-AMC cohort (should show zero shift impact — sanity check).
    non_amc = events[~events["is_amc"]]
    rows.append(_summary_row(non_amc["original_ret"], "non_amc__original_reported"))
    rows.append(_summary_row(non_amc["current_ret_recomputed"], "non_amc__recomputed_current"))
    rows.append(_summary_row(non_amc["shifted_ret"], "non_amc__amc_shifted"))

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run(
    eps_csv: str = "data/eps_sp100.csv",
    calendar_csv: str = "data/earnings_calendar_sp100.csv",
    sec_csv: str = "data/sec_financials.csv",
    strategy_csv: str = "data/strategy_returns_regime_hold_cut_sp100.csv",
    price_csv: str = "data/price_data_sp100.csv",
    out_dir: str = "data",
    filter_cols: Tuple[str, ...] = ("macd_bullish",),
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    logger.info("Loading inputs…")
    eps_df = pd.read_csv(eps_csv)
    cal_df = pd.read_csv(calendar_csv)
    strategy_df = pd.read_csv(strategy_csv)
    price_df = pd.read_csv(price_csv)

    # ------------------------------------------------------------------ #
    # (1) Coverage
    # ------------------------------------------------------------------ #
    coverage = verify_calendar_coverage(eps_df, cal_df)
    coverage_path = os.path.join(out_dir, "verify_eps_coverage_summary.csv")
    coverage.to_csv(coverage_path, index=False)
    logger.info("Wrote %s", coverage_path)

    # ------------------------------------------------------------------ #
    # (2) SEC cross-check (optional — depends on sec_financials.csv)
    # ------------------------------------------------------------------ #
    if os.path.exists(sec_csv):
        sec_df = pd.read_csv(sec_csv)
        sec_cross = verify_sec_crosscheck(cal_df, sec_df)
        if not sec_cross.empty:
            sec_path = os.path.join(out_dir, "verify_eps_sec_crosscheck.csv")
            sec_cross.to_csv(sec_path, index=False)
            logger.info("Wrote %s", sec_path)
    else:
        logger.info("Skipping SEC cross-check (no %s)", sec_csv)

    # ------------------------------------------------------------------ #
    # (3) AMC entry-shift impact
    # ------------------------------------------------------------------ #
    all_summaries: List[pd.DataFrame] = []
    all_events: List[pd.DataFrame] = []

    # Unfiltered cohort.
    logger.info("AMC shift — unfiltered cohort")
    events, summary = verify_amc_shift(strategy_df, cal_df, price_df, filter_col=None)
    events["cohort"] = "unfiltered"
    summary["cohort"] = "unfiltered"
    all_events.append(events)
    all_summaries.append(summary)

    # Filtered cohorts (default: MACD-bullish).
    for fcol in filter_cols:
        if fcol not in strategy_df.columns:
            logger.warning("Filter column %r not in strategy_df; skipping.", fcol)
            continue
        logger.info("AMC shift — %s == True cohort", fcol)
        events, summary = verify_amc_shift(
            strategy_df, cal_df, price_df, filter_col=fcol, filter_val=True,
        )
        events["cohort"] = fcol
        summary["cohort"] = fcol
        all_events.append(events)
        all_summaries.append(summary)

    events_all = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    summary_all = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()

    events_path = os.path.join(out_dir, "verify_eps_amc_shift_events.csv")
    summary_path = os.path.join(out_dir, "verify_eps_amc_shift_summary.csv")
    events_all.to_csv(events_path, index=False)
    summary_all.to_csv(summary_path, index=False)
    logger.info("Wrote %s (rows=%d)", events_path, len(events_all))
    logger.info("Wrote %s (rows=%d)", summary_path, len(summary_all))


def main() -> None:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eps", default="data/eps_sp100.csv")
    parser.add_argument("--calendar", default="data/earnings_calendar_sp100.csv")
    parser.add_argument("--sec", default="data/sec_financials.csv")
    parser.add_argument("--strategy", default="data/strategy_returns_regime_hold_cut_sp100.csv")
    parser.add_argument("--prices", default="data/price_data_sp100.csv")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--filter-cols", nargs="*", default=["macd_bullish"])
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run(
        eps_csv=args.eps,
        calendar_csv=args.calendar,
        sec_csv=args.sec,
        strategy_csv=args.strategy,
        price_csv=args.prices,
        out_dir=args.out_dir,
        filter_cols=tuple(args.filter_cols),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
