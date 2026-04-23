"""Live monitoring + trade-watchlist for the MACD-bullish Hold/Cut strategy.

Three artefacts are produced under ``data/``:

* ``watchlist.csv``          — upcoming and recent S&P 100 earnings events
                               with the state of the MACD filter on the
                               strategy's day-−10 entry. Sorted so the rows
                               that are actionable right now float to the top.
* ``monitoring_summary.csv`` — rolling-30-event edge metrics computed on
                               realised MACD-bullish trades (hit-rate, mean,
                               t-stat, Sharpe) vs the in-sample baseline.
                               Flags "alarm" when a metric falls outside the
                               pre-agreed band for two consecutive windows.
* ``monitoring_health.csv``  — one-row pipeline-health snapshot (calendar
                               coverage, SEC agreement, AMC-shift delta,
                               freshness of the latest indicator pull).

The module is deliberately conservative: it reads the CSVs the existing
pipeline already produces and computes plain pandas rollups. Nothing here
fits parameters, decides trades, or touches brokerage APIs — the purpose is
to tell the human whether the edge that was measured in-sample is still
showing up in realised events, and to surface the small list of upcoming
announcements where today's MACD state says the filter *would* fire.

Definitions
-----------
Entry day
    The close of ``rel_day = -10`` relative to the earnings announcement,
    measured in trading days on the NYSE/NASDAQ calendar (approximated here
    by the union of dates present in ``indicators_sp100_v2.csv``).

MACD-bullish
    ``macd_line > macd_signal`` at the close of the entry day.

Status tags on the watchlist
    * ``pre-watch``       — entry day is more than ~2 trading days ahead;
                            MACD state shown is a *preview* using the latest
                            available indicator close, not the true entry.
    * ``entry-imminent``  — entry day is within the next ~2 trading days.
    * ``entered``         — entry day is today or has already passed,
                            announcement has not.
    * ``post-event``      — announcement has happened, exit window still open
                            or recently closed. Realised return reported if
                            present in the strategy returns file.
    * ``closed``          — realised return is available.

All relative-day arithmetic ignores the AMC exit-shift here; that correction
is applied downstream by ``src.analysis.verify_eps_dates`` and reflected in
``strategy_returns_regime_hold_cut_sp100_amc_corrected.csv``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Strategy constants (mirror src.analysis.verify_eps_dates) ─────────────
HOLD_CUT_ENTRY_REL_DAY = -10
HOLD_CUT_EXIT_HOLD_REL_DAY = 10
HOLD_CUT_EXIT_CUT_REL_DAY = 0

# ── Monitoring configuration ──────────────────────────────────────────────
ROLLING_WINDOW = 30         # events
ALARM_CONSECUTIVE = 2        # windows below threshold to trigger alarm

# Baseline edge metrics from the in-sample MACD-bullish cohort (AMC-corrected
# per Report_DataQuality.docx). Alarm bands are chosen to survive normal
# sampling noise at n=30 but flag a real break in the edge.
BASELINE = {
    "mean_per_event": 0.0580,   # AMC-corrected
    "hit_rate": 0.810,
    "t_stat": 16.69,            # on the full 894 AMC-corrected sample
    "sharpe_daily": 3.63,
}
ALARM = {
    "mean_per_event": 0.030,    # <3% per event is a warning
    "hit_rate": 0.700,          # <70% is a warning
    "sharpe_daily": 2.00,       # sized-curve daily-Sharpe
}

# Data-quality thresholds (from Report_DataQuality.docx results tables).
DQ_THRESHOLDS = {
    "calendar_match_min": 1.00,         # 100% of eps_sp100 events matched
    "sec_crosscheck_min": 0.96,         # at least 96% within ±3 trading days
    "amc_delta_bps_range": (-0.005, 0.001),  # per-AMC-event delta expected
}

DEFAULT_INDICATORS_PATH = "data/indicators_sp100_v2.csv"
DEFAULT_CALENDAR_PATH = "data/earnings_calendar_sp100.csv"
DEFAULT_REALISED_PATHS = (
    "data/strategy_returns_regime_hold_cut_sp100_amc_corrected.csv",
    "data/strategy_returns_regime_hold_cut_sp100.csv",
)

DEFAULT_OUT_WATCHLIST = "data/watchlist.csv"
DEFAULT_OUT_SUMMARY = "data/monitoring_summary.csv"
DEFAULT_OUT_HEALTH = "data/monitoring_health.csv"


# ─────────────────────────────────────────────────────────────────────────── #
# Loaders
# ─────────────────────────────────────────────────────────────────────────── #
def load_indicators(path: str = DEFAULT_INDICATORS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def load_calendar(path: str = DEFAULT_CALENDAR_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.rename(columns={"date": "eps_date"})
    # Normalise the timing tag: FMP returns lowercase bmo/amc/dmh, with
    # occasional nulls or "--". The extractor already canonicalises, but we
    # re-assert here in case a different pull is dropped in.
    if "time" in df.columns:
        df["time"] = (
            df["time"].astype(str).str.strip().str.lower()
            .replace({"--": None, "nan": None, "none": None, "": None})
        )
    df["is_amc"] = df.get("time").eq("amc") if "time" in df.columns else False
    return df.sort_values(["ticker", "eps_date"]).reset_index(drop=True)


def load_realised(
    paths: tuple[str, ...] = DEFAULT_REALISED_PATHS,
) -> pd.DataFrame:
    for p in paths:
        if os.path.exists(p):
            logger.info("monitor: using realised returns from %s", p)
            df = pd.read_csv(p, parse_dates=["eps_date"])
            return df.sort_values("eps_date").reset_index(drop=True)
    logger.warning("monitor: no realised returns file found in %s", paths)
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────── #
# Trading-day calendar helpers
# ─────────────────────────────────────────────────────────────────────────── #
def trading_calendar(
    indicators: pd.DataFrame, extend_days: int = 120
) -> pd.DatetimeIndex:
    """Trading-day calendar spanning the indicator panel plus a forward window.

    The indicator panel is essentially the NYSE/NASDAQ calendar up to the last
    pull date. For forward-looking entry-date arithmetic on upcoming events we
    extend the calendar by roughly six months using ``pd.bdate_range`` (which
    excludes Sat/Sun but *not* US market holidays — an acceptable approximation
    given the strategy trades at daily closes and the few mis-placed holidays
    only shift the preview by a session).
    """
    observed = pd.DatetimeIndex(sorted(indicators["date"].unique()))
    if len(observed) == 0:
        return observed
    last = observed[-1]
    forward = pd.bdate_range(
        start=last + pd.Timedelta(days=1),
        periods=extend_days,
    )
    return observed.append(forward).unique().sort_values()


def nearest_trading_day(cal: pd.DatetimeIndex, day: pd.Timestamp) -> pd.Timestamp:
    """Largest trading day ≤ ``day``. Used to anchor ``today`` to the last
    completed session when the user runs the monitor mid-day."""
    idx = cal.searchsorted(day, side="right") - 1
    if idx < 0:
        return cal[0]
    return cal[int(idx)]


def offset_trading_day(
    cal: pd.DatetimeIndex, anchor: pd.Timestamp, offset: int
) -> tuple[pd.Timestamp, int]:
    """Return (date, clipped_offset) for ``anchor + offset`` trading days.

    Negative offsets walk backwards. If the target would fall outside the
    calendar, we clip and report the offset actually used.
    """
    # Anchor to the first trading day >= anchor so the arithmetic is stable
    # for event days that fall on weekends/holidays.
    a_idx = cal.searchsorted(anchor, side="left")
    if a_idx >= len(cal):
        a_idx = len(cal) - 1
    target = a_idx + offset
    if target < 0:
        return cal[0], -a_idx
    if target >= len(cal):
        return cal[-1], (len(cal) - 1) - a_idx
    return cal[int(target)], offset


def session_count_between(
    cal: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp
) -> int:
    """Trading days in (start, end]. Negative if end < start."""
    if end == start:
        return 0
    s = cal.searchsorted(start, side="left")
    e = cal.searchsorted(end, side="left")
    return int(e - s)


# ─────────────────────────────────────────────────────────────────────────── #
# Watchlist
# ─────────────────────────────────────────────────────────────────────────── #
@dataclass
class WatchlistConfig:
    lookback_days: int = 20        # trading days of realised events to include
    lookforward_days: int = 60     # trading days of upcoming events to include
    entry_rel_day: int = HOLD_CUT_ENTRY_REL_DAY


def _macd_state_on(
    indicators: pd.DataFrame, ticker: str, on_or_before: pd.Timestamp
) -> pd.Series:
    """Most-recent indicator row for ``ticker`` at-or-before ``on_or_before``.

    Returns an empty row of NaNs if the ticker has no coverage that far back.
    """
    sub = indicators.loc[
        (indicators["ticker"] == ticker) & (indicators["date"] <= on_or_before)
    ]
    if sub.empty:
        return pd.Series(dtype="float64")
    return sub.iloc[-1]


def compute_watchlist(
    indicators: pd.DataFrame,
    calendar: pd.DataFrame,
    today: pd.Timestamp,
    realised: Optional[pd.DataFrame] = None,
    cfg: Optional[WatchlistConfig] = None,
) -> pd.DataFrame:
    """Build the forward-looking trade watchlist.

    One row per earnings event in the window
    ``[today - lookback, today + lookforward]`` (measured in calendar days,
    not trading days — we just want a broad-enough window). For each event,
    compute:

    * the actual entry date (close of rel_day = −10)
    * MACD state on that entry date if observable, or the latest available
      indicator close as a preview if the entry is still in the future
    * a status tag (see module docstring)
    * realised return if present in ``realised``
    """
    if cfg is None:
        cfg = WatchlistConfig()

    cal = trading_calendar(indicators)
    today_td = nearest_trading_day(cal, today)

    window_start = today - pd.Timedelta(days=cfg.lookback_days * 2)   # slack
    window_end = today + pd.Timedelta(days=cfg.lookforward_days * 2)
    upcoming = calendar.loc[
        (calendar["eps_date"] >= window_start)
        & (calendar["eps_date"] <= window_end)
    ].copy()

    # Pre-index realised returns for quick lookup.
    realised_idx = None
    if realised is not None and not realised.empty:
        realised_idx = realised.set_index(["ticker", "eps_date"])[
            ["ret", "decision", "amc", "macd_bullish", "snapshot_close", "snapshot_date"]
        ].copy()

    rows: list[dict] = []
    for _, ev in upcoming.iterrows():
        ticker = ev["ticker"]
        eps_date = ev["eps_date"]
        entry_dt, _ = offset_trading_day(cal, eps_date, cfg.entry_rel_day)
        days_to_entry = session_count_between(cal, today_td, entry_dt)
        days_to_eps = session_count_between(cal, today_td, eps_date)

        # MACD state: prefer the entry-day row if observable; otherwise latest.
        macd_ref_date = min(entry_dt, cal[-1])
        macd_state = _macd_state_on(indicators, ticker, macd_ref_date)
        macd = float(macd_state.get("macd", np.nan)) if len(macd_state) else np.nan
        macd_signal = float(macd_state.get("macd_signal", np.nan)) if len(macd_state) else np.nan
        rsi = float(macd_state.get("rsi_14", np.nan)) if len(macd_state) else np.nan
        macd_bullish = (not np.isnan(macd)) and (not np.isnan(macd_signal)) and macd > macd_signal
        entry_observable = (entry_dt <= cal[-1])

        # Status classification.
        if realised_idx is not None and (ticker, eps_date) in realised_idx.index:
            status = "closed"
        elif days_to_eps < 0 - HOLD_CUT_EXIT_HOLD_REL_DAY:
            status = "post-event"
        elif days_to_entry > 2:
            status = "pre-watch"
        elif days_to_entry >= 0:
            status = "entry-imminent"
        elif days_to_eps >= 0:
            status = "entered"
        else:
            status = "post-event"

        # Realised return if we have one.
        ret = decision = amc_flag = np.nan
        if realised_idx is not None and (ticker, eps_date) in realised_idx.index:
            realised_row = realised_idx.loc[(ticker, eps_date)]
            ret = float(realised_row.get("ret", np.nan))
            decision = realised_row.get("decision", None)
            amc_flag = realised_row.get("amc", None)

        rows.append(
            {
                "ticker": ticker,
                "eps_date": eps_date,
                "timing": ev.get("time"),
                "is_amc": bool(ev.get("is_amc", False)),
                "entry_date": entry_dt,
                "entry_observable": entry_observable,
                "days_to_entry": days_to_entry,
                "days_to_eps": days_to_eps,
                "macd": macd,
                "macd_signal": macd_signal,
                "macd_bullish": macd_bullish,
                "rsi_14": rsi,
                "status": status,
                "realised_ret": ret,
                "decision": decision,
                "eps_estimate": ev.get("epsestimated"),
                "revenue_estimate": ev.get("revenueestimated"),
            }
        )

    wl = pd.DataFrame(rows)
    if wl.empty:
        return wl

    # Rank order: actionable rows first, then by entry date.
    status_priority = {
        "entry-imminent": 0,
        "entered": 1,
        "pre-watch": 2,
        "post-event": 3,
        "closed": 4,
    }
    wl["_prio"] = wl["status"].map(status_priority).fillna(9)
    wl = wl.sort_values(
        ["_prio", "macd_bullish", "entry_date"],
        ascending=[True, False, True],
    ).drop(columns="_prio").reset_index(drop=True)

    return wl


# ─────────────────────────────────────────────────────────────────────────── #
# Rolling edge metrics
# ─────────────────────────────────────────────────────────────────────────── #
def rolling_metrics(
    realised: pd.DataFrame,
    window: int = ROLLING_WINDOW,
    filter_col: Optional[str] = "macd_bullish",
) -> pd.DataFrame:
    """Compute rolling-``window`` edge metrics on realised per-event returns.

    If ``filter_col`` is given, rolls only over rows where that column is True
    (i.e. the MACD-bullish cohort).
    """
    if realised.empty:
        return pd.DataFrame()

    df = realised.copy().sort_values("eps_date")
    if filter_col and filter_col in df.columns:
        df = df.loc[df[filter_col].astype(bool)].copy()

    df["hit"] = (df["ret"] > 0).astype(float)

    roll_mean = df["ret"].rolling(window).mean()
    roll_std = df["ret"].rolling(window).std(ddof=1)
    roll_hit = df["hit"].rolling(window).mean()
    n = df["ret"].rolling(window).count()

    roll_tstat = roll_mean / (roll_std / np.sqrt(n))
    # Rough proxy for daily-Sharpe expressed in per-event units; the sized
    # simulator gives the honest daily-Sharpe. This is here as a quick sanity
    # metric on the per-event series, not a replacement.
    roll_er_sharpe = roll_mean / roll_std

    out = pd.DataFrame(
        {
            "eps_date": df["eps_date"].values,
            "ticker": df["ticker"].values,
            "ret": df["ret"].values,
            "n_in_window": n.values,
            "rolling_mean": roll_mean.values,
            "rolling_std": roll_std.values,
            "rolling_hit_rate": roll_hit.values,
            "rolling_tstat": roll_tstat.values,
            "rolling_event_sharpe": roll_er_sharpe.values,
        }
    )
    out["alarm_mean"] = out["rolling_mean"] < ALARM["mean_per_event"]
    out["alarm_hit"] = out["rolling_hit_rate"] < ALARM["hit_rate"]
    # Consecutive-window alarm: two in a row.
    out["alarm_mean_confirmed"] = (
        out["alarm_mean"].rolling(ALARM_CONSECUTIVE).sum() == ALARM_CONSECUTIVE
    )
    out["alarm_hit_confirmed"] = (
        out["alarm_hit"].rolling(ALARM_CONSECUTIVE).sum() == ALARM_CONSECUTIVE
    )
    return out


# ─────────────────────────────────────────────────────────────────────────── #
# Data-quality snapshot
# ─────────────────────────────────────────────────────────────────────────── #
def data_quality_snapshot(data_dir: str = "data") -> dict:
    """One-row pipeline-health report. Reads whatever verify_eps_* artefacts
    are present and returns a flat dict with pass/fail flags.
    """
    snap: dict = {}

    # Freshness of the indicator pull.
    ind_path = os.path.join(data_dir, "indicators_sp100_v2.csv")
    if os.path.exists(ind_path):
        ind = pd.read_csv(ind_path, usecols=["date"], parse_dates=["date"])
        snap["indicators_last_date"] = ind["date"].max().date().isoformat()
        snap["indicators_rows"] = len(ind)
    else:
        snap["indicators_last_date"] = None
        snap["indicators_rows"] = 0

    # Calendar coverage vs eps sp100.
    # Schema: source, bucket, count. Matched rows live under source="both",
    # one-sided rows under source="eps_only" / "calendar_only".
    cov_path = os.path.join(data_dir, "verify_eps_coverage_summary.csv")
    if os.path.exists(cov_path):
        cov = pd.read_csv(cov_path)
        matched = int(cov.loc[cov["source"] == "both", "count"].sum())
        only_eps = int(cov.loc[cov["source"] == "eps_only", "count"].sum())
        total = matched + only_eps
        snap["calendar_match_rate"] = float(matched / total) if total else None
        snap["calendar_match_rate_ok"] = (
            snap["calendar_match_rate"] is not None
            and snap["calendar_match_rate"] >= DQ_THRESHOLDS["calendar_match_min"]
        )
    else:
        snap["calendar_match_rate"] = None
        snap["calendar_match_rate_ok"] = None

    # SEC cross-check agreement. The file is per-event with columns
    # (ticker, sec_date, calendar_date, calendar_time, delta_days, match).
    sec_path = os.path.join(data_dir, "verify_eps_sec_crosscheck.csv")
    if os.path.exists(sec_path):
        sec = pd.read_csv(sec_path)
        if "match" in sec.columns and len(sec):
            match_bool = sec["match"].astype(bool)
            rate = float(match_bool.mean())
            snap["sec_match_rate"] = rate
            snap["sec_match_rate_ok"] = rate >= DQ_THRESHOLDS["sec_crosscheck_min"]
            snap["sec_exact_match_rate"] = float((sec["delta_days"].abs() == 0).mean()) if "delta_days" in sec.columns else None
        else:
            snap["sec_match_rate"] = None
            snap["sec_match_rate_ok"] = None
            snap["sec_exact_match_rate"] = None
    else:
        snap["sec_match_rate"] = None
        snap["sec_match_rate_ok"] = None
        snap["sec_exact_match_rate"] = None

    # AMC entry-shift delta. The summary file holds the original/recomputed/
    # shifted mean per cohort; the meaningful delta is
    # amc_only__amc_shifted - amc_only__recomputed_current under cohort=unfiltered.
    amc_path = os.path.join(data_dir, "verify_eps_amc_shift_summary.csv")
    if os.path.exists(amc_path):
        amc = pd.read_csv(amc_path)
        unf = amc.loc[amc["cohort"] == "unfiltered"] if "cohort" in amc.columns else amc
        try:
            base = float(
                unf.loc[unf["variant"] == "amc_only__recomputed_current", "mean"].iloc[0]
            )
            shifted = float(
                unf.loc[unf["variant"] == "amc_only__amc_shifted", "mean"].iloc[0]
            )
            delta = shifted - base
            snap["amc_mean_delta_ret"] = delta
            lo, hi = DQ_THRESHOLDS["amc_delta_bps_range"]
            snap["amc_delta_in_band"] = (not np.isnan(delta)) and (lo <= delta <= hi)
        except (IndexError, KeyError):
            snap["amc_mean_delta_ret"] = None
            snap["amc_delta_in_band"] = None
    else:
        snap["amc_mean_delta_ret"] = None
        snap["amc_delta_in_band"] = None

    return snap


# ─────────────────────────────────────────────────────────────────────────── #
# Orchestration
# ─────────────────────────────────────────────────────────────────────────── #
def run(
    data_dir: str = "data",
    today: Optional[pd.Timestamp] = None,
    window: int = ROLLING_WINDOW,
) -> dict:
    """Build all three artefacts. Returns a dict of paths for downstream use."""
    if today is None:
        today = pd.Timestamp.today().normalize()

    ind = load_indicators(os.path.join(data_dir, os.path.basename(DEFAULT_INDICATORS_PATH)))
    cal = load_calendar(os.path.join(data_dir, os.path.basename(DEFAULT_CALENDAR_PATH)))
    realised = load_realised(
        tuple(os.path.join(data_dir, os.path.basename(p)) for p in DEFAULT_REALISED_PATHS)
    )

    wl = compute_watchlist(ind, cal, today, realised=realised)
    rm = rolling_metrics(realised, window=window)
    dq = data_quality_snapshot(data_dir=data_dir)

    out_wl = os.path.join(data_dir, os.path.basename(DEFAULT_OUT_WATCHLIST))
    out_rm = os.path.join(data_dir, os.path.basename(DEFAULT_OUT_SUMMARY))
    out_dq = os.path.join(data_dir, os.path.basename(DEFAULT_OUT_HEALTH))

    wl.to_csv(out_wl, index=False)
    rm.to_csv(out_rm, index=False)
    pd.DataFrame([{"generated_at": pd.Timestamp.utcnow().isoformat(), **dq}]).to_csv(
        out_dq, index=False
    )

    logger.info("monitor: watchlist rows=%d -> %s", len(wl), out_wl)
    logger.info("monitor: rolling rows=%d -> %s", len(rm), out_rm)
    logger.info("monitor: health -> %s", out_dq)

    return {"watchlist": out_wl, "summary": out_rm, "health": out_dq}


# ─────────────────────────────────────────────────────────────────────────── #
# CLI
# ─────────────────────────────────────────────────────────────────────────── #
def main() -> None:  # pragma: no cover - thin wrapper
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--today", default=None, help="Override today (YYYY-MM-DD).")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    today = pd.Timestamp(args.today) if args.today else pd.Timestamp.today().normalize()
    paths = run(data_dir=args.data_dir, today=today, window=args.window)
    for k, v in paths.items():
        logger.info("  %s: %s", k, v)


if __name__ == "__main__":  # pragma: no cover
    main()
