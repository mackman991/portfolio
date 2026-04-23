"""Runner: technical-regime × earnings interaction analysis on the SP100 panel.

Outputs (written to config.DATA_DIR):
  - regime_features_sp100.csv          per-event snapshot + regime flags
  - strategy_returns_regime_sp100.csv  per-event returns for each strategy
                                       with all regime features joined
  - regime_grid_<strategy>_sp100.csv   (regime, bucket) -> n/mean/hit/t-stat
  - regime_uplift_<strategy>_sp100.csv same grid sorted by mean-return uplift

Usage:
  python -m src.run_technical_regime

Look-ahead audit
----------------
  - Post-earnings strategies ("momentum_1_H", "hold_cut") enter at the OPEN of
    rel_day +1. Indicator values at the CLOSE of rel_day 0 are observable
    before that entry, so we snapshot regime features at rel_day=0.

  - Pre-earnings strategy ("pre_runup_P_1") enters at the OPEN of rel_day -P
    (default -10). We therefore snapshot at rel_day=-11: values at the close
    of -11 are observable before the -10 open.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List

import pandas as pd

from src import config
from src.analysis.technical_regime import (
    REGIME_COLS,
    add_regime_features,
    grid_splits,
    per_event_returns,
    snapshot_at_rel_day,
    uplift_table,
)

logger = logging.getLogger(__name__)


STRATEGY_SPECS = [
    # (name, snapshot rel_day, per-event-returns kwargs)
    ("momentum_1_H",  0,   {"strategy": "momentum_1_H",  "H": 6}),
    ("hold_cut",      0,   {"strategy": "hold_cut",      "P": 10,
                            "hold_cut_threshold": 0.05, "hold_cut_extension": 10}),
    ("pre_runup_P_1", -11, {"strategy": "pre_runup_P_1", "P": 10}),
]


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the technical-regime × earnings analysis.")
    p.add_argument("--events", default=None,
                   help="Path to event-window CSV (default: data/eps_event_windows_15_sp100.csv).")
    p.add_argument("--indicators", default=None,
                   help="Path to indicators CSV (default: data/indicators_sp100.csv).")
    p.add_argument("--out-prefix", default="sp100",
                   help="Suffix for output files (default: sp100).")
    p.add_argument("--min-n", type=int, default=50,
                   help="Minimum bucket size before suppressing a split in the grid.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    data_dir = config.DATA_DIR
    os.makedirs(data_dir, exist_ok=True)
    prefix = args.out_prefix

    events_path = args.events or os.path.join(data_dir, "eps_event_windows_15_sp100.csv")
    ind_path = args.indicators or os.path.join(data_dir, "indicators_sp100.csv")
    cal_path = os.path.join(data_dir, "earnings_calendar_sp100.csv")

    logger.info("Loading event windows: %s", events_path)
    evw = pd.read_csv(events_path, parse_dates=["eps_date", "price_date"])
    logger.info("  rows=%d events=%d tickers=%d",
                len(evw), evw[["ticker","eps_date"]].drop_duplicates().shape[0],
                evw.ticker.nunique())

    logger.info("Loading indicators: %s", ind_path)
    ind = pd.read_csv(ind_path, parse_dates=["date"])
    logger.info("  rows=%d tickers=%d range=%s..%s",
                len(ind), ind.ticker.nunique(), ind.date.min().date(), ind.date.max().date())

    # Build AMC set for the hold_cut correction
    amc_set: set = set()
    if os.path.exists(cal_path):
        cal = pd.read_csv(cal_path)
        cal["date"] = pd.to_datetime(cal["date"], errors="coerce").dt.date.astype(str)
        amc_rows = cal[cal["time"] == "amc"][["ticker", "date"]].dropna()
        amc_set = set(zip(amc_rows["ticker"], amc_rows["date"]))
        logger.info("AMC set built: %d (ticker, date) pairs from %s", len(amc_set), cal_path)
    else:
        logger.warning("No earnings calendar found at %s — AMC correction skipped", cal_path)

    # Per-strategy: snapshot -> features -> returns -> grid
    for name, rel_day, kwargs in STRATEGY_SPECS:
        logger.info("=== Strategy: %s (snapshot rel_day=%d) ===", name, rel_day)

        snap = snapshot_at_rel_day(evw, ind, at_rel_day=rel_day)
        feats = add_regime_features(snap)
        coverage = feats["rsi_14"].notna().mean()
        logger.info("  snapshot rows=%d coverage=%.1f%% of events",
                    len(feats), 100 * coverage)

        extra = {"amc_set": amc_set} if kwargs.get("strategy") == "hold_cut" else {}
        per_ev = per_event_returns(evw, **kwargs, **extra)
        logger.info("  per-event returns: n=%d", len(per_ev))

        merged = per_ev.merge(
            feats,
            on=["ticker", "eps_date"],
            how="left",
        )

        grid = grid_splits(merged, REGIME_COLS, min_n=args.min_n)
        up = uplift_table(grid)

        features_out = os.path.join(data_dir, f"regime_features_{name}_{prefix}.csv")
        returns_out  = os.path.join(data_dir, f"strategy_returns_regime_{name}_{prefix}.csv")
        grid_out     = os.path.join(data_dir, f"regime_grid_{name}_{prefix}.csv")
        up_out       = os.path.join(data_dir, f"regime_uplift_{name}_{prefix}.csv")

        feats.to_csv(features_out, index=False)
        merged.to_csv(returns_out, index=False)
        grid.to_csv(grid_out, index=False)
        up.to_csv(up_out, index=False)

        logger.info("  saved: %s, %s, %s, %s",
                    os.path.basename(features_out), os.path.basename(returns_out),
                    os.path.basename(grid_out), os.path.basename(up_out))

        # Log the top-5 uplift buckets for a quick eyeball
        top = up.dropna(subset=["mean"]).head(5)
        logger.info("  Top-5 uplift buckets (filter out small_n=True afterwards):")
        for _, r in top.iterrows():
            logger.info("    %-24s %-20s n=%4d mean=%+.2f%% hit=%.1f%% t=%+.2f uplift=%+.2f%% small_n=%s",
                        r["regime"], r["bucket"], int(r["n"]),
                        100 * r["mean"], 100 * r["hit_rate"], r["t_stat"],
                        100 * r["mean_uplift"], r["small_n"])

    logger.info("Done. All outputs in %s/", data_dir)


if __name__ == "__main__":  # pragma: no cover
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
