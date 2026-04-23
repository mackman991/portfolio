"""Pre-earnings run-up strategy with explicit look-ahead-bias warning.

⚠️  IMPORTANT — LOOK-AHEAD BIAS
================================
The original capstone notebook reported that filtering pre-earnings entries by
``|eps_surprise| > 5%`` lifted average per-event return from 2.85% to 5.95%
and hit-rate from 62% to 75%. **That filter is not tradable in real life**:
``eps_surprise = (actual - estimate) / |estimate|`` is only knowable AFTER the
earnings release, but the strategy entries fire BEFORE the release. Using
``surprise_filter`` therefore selects future winners with hindsight.

The naive variant (``surprise_filter=None``) is tradable. Anything that
filters on ``eps_surprise`` is research-only and clearly tagged as such in
the printed output.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PreEarningsStrategy:
    """Long the equity from ``entry_day`` to ``exit_day`` around each EPS print."""

    REQUIRED = {"ticker", "eps_date", "ret", "rel_day", "beat", "eps_surprise"}

    def __init__(self, df: Optional[pd.DataFrame] = None, data_path: Optional[str] = None):
        if df is not None:
            self.df = df.copy()
        elif data_path:
            self.df = pd.read_csv(data_path)
        else:
            raise ValueError("Provide either df or data_path.")
        self._prepare()

    def _prepare(self) -> None:
        df = self.df.copy()

        rename_map = {}
        if "surprise" in df.columns and "eps_surprise" not in df.columns:
            rename_map["surprise"] = "eps_surprise"
        if "stock splits" in df.columns and "stock_splits" not in df.columns:
            rename_map["stock splits"] = "stock_splits"
        df = df.rename(columns=rename_map)

        if "date" not in df.columns:
            for cand in ("price_date", "Price_Date", "Date", "DATE"):
                if cand in df.columns:
                    df["date"] = df[cand]
                    break

        missing = self.REQUIRED - set(df.columns)
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

        for c in ("date", "eps_date", "price_date"):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
        for c in ("ret", "rel_day", "eps_surprise", "close"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        if df["beat"].dtype != bool:
            df["beat"] = (
                df["beat"].astype(str).str.strip().str.lower()
                .map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
                .fillna(False)
            )

        critical = ["ret", "rel_day", "eps_surprise", "eps_date", "ticker"]
        df = df.dropna(subset=critical).copy()
        df["rel_day"] = df["rel_day"].astype(int)
        self.df = df

        logger.info(
            "Loaded %d obs, %d events, %d tickers",
            len(df),
            df.groupby(["ticker", "eps_date"]).ngroup().nunique(),
            df["ticker"].nunique(),
        )

    def calculate_strategy_returns(
        self,
        entry_day: int = -10,
        exit_day: int = -1,
        surprise_filter: Optional[float] = None,
    ) -> pd.DataFrame:
        """Cumulative return per event over [entry_day, exit_day].

        Args:
            entry_day: rel_day to enter (negative = before announcement).
            exit_day: rel_day to exit.
            surprise_filter: if set, only trade events where
                ``|eps_surprise| > surprise_filter``. **Look-ahead bias** —
                research only.
        """
        if surprise_filter is not None:
            logger.warning(
                "surprise_filter=%s introduces look-ahead bias; results are research-only.",
                surprise_filter,
            )

        events = self.df.groupby(["ticker", "eps_date"]).first().reset_index()
        out = []
        for _, event in events.iterrows():
            if surprise_filter is not None and abs(event["eps_surprise"]) <= surprise_filter:
                continue

            event_data = self.df[
                (self.df["ticker"] == event["ticker"])
                & (self.df["eps_date"] == event["eps_date"])
            ].sort_values("rel_day")
            window = event_data[
                (event_data["rel_day"] >= entry_day) & (event_data["rel_day"] <= exit_day)
            ]
            expected_days = abs(exit_day - entry_day) + 1
            if len(window) < max(8, expected_days * 0.8):
                continue

            cum_ret = float(np.prod(1 + window["ret"].values) - 1)
            out.append({
                "ticker": event["ticker"],
                "eps_date": event["eps_date"],
                "entry_day": entry_day,
                "exit_day": exit_day,
                "return": cum_ret,
                "beat": event["beat"],
                "eps_surprise": event["eps_surprise"],
                "days_held": len(window),
                "surprise_filter": surprise_filter,
            })
        return pd.DataFrame(out)

    def calculate_with_post_event_cut(
        self,
        entry_day: int = -10,
        base_exit_day: int = 0,
        extended_exit_day: int = 5,
        surprise_threshold: float = 0.05,
    ) -> pd.DataFrame:
        """Tradable variant: enter early, use the surprise (known post-print) to decide hold/cut.

        Mechanics:
          1. Enter long at ``entry_day`` on EVERY event (no pre-entry filter).
          2. At day 0, the EPS surprise becomes public knowledge.
          3. If ``|surprise| > surprise_threshold``, hold through ``extended_exit_day``;
             otherwise close at ``base_exit_day``.

        This is NOT look-ahead — the surprise is observed before the hold/cut
        decision. Contrast with ``calculate_strategy_returns(surprise_filter=...)``,
        which uses the surprise to decide at ``entry_day`` whether to enter at all.
        """
        events = self.df.groupby(["ticker", "eps_date"]).first().reset_index()
        out = []
        for _, event in events.iterrows():
            event_data = self.df[
                (self.df["ticker"] == event["ticker"])
                & (self.df["eps_date"] == event["eps_date"])
            ].sort_values("rel_day")

            exit_day = (
                extended_exit_day
                if abs(event["eps_surprise"]) > surprise_threshold
                else base_exit_day
            )
            window = event_data[
                (event_data["rel_day"] >= entry_day) & (event_data["rel_day"] <= exit_day)
            ]
            expected_days = abs(exit_day - entry_day) + 1
            if len(window) < max(8, expected_days * 0.8):
                continue

            cum_ret = float(np.prod(1 + window["ret"].values) - 1)
            out.append({
                "ticker": event["ticker"],
                "eps_date": event["eps_date"],
                "entry_day": entry_day,
                "exit_day": exit_day,
                "decision": "hold" if exit_day == extended_exit_day else "cut",
                "return": cum_ret,
                "beat": event["beat"],
                "eps_surprise": event["eps_surprise"],
                "days_held": len(window),
            })
        return pd.DataFrame(out)

    def optimize_timing(
        self,
        entry_range: Tuple[int, int] = (-15, -5),
        exit_range: Tuple[int, int] = (-3, 3),
        min_trades: int = 30,
    ) -> pd.DataFrame:
        """Grid-search entry/exit windows. No surprise filter (tradable variant)."""
        rows = []
        for entry in range(entry_range[0], entry_range[1]):
            for exit_d in range(exit_range[0], exit_range[1]):
                if entry >= exit_d:
                    continue
                res = self.calculate_strategy_returns(entry, exit_d)
                if len(res) < min_trades:
                    continue
                r = res["return"]
                std = r.std()
                rows.append({
                    "entry_day": entry,
                    "exit_day": exit_d,
                    "avg_return": r.mean(),
                    "hit_rate": (r > 0).mean(),
                    "t_stat": r.mean() / (std / np.sqrt(len(r))) if std > 0 else np.nan,
                    "sharpe": r.mean() / std if std > 0 else np.nan,
                    "n_trades": len(r),
                    "total_days": exit_d - entry + 1,
                })
        return pd.DataFrame(rows)
