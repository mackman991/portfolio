"""Realistic position-sizing backtest for the Hold/Cut strategy.

Replaces the unrealistic 100%-per-event compounding assumption with a
portfolio-level, equity-scaled sizing framework:
  * Base size: 3% of current equity per new event
  * Pyramid: +3% tranche (total 6%) on day +1 if close > entry close * 1.01
  * Gross-exposure cap: 30% of current equity. New entries or pyramid
    tranches that would breach the cap are skipped (tracked in stats).
  * Exit: uses the existing strategy's realised per-event `ret`. For
    capital-release timing in the portfolio walk, exit_date is taken as
    the 10th trading day after entry (unified with the Hold/Cut horizon).

Outputs, written under `data/`:
  position_sizing_summary.csv           — per-variant headline metrics
  equity_curve_sized_<variant>.csv      — daily equity series per variant

Variants:
  unfiltered_flat       — all events, base only (no pyramid)
  unfiltered_pyramid    — all events, base + pyramid
  macd_bullish_flat     — MACD-bullish filter, base only
  macd_bullish_pyramid  — MACD-bullish filter, base + pyramid
"""
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_WEIGHT = 0.03        # 3% of equity per new event
PYRAMID_WEIGHT = 0.03     # + 3% when price-confirmation fires
GROSS_CAP = 0.30          # max 30% of equity across open positions
PYRAMID_THRESHOLD = 0.01  # day +1 close > entry * 1.01 triggers pyramid
HOLD_DAYS = 10            # approximate exit horizon for capital-release timing


# --- trade-level preparation ---------------------------------------------

@dataclass
class EventPath:
    ticker: str
    entry_date: pd.Timestamp
    day1_date: Optional[pd.Timestamp]
    exit_date: pd.Timestamp
    entry_price: float
    day1_price: Optional[float]
    base_return: float
    pyramid_return: float       # return from day+1 close to exit close
    pyramid_triggered: bool
    macd_bullish: bool


def _prepare_prices(prices: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    p = p.dropna(subset=["date"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    return {t: g.reset_index(drop=True) for t, g in p.groupby("ticker")}


def _trading_calendar(prices_by_ticker: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    all_days: set = set()
    for g in prices_by_ticker.values():
        all_days.update(g["date"].tolist())
    return pd.DatetimeIndex(sorted(all_days))


def _build_event_paths(events: pd.DataFrame,
                       prices_by_ticker: Dict[str, pd.DataFrame],
                       calendar: pd.DatetimeIndex) -> List[EventPath]:
    paths: List[EventPath] = []
    skipped = 0
    for r in events.itertuples(index=False):
        tic = r.ticker
        ed = pd.to_datetime(r.eps_date).normalize()
        g = prices_by_ticker.get(tic)
        if g is None or g.empty:
            skipped += 1; continue
        idx_entry = g.index[g["date"] >= ed]
        if len(idx_entry) == 0:
            skipped += 1; continue
        i0 = int(idx_entry[0])
        entry_date = g.loc[i0, "date"]
        entry_price = float(g.loc[i0, "close"])
        if not np.isfinite(entry_price) or entry_price <= 0:
            skipped += 1; continue

        i1 = i0 + 1 if i0 + 1 < len(g) else None
        day1_date = pd.to_datetime(g.loc[i1, "date"]) if i1 is not None else None
        day1_price = float(g.loc[i1, "close"]) if i1 is not None else None

        ret = float(r.ret)
        exit_price = entry_price * (1.0 + ret)
        base_return = ret
        if day1_price and day1_price > 0 and np.isfinite(exit_price):
            pyramid_return = (exit_price - day1_price) / day1_price
            pyramid_triggered = day1_price > entry_price * (1.0 + PYRAMID_THRESHOLD)
        else:
            pyramid_return = 0.0
            pyramid_triggered = False

        i_exit = min(i0 + HOLD_DAYS, len(g) - 1)
        exit_date = pd.to_datetime(g.loc[i_exit, "date"])

        paths.append(EventPath(
            ticker=tic,
            entry_date=pd.to_datetime(entry_date),
            day1_date=day1_date,
            exit_date=exit_date,
            entry_price=entry_price,
            day1_price=day1_price,
            base_return=base_return,
            pyramid_return=pyramid_return,
            pyramid_triggered=bool(pyramid_triggered),
            macd_bullish=bool(getattr(r, "macd_bullish", False) is True),
        ))
    logger.info("Built %d event paths (skipped %d with missing prices)", len(paths), skipped)
    return paths


# --- portfolio simulation ------------------------------------------------

@dataclass
class OpenPosition:
    event_id: int
    entry_date: pd.Timestamp
    day1_date: Optional[pd.Timestamp]
    exit_date: pd.Timestamp
    base_notional: float
    pyramid_notional: float
    base_return: float
    pyramid_return: float
    pyramid_triggered: bool
    pyramid_applied: bool = False


@dataclass
class SimResult:
    equity_curve: pd.DataFrame
    summary: Dict[str, float]


def simulate(paths: List[EventPath],
             calendar: pd.DatetimeIndex,
             allow_pyramid: bool,
             label: str) -> SimResult:
    """Walk the trading calendar, opening / pyramiding / closing positions.

    Invariant: equity = cash + sum(open positions notional at entry prices).
    (We don't mark open positions to market daily — an approximation that
    is consistent with how the existing compounded-return curves value
    trades only at exit. MDD is therefore a lower bound.)
    """
    entries_by_date: Dict[pd.Timestamp, List[int]] = {}
    day1_by_date: Dict[pd.Timestamp, List[int]] = {}
    exits_by_date: Dict[pd.Timestamp, List[int]] = {}
    for i, ep in enumerate(paths):
        entries_by_date.setdefault(ep.entry_date, []).append(i)
        if ep.day1_date is not None:
            day1_by_date.setdefault(ep.day1_date, []).append(i)
        exits_by_date.setdefault(ep.exit_date, []).append(i)

    cash = 1.0
    open_positions: Dict[int, OpenPosition] = {}
    eq_rows: List[Dict] = []
    skipped_base = 0
    skipped_pyramid = 0
    applied_pyramid = 0
    taken_base = 0
    concurrent_counts: List[int] = []

    def gross_notional() -> float:
        return sum(p.base_notional + p.pyramid_notional for p in open_positions.values())

    for d in calendar:
        # 1) exits first — frees capital for same-day entries
        for i in exits_by_date.get(d, []):
            p = open_positions.pop(i, None)
            if p is None:
                continue
            pnl = p.base_notional * p.base_return + p.pyramid_notional * p.pyramid_return
            cash += p.base_notional + p.pyramid_notional + pnl

        # 2) pyramid check (day +1) before new entries so the cap is fair
        if allow_pyramid:
            for i in day1_by_date.get(d, []):
                p = open_positions.get(i)
                if p is None or p.pyramid_applied or not p.pyramid_triggered:
                    continue
                equity = cash + gross_notional()
                add = PYRAMID_WEIGHT * equity
                if gross_notional() + add <= GROSS_CAP * equity + 1e-12:
                    p.pyramid_notional = add
                    p.pyramid_applied = True
                    cash -= add
                    applied_pyramid += 1
                else:
                    skipped_pyramid += 1

        # 3) new entries
        for i in entries_by_date.get(d, []):
            equity = cash + gross_notional()
            size = BASE_WEIGHT * equity
            if gross_notional() + size <= GROSS_CAP * equity + 1e-12:
                open_positions[i] = OpenPosition(
                    event_id=i,
                    entry_date=paths[i].entry_date,
                    day1_date=paths[i].day1_date,
                    exit_date=paths[i].exit_date,
                    base_notional=size,
                    pyramid_notional=0.0,
                    base_return=paths[i].base_return,
                    pyramid_return=paths[i].pyramid_return,
                    pyramid_triggered=paths[i].pyramid_triggered,
                )
                cash -= size
                taken_base += 1
            else:
                skipped_base += 1

        # 4) log
        equity_eod = cash + gross_notional()
        eq_rows.append({
            "date": d,
            "equity": equity_eod,
            "cash": cash,
            "gross": gross_notional(),
            "n_open": len(open_positions),
        })
        concurrent_counts.append(len(open_positions))

    ec = pd.DataFrame(eq_rows).sort_values("date").reset_index(drop=True)
    years = (ec["date"].iloc[-1] - ec["date"].iloc[0]).days / 365.25 if len(ec) > 1 else np.nan
    total_ret = float(ec["equity"].iloc[-1] - 1.0) if not ec.empty else np.nan
    cagr = ((1.0 + total_ret) ** (1.0 / years) - 1.0) if (years and years > 0 and total_ret > -1) else np.nan
    rm = ec["equity"].cummax()
    mdd = float((ec["equity"] / rm - 1.0).min()) if not ec.empty else np.nan
    car_mdd = cagr / abs(mdd) if (mdd and mdd < 0 and cagr == cagr) else np.nan

    summary = {
        "variant": label,
        "n_events_eligible": len(paths),
        "n_base_taken": taken_base,
        "n_base_skipped_cap": skipped_base,
        "n_pyramid_applied": applied_pyramid,
        "n_pyramid_skipped_cap": skipped_pyramid,
        "skip_rate": skipped_base / len(paths) if paths else np.nan,
        "avg_concurrent_open": float(np.mean(concurrent_counts)) if concurrent_counts else 0.0,
        "max_concurrent_open": int(np.max(concurrent_counts)) if concurrent_counts else 0,
        "final_equity": float(ec["equity"].iloc[-1]) if not ec.empty else np.nan,
        "total_return": total_ret,
        "cagr": cagr,
        "max_drawdown": mdd,
        "car_mdd": car_mdd,
    }
    return SimResult(equity_curve=ec, summary=summary)


# --- entry point ---------------------------------------------------------

def run(returns_csv: str = "data/strategy_returns_regime_hold_cut_sp100.csv",
        prices_csv:  str = "data/price_data_sp100.csv",
        out_dir:     str = "data") -> pd.DataFrame:
    logger.info("Loading %s", returns_csv)
    df = pd.read_csv(returns_csv)
    df["eps_date"] = pd.to_datetime(df["eps_date"]).dt.normalize()
    df = df.sort_values("eps_date").reset_index(drop=True)
    logger.info("  %d events, %d tickers", len(df), df["ticker"].nunique())

    logger.info("Loading prices from %s", prices_csv)
    prices = pd.read_csv(prices_csv)
    by_tic = _prepare_prices(prices)
    calendar = _trading_calendar(by_tic)
    logger.info("  calendar: %s → %s (%d trading days)",
                calendar.min().date(), calendar.max().date(), len(calendar))

    paths_all = _build_event_paths(df, by_tic, calendar)
    paths_macd = [p for p in paths_all if p.macd_bullish]
    logger.info("  all-events paths: %d | macd-bullish paths: %d",
                len(paths_all), len(paths_macd))

    variants = [
        ("unfiltered_flat",      paths_all,  False),
        ("unfiltered_pyramid",   paths_all,  True),
        ("macd_bullish_flat",    paths_macd, False),
        ("macd_bullish_pyramid", paths_macd, True),
    ]

    summaries: List[Dict] = []
    for label, paths, pyr in variants:
        logger.info("Simulating %s (pyramid=%s, n_events=%d)", label, pyr, len(paths))
        result = simulate(paths, calendar, allow_pyramid=pyr, label=label)
        eq_path = os.path.join(out_dir, f"equity_curve_sized_{label}.csv")
        result.equity_curve.to_csv(eq_path, index=False)
        summaries.append(result.summary)
        s = result.summary
        logger.info("  %s: final=%.3f cagr=%+.2f%% mdd=%+.2f%% car/mdd=%.2f skip_rate=%.1f%% avg_open=%.1f",
                    label, s["final_equity"], (s["cagr"] or 0)*100, (s["max_drawdown"] or 0)*100,
                    s["car_mdd"] or 0, (s["skip_rate"] or 0)*100, s["avg_concurrent_open"])

    summary = pd.DataFrame(summaries)
    summary_path = os.path.join(out_dir, "position_sizing_summary.csv")
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
