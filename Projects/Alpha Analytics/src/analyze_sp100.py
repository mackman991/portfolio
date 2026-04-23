"""Full SP100 event-study analysis: build event windows, run all strategies, summarise.

Run (from repo root, after pull_sp100 completes):
    python -m src.analyze_sp100
"""
from __future__ import annotations

import logging
import os

import pandas as pd

from src import config
from src.analysis.event_study import build_event_windows
from src.strategies.earnings_strategies import (
    backtest_contrarian,
    backtest_contrarian_agnostic,
    backtest_post_earnings_momentum,
    backtest_pre_earnings_runup,
    equity_curve,
    prepare_event_frame,
    summarise_strategy,
)
from src.strategies.pre_earnings import PreEarningsStrategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = config.DATA_DIR
WINDOW = 15
COST_BPS = 5.0  # one-way 2.5 bps is realistic for liquid large-caps


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = pd.read_csv(os.path.join(DATA_DIR, "price_data_sp100.csv"))
    prices["date"] = pd.to_datetime(prices["date"], utc=True).dt.tz_localize(None)
    prices = prices.set_index("date")

    eps = pd.read_csv(os.path.join(DATA_DIR, "eps_sp100.csv"))
    eps["date"] = pd.to_datetime(eps["date"], utc=True).dt.tz_localize(None)
    eps = eps.set_index("date")
    log.info("Prices: %d rows, %d tickers", len(prices), prices["ticker"].nunique())
    log.info("EPS:    %d rows, %d tickers", len(eps), eps["ticker"].nunique())
    return prices, eps


def build_windows(prices: pd.DataFrame, eps: pd.DataFrame) -> pd.DataFrame:
    out_path = os.path.join(DATA_DIR, f"eps_event_windows_{WINDOW}_sp100.csv")
    if os.path.exists(out_path):
        log.info("Loading cached event windows from %s", out_path)
        return pd.read_csv(out_path, parse_dates=["eps_date", "price_date"])
    log.info("Building ±%d event windows for SP100 …", WINDOW)
    ev = build_event_windows(eps, prices, window=WINDOW, require_full_window=True)
    ev.to_csv(out_path, index=False)
    log.info("Saved event windows → %s  (%d rows, %d events)",
             out_path, len(ev), ev[["ticker", "eps_date"]].drop_duplicates().shape[0])
    return ev


def run_strategies(ev: pd.DataFrame) -> pd.DataFrame:
    df = prepare_event_frame(ev)
    n_events = df[["ticker", "eps_date"]].drop_duplicates().shape[0]
    n_tickers = df["ticker"].nunique()
    beat_pct = df[["ticker", "eps_date", "beat"]].drop_duplicates()["beat"].mean()
    log.info("Event frame: %d events, %d tickers, %.1f%% beats", n_events, n_tickers, 100 * beat_pct)

    strategies = {
        "Momentum +1..+6":     backtest_post_earnings_momentum(df, H=6,  cost_bps=COST_BPS),
        "Momentum +1..+3":     backtest_post_earnings_momentum(df, H=3,  cost_bps=COST_BPS),
        "Contrarian +1..+1":   backtest_contrarian(df,            H=1,  cost_bps=COST_BPS),
        "Contrarian +1..+3":   backtest_contrarian(df,            H=3,  cost_bps=COST_BPS),
        "Agnostic +1..+1":     backtest_contrarian_agnostic(df,   H=1,  cost_bps=COST_BPS),
        "Pre Run-up -10..-1":  backtest_pre_earnings_runup(df,    P=10, cost_bps=COST_BPS),
        "Pre Run-up -5..-1":   backtest_pre_earnings_runup(df,    P=5,  cost_bps=COST_BPS),
    }

    rows = []
    for name, per_event in strategies.items():
        s = summarise_strategy(per_event)
        s.name = name
        rows.append(s)
        log.info("%-28s  n=%d  mean=%+.2f%%  hit=%.1f%%  t=%.2f",
                 name, s["n"], 100 * s["mean"], 100 * s["hit_rate"], s["t_stat"])

    # Pre-earnings Hold/Cut variants
    strat = PreEarningsStrategy(ev)
    for thr, ext in [(0.05, 5), (0.05, 10), (0.10, 5)]:
        pe = strat.calculate_with_post_event_cut(
            entry_day=-10, base_exit_day=0, extended_exit_day=ext, surprise_threshold=thr
        )
        name = f"Hold/Cut thr={thr:.0%} ext=+{ext}"
        pe = pe.rename(columns={"return": "ret"})
        pe["ret"] = pe["ret"].apply(lambda r: (1 + r) * (1 - COST_BPS / 10000) - 1)
        s = summarise_strategy(pe)
        s.name = name
        rows.append(s)
        log.info("%-28s  n=%d  mean=%+.2f%%  hit=%.1f%%  t=%.2f",
                 name, s["n"], 100 * s["mean"], 100 * s["hit_rate"], s["t_stat"])

    summary = pd.DataFrame(rows)
    summary["mean"] = summary["mean"].map("{:.2%}".format)
    summary["median"] = summary["median"].map("{:.2%}".format)
    summary["hit_rate"] = summary["hit_rate"].map("{:.1%}".format)
    summary["std"] = summary["std"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    summary["se"] = summary["se"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    summary["t_stat"] = summary["t_stat"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")

    out = os.path.join(DATA_DIR, "strategy_summary_sp100.csv")
    summary.to_csv(out)
    log.info("Saved strategy summary → %s", out)
    return summary


def beat_miss_breakdown(ev: pd.DataFrame) -> pd.DataFrame:
    """Beat/miss rate and mean day-0 return by ticker."""
    df = prepare_event_frame(ev)
    events = df[df["rel_day"] == 0].copy()
    events["surprise_pct"] = events["surprise"] * 100

    by_ticker = (
        events.groupby("ticker")
        .agg(
            n_events=("eps_date", "count"),
            beat_rate=("beat", "mean"),
            mean_surprise_pct=("surprise_pct", "mean"),
            mean_day0_ret=("ret", "mean"),
        )
        .reset_index()
    )
    by_ticker["mean_day0_ret"] = by_ticker["mean_day0_ret"].map("{:.2%}".format)
    by_ticker["mean_surprise_pct"] = by_ticker["mean_surprise_pct"].map("{:.2f}%".format)
    by_ticker["beat_rate"] = by_ticker["beat_rate"].map("{:.1%}".format)

    out = os.path.join(DATA_DIR, "beat_miss_sp100.csv")
    by_ticker.to_csv(out, index=False)
    log.info("Saved beat/miss breakdown → %s", out)
    return by_ticker


def sector_surprise_summary(ev: pd.DataFrame) -> pd.DataFrame:
    """Aggregate post-EPS 1-day and 5-day returns by beat/miss across the full universe."""
    df = prepare_event_frame(ev)
    events_day0 = df[df["rel_day"] == 0][["ticker", "eps_date", "beat", "surprise"]].copy()

    # compute 1-day and 5-day compounded returns post-EPS
    rows = []
    for (tic, ed), g in df.groupby(["ticker", "eps_date"]):
        r1 = g.loc[g["rel_day"] == 1, "ret"]
        r5 = g.loc[(g["rel_day"] >= 1) & (g["rel_day"] <= 5), "ret"]
        beat = bool(g["beat"].iloc[0])
        surprise = float(g["surprise"].iloc[0]) if "surprise" in g.columns else float("nan")
        rows.append({
            "ticker": tic, "eps_date": ed, "beat": beat, "surprise": surprise,
            "ret_1d": float(r1.iloc[0]) if not r1.empty else float("nan"),
            "ret_5d": float((1 + r5).prod() - 1) if not r5.empty else float("nan"),
        })

    rdf = pd.DataFrame(rows)
    agg = (
        rdf.groupby("beat")
        .agg(n=("ret_1d", "count"), mean_1d=("ret_1d", "mean"), mean_5d=("ret_5d", "mean"))
        .reset_index()
    )
    agg["beat"] = agg["beat"].map({True: "Beat", False: "Miss"})
    agg["mean_1d"] = agg["mean_1d"].map("{:.2%}".format)
    agg["mean_5d"] = agg["mean_5d"].map("{:.2%}".format)

    out = os.path.join(DATA_DIR, "beat_miss_returns_sp100.csv")
    rdf.to_csv(out, index=False)
    log.info("Saved beat/miss returns detail → %s", out)
    return agg


def equity_curves(ev: pd.DataFrame) -> None:
    """Save equity curve CSVs for best strategies."""
    df = prepare_event_frame(ev)
    curves = {
        "momentum_h6":  backtest_post_earnings_momentum(df, H=6,  cost_bps=COST_BPS),
        "pre_runup_p10": backtest_pre_earnings_runup(df, P=10, cost_bps=COST_BPS),
    }
    for name, per_event in curves.items():
        ec = equity_curve(per_event)
        out = os.path.join(DATA_DIR, f"equity_curve_{name}_sp100.csv")
        ec.to_csv(out, index=False)
        log.info("Saved equity curve (%s) → %s  (final equity=%.3f)", name, out, ec["equity"].iloc[-1])


def main() -> None:
    prices, eps = load_data()
    ev = build_windows(prices, eps)

    log.info("=== Beat/Miss Breakdown ===")
    bm = beat_miss_breakdown(ev)
    print(bm.to_string(index=False))

    log.info("=== Post-EPS Return Summary ===")
    ret_summary = sector_surprise_summary(ev)
    print(ret_summary.to_string(index=False))

    log.info("=== Strategy Backtests ===")
    summary = run_strategies(ev)
    print(summary.to_string())

    equity_curves(ev)

    log.info("Done. All outputs in %s/", DATA_DIR)


if __name__ == "__main__":
    main()
