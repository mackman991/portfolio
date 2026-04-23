"""Event-driven earnings strategies: momentum, contrarian, agnostic, pre-runup."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Required columns on the event-window DataFrame fed to these backtests.
REQUIRED_COLS = {"ticker", "eps_date", "rel_day", "beat", "act_eps", "est_eps", "surprise", "ret"}


def prepare_event_frame(eps_data: pd.DataFrame, surprise_threshold: float = 0.0) -> pd.DataFrame:
    """Validate and lightly normalise an event-window frame in place."""
    df = eps_data.copy()

    if "eps_date" in df.columns:
        df["eps_date"] = pd.to_datetime(df["eps_date"], errors="coerce")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.dropna(subset=["eps_date"]).copy()

    if "beat" in df.columns and df["beat"].dtype != bool:
        df["beat"] = (
            df["beat"].astype(str).str.strip().str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
            .fillna(False).astype(bool)
        )

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Event frame is missing required columns: {missing}")

    if surprise_threshold > 0:
        df = df.loc[df["surprise"].abs() >= surprise_threshold].copy()
    return df


def compounded_return(returns: pd.Series) -> float:
    """Compound (1+r) - 1; ignores NaNs."""
    r = pd.to_numeric(returns, errors="coerce").dropna().values
    if r.size == 0:
        return 0.0
    return float(np.prod(1 + r) - 1)


def window_compounded(df_group: pd.DataFrame, start_rel: int, end_rel: int) -> float:
    """Compounded return for a given event group between rel_day in [start_rel, end_rel]."""
    x = df_group.loc[
        (df_group["rel_day"] >= start_rel) & (df_group["rel_day"] <= end_rel), "ret"
    ]
    return compounded_return(x)


def apply_cost(gross_ret: float, cost_bps: float) -> float:
    """Apply a single round-trip cost (entry+exit) in basis points to a per-event return."""
    if cost_bps <= 0:
        return gross_ret
    cost_mult = 1.0 - (cost_bps / 10000.0)
    return (1.0 + gross_ret) * cost_mult - 1.0


def summarise_strategy(per_event: pd.DataFrame) -> pd.Series:
    """Per-strategy summary: n, mean, median, hit-rate, t-stat."""
    r = per_event["ret"].dropna()
    n = r.shape[0]
    mean = r.mean() if n else np.nan
    median = r.median() if n else np.nan
    hit = (r > 0).mean() if n else np.nan
    std = r.std(ddof=1) if n > 1 else np.nan
    se = (std / np.sqrt(n)) if (n > 1 and std > 0) else np.nan
    t_stat = (mean / se) if (se and se > 0) else np.nan
    return pd.Series(
        {"n": n, "mean": mean, "median": median, "hit_rate": hit, "std": std, "se": se, "t_stat": t_stat}
    )


# --------------------------
# Strategies
# --------------------------
def backtest_post_earnings_momentum(df: pd.DataFrame, H: int, cost_bps: float = 0.0) -> pd.DataFrame:
    """Long beats, short misses, hold +1..+H. Sign known the morning after the print."""
    out = []
    for (tic, ed), g in df.groupby(["ticker", "eps_date"]):
        gross = window_compounded(g, 1, H)
        sign = 1.0 if bool(g["beat"].iloc[0]) else -1.0
        out.append({"ticker": tic, "eps_date": ed, "ret": apply_cost(sign * gross, cost_bps)})
    return pd.DataFrame(out)


def backtest_contrarian(df: pd.DataFrame, H: int, cost_bps: float = 0.0) -> pd.DataFrame:
    """Short beats, long misses, hold +1..+H."""
    out = []
    for (tic, ed), g in df.groupby(["ticker", "eps_date"]):
        gross = window_compounded(g, 1, H)
        sign = -1.0 if bool(g["beat"].iloc[0]) else 1.0
        out.append({"ticker": tic, "eps_date": ed, "ret": apply_cost(sign * gross, cost_bps)})
    return pd.DataFrame(out)


def backtest_contrarian_agnostic(
    df: pd.DataFrame,
    H: int,
    cost_bps: float = 0.0,
    signal_priority=(0, 1, -1),
) -> pd.DataFrame:
    """Fade the announcement-day move, regardless of beat/miss."""
    out = []
    for (tic, ed), g in df.groupby(["ticker", "eps_date"]):
        sig_ret = None
        for d in signal_priority:
            r = g.loc[g["rel_day"] == d, "ret"]
            if not r.empty and pd.notna(r.iloc[0]):
                sig_ret = float(r.iloc[0])
                break
        if sig_ret is None or np.isclose(sig_ret, 0.0):
            continue

        gross = window_compounded(g, 1, H)
        sign = -1.0 if sig_ret > 0 else 1.0
        out.append({
            "ticker": tic, "eps_date": ed,
            "ret": apply_cost(sign * gross, cost_bps),
            "signal_ret": sig_ret,
        })
    return pd.DataFrame(out)


def backtest_pre_earnings_runup(df: pd.DataFrame, P: int, cost_bps: float = 0.0) -> pd.DataFrame:
    """Long the run-up window -P..-1 going into the print.

    NOTE: implementation-only. The naive variant (no surprise filter) is tradable;
    any subset filtered by ``surprise`` magnitude embeds **look-ahead bias** because
    the surprise is only knowable after the announcement. See README for caveat.
    """
    out = []
    for (tic, ed), g in df.groupby(["ticker", "eps_date"]):
        gross = window_compounded(g, -P, -1)
        out.append({"ticker": tic, "eps_date": ed, "ret": apply_cost(gross, cost_bps)})
    return pd.DataFrame(out)


def equity_curve(per_event: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight equity curve across events on each EPS date."""
    daily = (
        per_event.dropna(subset=["eps_date", "ret"])
        .groupby("eps_date", as_index=False)["ret"].mean()
        .sort_values("eps_date")
    )
    daily["equity"] = (1.0 + daily["ret"]).cumprod()
    return daily
