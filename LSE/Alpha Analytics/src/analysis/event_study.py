"""Event-study windows around EPS announcements."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

BEATS_COLOR = "forestgreen"
MISSES_COLOR = "red"


def _pct_fmt(x, _pos):
    return f"{x:.2%}"


def prepare_inputs(eps_df: pd.DataFrame, price_df: pd.DataFrame):
    """Normalise EPS and price frames so they share ticker labels and have surprise/beat columns."""
    eps_df = eps_df.copy()
    price_df = price_df.copy()

    rename_map = {"actualearningresult": "act_eps", "estimatedearning": "est_eps"}
    eps_df = eps_df.rename(columns={c: rename_map[c] for c in rename_map if c in eps_df.columns})

    if "ticker" not in eps_df.columns and "symbol" in eps_df.columns:
        eps_df = eps_df.rename(columns={"symbol": "ticker"})
    if "ticker" not in eps_df.columns:
        raise KeyError("eps_df must have a 'ticker' column.")
    for col in ("close", "ticker"):
        if col not in price_df.columns:
            raise KeyError(f"price_df missing required column: {col!r}")

    if not {"act_eps", "est_eps"}.issubset(eps_df.columns):
        raise KeyError("eps_df must include 'act_eps' and 'est_eps'.")

    if "beat" not in eps_df.columns:
        eps_df["beat"] = eps_df["act_eps"] > eps_df["est_eps"]

    eps_df["surprise"] = (
        (eps_df["act_eps"] - eps_df["est_eps"])
        .div(eps_df["est_eps"].abs())
        .where(eps_df["est_eps"].abs() != 0)
    )

    valid_tickers = sorted(set(eps_df["ticker"]).intersection(set(price_df["ticker"])))
    if not valid_tickers:
        raise RuntimeError("No overlapping tickers between eps_df and price_df.")

    eps_df = eps_df[eps_df["ticker"].isin(valid_tickers)].sort_index()
    price_df = price_df[price_df["ticker"].isin(valid_tickers)].sort_index()

    price_df["ret"] = price_df.groupby("ticker", group_keys=False)["close"].pct_change()
    return eps_df, price_df, valid_tickers


def build_event_windows(
    eps_df: pd.DataFrame,
    price_df: pd.DataFrame,
    window: int = 10,
    require_full_window: bool = True,
) -> pd.DataFrame:
    """Build a ±``window`` trading-day event panel around each EPS date."""
    eps_df, price_df, _ = prepare_inputs(eps_df, price_df)

    records = []
    for tkr, gpx in price_df.groupby("ticker"):
        idx_dt = pd.to_datetime(gpx.index)
        close = gpx["close"]
        eg = eps_df[eps_df["ticker"] == tkr].sort_index()

        for eps_date, ev in eg.iterrows():
            eps_ts = pd.Timestamp(eps_date)
            pos = idx_dt.get_indexer([eps_ts], method="nearest")[0]

            lo = pos - window
            hi = pos + window

            if require_full_window and (lo - 1 < 0 or hi >= len(idx_dt)):
                continue

            start = max(lo - 1, 0)
            stop = hi + 1

            win_close = close.iloc[start:stop]
            win_ret = win_close.pct_change().iloc[1:]

            for k, rr in enumerate(win_ret.values):
                i = start + 1 + k
                rel_day = i - pos
                if rel_day < -window or rel_day > window or pd.isna(rr):
                    continue

                row = {
                    "ticker": tkr,
                    "eps_date": eps_date,
                    "beat": bool(ev["beat"]),
                    "rel_day": int(rel_day),
                    "ret": float(rr),
                    "price_date": gpx.index[i],
                    "act_eps": ev.get("act_eps", np.nan),
                    "est_eps": ev.get("est_eps", np.nan),
                    "surprise": ev.get("surprise", np.nan),
                }
                row.update(gpx.iloc[i].to_dict())
                row["ret"] = float(rr)
                records.append(row)

    event_df = pd.DataFrame.from_records(records)
    if event_df.empty:
        raise RuntimeError(
            f"No event windows assembled for window={window}. Check ticker overlap and index alignment."
        )

    for col in ("act_eps", "est_eps", "surprise"):
        if col not in event_df.columns:
            event_df[col] = np.nan

    event_df["window"] = int(window)
    return event_df


def plot_event_panel(event_df: pd.DataFrame, window: int, title_prefix: str = ""):
    """Plot mean close-to-close returns by relative day, beats vs. misses, per ticker and pooled."""
    dfw = event_df[event_df["window"] == window].copy()
    if dfw.empty:
        print(f"[plot] No rows for window={window}.")
        return

    mean_ret = (
        dfw.groupby(["ticker", "beat", "rel_day"], as_index=False)["ret"]
        .mean()
        .sort_values(["ticker", "beat", "rel_day"])
    )

    for tkr in mean_ret["ticker"].unique():
        d = mean_ret[mean_ret["ticker"] == tkr]
        beats = d[d["beat"] == True]  # noqa: E712
        misses = d[d["beat"] == False]  # noqa: E712

        n_beats = dfw[(dfw["ticker"] == tkr) & (dfw["beat"] == True)]["eps_date"].nunique()  # noqa: E712
        n_misses = dfw[(dfw["ticker"] == tkr) & (dfw["beat"] == False)]["eps_date"].nunique()  # noqa: E712

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        fig.suptitle(
            f"{title_prefix}{tkr} — Mean close-to-close return by day around EPS (±{window} trading days)"
        )
        for ax, src, label, color, n in (
            (axes[0], beats, "Beats", BEATS_COLOR, n_beats),
            (axes[1], misses, "Misses", MISSES_COLOR, n_misses),
        ):
            ax.bar(src["rel_day"], src["ret"], alpha=0.6, color=color, label=label)
            ax.plot(src["rel_day"], src["ret"], linewidth=1.5, color=color)
            ax.axhline(0, linewidth=1)
            ax.axvline(0, linestyle="--", color="black", linewidth=1)
            ax.yaxis.set_major_formatter(FuncFormatter(_pct_fmt))
            ax.set_ylabel("Mean return")
            ax.set_title(f"{label} (n={n})")
            ax.grid(True, axis="y", linewidth=0.3)
        axes[1].set_xlabel("Relative trading day (0 = EPS day)")
        plt.tight_layout()
        plt.show()

    overall = (
        dfw.groupby(["beat", "rel_day"], as_index=False)["ret"]
        .mean()
        .sort_values(["beat", "rel_day"])
    )
    overall_beats = overall[overall["beat"] == True]  # noqa: E712
    overall_misses = overall[overall["beat"] == False]  # noqa: E712
    n_beats_all = dfw[dfw["beat"] == True][["ticker", "eps_date"]].drop_duplicates().shape[0]  # noqa: E712
    n_misses_all = dfw[dfw["beat"] == False][["ticker", "eps_date"]].drop_duplicates().shape[0]  # noqa: E712

    fig3, ax3 = plt.subplots(figsize=(12, 5))
    fig3.suptitle(f"{title_prefix}All Tickers — Beats vs Misses (±{window} trading days)")
    if not overall_beats.empty:
        ax3.plot(overall_beats["rel_day"], overall_beats["ret"], linewidth=1.8, marker="o", ms=3,
                 label=f"Beats (events={n_beats_all})", color=BEATS_COLOR)
    if not overall_misses.empty:
        ax3.plot(overall_misses["rel_day"], overall_misses["ret"], linewidth=1.8, marker="o", ms=3,
                 label=f"Misses (events={n_misses_all})", color=MISSES_COLOR)
    ax3.axhline(0, linewidth=1)
    ax3.axvline(0, linestyle="--", color="black", linewidth=1)
    ax3.yaxis.set_major_formatter(FuncFormatter(_pct_fmt))
    ax3.set_xlabel("Relative trading day (0 = EPS day)")
    ax3.set_ylabel("Mean return")
    ax3.grid(True, axis="y", linewidth=0.3)
    ax3.legend()
    plt.tight_layout()
    plt.show()
