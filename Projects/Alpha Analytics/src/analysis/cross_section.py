"""Cross-section analysis: post-EPS returns × QoQ growth × correlations."""
from __future__ import annotations

import warnings
from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")


def _first_present_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _ensure_datetime(df: pd.DataFrame, col: Optional[str]) -> None:
    if col and col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")


def calculate_event_returns(df: pd.DataFrame) -> pd.DataFrame:
    """1, 3, 5, 10-day post-EPS returns for each (ticker, eps_date) event."""
    df = df.copy()

    ticker_col = "ticker"
    eps_col = "eps_date"
    rel_col = _first_present_column(df, ["rel_day", "relative_day", "relativeday"])
    date_col = _first_present_column(df, ["date", "trading_date", "price_date"])
    price_col = _first_present_column(df, ["adj_close", "adjclose", "adjusted_close", "close", "price"])
    act_eps_col = _first_present_column(df, ["act_eps", "actual_eps", "actual"])
    est_eps_col = _first_present_column(df, ["est_eps", "estimate_eps", "estimate"])
    surprise_col = _first_present_column(df, ["eps_surprise", "surprise"])
    beat_col = _first_present_column(df, ["beat", "beat_flag"])

    for rc in (ticker_col, eps_col, rel_col):
        if rc is None or rc not in df.columns:
            raise ValueError("event window data must include 'ticker', 'eps_date' and 'rel_day'.")

    if price_col is None:
        ret_col = _first_present_column(df, ["ret", "return", "daily_return"])
        if ret_col is None:
            raise ValueError("Need a price column or a daily-return column.")
        use_returns_only = True
    else:
        use_returns_only = False

    _ensure_datetime(df, eps_col)
    _ensure_datetime(df, date_col)
    df = df.sort_values([ticker_col, eps_col, rel_col])

    results = []
    for (tkr, edate), g in df.groupby([ticker_col, eps_col], dropna=False):
        g0 = g[g[rel_col] == 0]
        if g0.empty:
            continue
        base_price = None if use_returns_only else g0[price_col].iloc[0]
        info_row = g0.iloc[0]

        rec = {ticker_col: tkr, eps_col: edate}
        if act_eps_col:
            rec["act_eps"] = info_row.get(act_eps_col, np.nan)
        if est_eps_col:
            rec["est_eps"] = info_row.get(est_eps_col, np.nan)
        if surprise_col:
            rec["eps_surprise"] = info_row.get(surprise_col, np.nan)
        if beat_col:
            rec["beat"] = info_row.get(beat_col, np.nan)

        for days in (1, 3, 5, 10):
            gd = g[g[rel_col] == days]
            if not gd.empty and not use_returns_only and pd.notna(base_price):
                pxN = gd[price_col].iloc[0]
                ret_n = (pxN / base_price) - 1.0 if pd.notna(pxN) else np.nan
            elif not gd.empty:
                sub = g[(g[rel_col] >= 1) & (g[rel_col] <= days)]
                rcol = _first_present_column(sub, ["ret", "return", "daily_return"])
                if rcol is None:
                    ret_n = np.nan
                else:
                    vals = pd.to_numeric(sub[rcol], errors="coerce").dropna()
                    ret_n = float(np.prod(1.0 + vals) - 1.0) if not vals.empty else np.nan
            else:
                ret_n = np.nan
            rec[f"return_{days}d"] = ret_n

        results.append(rec)

    return pd.DataFrame(results)


def calculate_qoq_growth(df: pd.DataFrame) -> pd.DataFrame:
    """Quarter-on-quarter growth per (ticker, concept)."""
    df = df.copy()

    ticker_col = "ticker"
    concept_col = "concept"
    fy_col = _first_present_column(df, ["fy", "fy_fiscal", "fiscal_year", "year"])
    fq_col = _first_present_column(df, ["fq", "fiscal_quarter", "quarter"])
    val_col = _first_present_column(df, ["value", "amount", "reported_value", "val"])
    date_col = _first_present_column(df, ["date", "period_end", "report_date", "end_date", "fyq_date"])
    release_col = _first_present_column(df, ["earnings_release", "eps_release_date"])
    fpl_col = _first_present_column(df, ["fiscal_period_label", "period_label"])

    for rc, nm in (
        (ticker_col, "ticker"),
        (concept_col, "concept"),
        (fy_col, "fy"),
        (fq_col, "fq"),
        (val_col, "value"),
    ):
        if rc is None or rc not in df.columns:
            raise ValueError(f"SEC frame must include '{nm}' (or a close variant).")

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if release_col:
        df[release_col] = pd.to_datetime(df[release_col], errors="coerce")

    df[fy_col] = pd.to_numeric(df[fy_col], errors="coerce").astype("Int64")
    df[fq_col] = pd.to_numeric(df[fq_col], errors="coerce").astype("Int64")
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")

    df = df.sort_values([ticker_col, concept_col, fy_col, fq_col])

    pieces = []
    for (_tkr, _cpt), g in df.groupby([ticker_col, concept_col], dropna=False):
        g = g.sort_values([fy_col, fq_col]).copy()
        g["prev_val"] = g[val_col].shift(1)
        g["prev_fy"] = g[fy_col].shift(1)
        g["prev_fq"] = g[fq_col].shift(1)

        same_year_next_q = (g[fy_col] == g["prev_fy"]) & (g[fq_col] == g["prev_fq"] + 1)
        year_roll_next_q = (
            (g[fy_col] == g["prev_fy"] + 1) & (g[fq_col] == 1) & (g["prev_fq"] == 4)
        )
        next_q_mask = (same_year_next_q | year_roll_next_q).fillna(False)

        denom_ok = g["prev_val"].notna() & (g["prev_val"] != 0) & g[val_col].notna()
        sub = g.loc[next_q_mask & denom_ok].copy()
        if sub.empty:
            continue

        sub["qoq_growth"] = (sub[val_col] - sub["prev_val"]) / sub["prev_val"].abs()

        keep_cols = [ticker_col, concept_col, fy_col, fq_col, val_col, "prev_val", "qoq_growth"]
        if date_col:
            keep_cols.append(date_col)
        if release_col:
            keep_cols.append(release_col)
        if fpl_col:
            keep_cols.append(fpl_col)

        sub = sub[keep_cols].rename(
            columns={fy_col: "fy", fq_col: "fq", val_col: "value", "prev_val": "previous_value"}
        )
        pieces.append(sub)

    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def merge_returns_and_growth(
    returns_df: pd.DataFrame,
    growth_df: pd.DataFrame,
    max_days: int = 30,
) -> pd.DataFrame:
    """Match each EPS event to the nearest earnings release within ``max_days``."""
    returns_df = returns_df.copy()
    growth_df = growth_df.copy()

    _ensure_datetime(returns_df, "eps_date")
    _ensure_datetime(growth_df, "earnings_release")

    merged_rows = []
    for _, r in returns_df.iterrows():
        tkr = r["ticker"]
        ed = r["eps_date"]
        if pd.isna(ed):
            continue
        tg = growth_df[growth_df["ticker"] == tkr].copy()
        if tg.empty or "earnings_release" not in tg.columns:
            continue
        tg["date_diff"] = (tg["earnings_release"] - ed).abs().dt.days
        tg = tg[tg["date_diff"] <= max_days]
        if tg.empty:
            continue
        closest_date = tg.loc[tg["date_diff"].idxmin(), "earnings_release"]
        same_release = tg[tg["earnings_release"] == closest_date]

        growth_pivot = same_release.pivot_table(
            index=["ticker", "earnings_release"],
            columns="concept",
            values="qoq_growth",
            aggfunc="first",
        ).reset_index()

        for _, g_row in growth_pivot.iterrows():
            row = r.to_dict()
            row["earnings_release"] = g_row["earnings_release"]
            row["date_diff"] = abs((g_row["earnings_release"] - ed).days)
            for concept in growth_pivot.columns[2:]:
                row[f"growth_{concept}"] = g_row[concept]
            merged_rows.append(row)

    return pd.DataFrame(merged_rows)


def analyze_correlations(
    df: pd.DataFrame,
    plot: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Correlation between post-EPS returns and concept growth (overall + per-ticker)."""
    return_cols = [c for c in df.columns if c.startswith("return_")]
    growth_cols = [c for c in df.columns if c.startswith("growth_")]
    analysis_cols = return_cols + growth_cols

    df_clean = df[analysis_cols + ["ticker"]].copy()
    min_non_na = max(1, len(analysis_cols) // 2)
    df_clean = df_clean.dropna(thresh=min_non_na)

    corr_matrix = None
    if len(df_clean) > 10 and len(analysis_cols) >= 2:
        corr_matrix = df_clean[analysis_cols].corr()

        if plot:
            plt.figure(figsize=(15, 12))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, cmap="RdBu_r", center=0, annot=False,
                        square=True, cbar_kws={"shrink": 0.8})
            plt.title("Overall Correlation Matrix: Returns vs Growth Metrics")
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.show()

            if return_cols and growth_cols:
                sub = corr_matrix.loc[return_cols, growth_cols]
                plt.figure(figsize=(20, 6))
                sns.heatmap(sub, annot=True, cmap="RdBu_r", center=0, fmt=".2f",
                            cbar_kws={"shrink": 0.8})
                plt.title("Returns vs Growth Correlations")
                plt.xticks(rotation=45, ha="right")
                plt.yticks(rotation=0)
                plt.tight_layout()
                plt.show()

    return df_clean, corr_matrix
