"""SEC XBRL fact cleaner with derived true-Q4 logic.

Companies report Q1/Q2/Q3 separately but bundle Q4 into the FY 10-K. To
get a clean quarterly series we derive Q4 = FY - (Q1 + Q2 + Q3) when no
true-quarterly Q4 row exists, then drop the FY/YTD rows so each
(ticker, concept, fy, fq) is represented once.

Fiscal-year ends differ per ticker (e.g. AAPL=Sep, GOOG=Dec, NVDA=Jan)
so calendar-month -> fiscal-quarter must be done per-ticker.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


# ---------------------------- Helpers ----------------------------
def _to_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)


def _fiscal_year(end_month: int, d: pd.Timestamp) -> Optional[int]:
    if pd.isna(d):
        return None
    return d.year if d.month <= end_month else d.year + 1


def _fiscal_quarter(end_month: int, d: pd.Timestamp) -> Optional[int]:
    if pd.isna(d):
        return None
    start_month = 1 if end_month == 12 else end_month + 1
    idx = (d.month - start_month) % 12
    return int(idx // 3) + 1


def _require_cols(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _make_fq_label(fq: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=fq.index, dtype="object")
    mask = fq.notna()
    out.loc[mask] = "Q" + fq[mask].astype("Int64").astype(str)
    return out


def _make_fiscal_period_label(fq_label: pd.Series, fy: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=fq_label.index, dtype="object")
    mask = fq_label.notna() & fy.notna()
    fy_str = fy.astype("Int64").astype(str)
    out.loc[mask] = fq_label[mask] + " FY" + fy_str[mask]
    return out


def _drop_fy_rows_df(
    df: pd.DataFrame,
    *,
    duration_threshold: int = 300,
    rule: str = "both",
) -> pd.DataFrame:
    """Remove FY/YTD rows but preserve derived true-Q4 rows."""
    d = df.copy()

    d["fp_norm"] = d["fp"].astype(str).str.strip().str.upper() if "fp" in d.columns else pd.NA

    if "duration_days" not in d.columns:
        if {"date", "start"} <= set(d.columns):
            d["date"] = _to_utc(d["date"])
            d["start"] = _to_utc(d["start"])
            d["duration_days"] = (d["date"] - d["start"]).dt.days
        else:
            d["duration_days"] = np.nan

    fy_by_fp = d["fp_norm"].eq("FY")
    fy_by_duration = d["duration_days"].ge(duration_threshold)

    if rule == "fp":
        fy_mask = fy_by_fp
    elif rule == "duration":
        fy_mask = fy_by_duration
    else:
        fy_mask = fy_by_fp | fy_by_duration

    if "is_derived_q4" in d.columns:
        keep_true_q4 = d["is_derived_q4"].fillna(False).astype(bool)
        fy_mask = fy_mask & ~keep_true_q4

    out = d.loc[~fy_mask].copy()
    if "fp_norm" in out.columns:
        out.drop(columns=["fp_norm"], inplace=True)
    return out


def _finalise_dates_to_ymd(
    df: pd.DataFrame,
    cols=("date", "start", "filed", "accepted", "report_date", "earnings_release"),
) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            ser = pd.to_datetime(out[c], errors="coerce")
            out[c] = ser.dt.strftime("%Y-%m-%d")
            out.loc[ser.isna(), c] = pd.NA
    return out


# ---------------------------- Core ----------------------------
DEFAULT_FISC_ENDS: Dict[str, int] = {"AAPL": 9, "GOOG": 12, "NVDA": 1}


def clean_sec_facts_df(
    df: pd.DataFrame,
    fisc_end_map: Dict[str, int] = DEFAULT_FISC_ENDS,
    group_keys: Iterable[str] = ("ticker", "concept", "date", "unit"),
    add_true_q4: bool = True,
    drop_fy_rows: bool = True,
    fy_drop_rule: str = "both",
    fy_duration_threshold: int = 300,
) -> pd.DataFrame:
    """Clean SEC facts, derive true Q4 where missing, drop FY rows.

    Returns a DataFrame with one row per (ticker, concept, fy, fq, unit)
    plus calendar/fiscal labels and an ``is_derived_q4`` flag.
    """
    _require_cols(df, ["ticker", "concept", "unit", "value", "date", "start", "filed"])

    orig = df.copy()
    orig["value"] = pd.to_numeric(orig["value"], errors="coerce")
    for c in ["date", "start", "filed", "accepted", "report_date", "earnings_release"]:
        if c in orig.columns:
            orig[c] = _to_utc(orig[c])
    orig["duration_days"] = (orig["date"] - orig["start"]).dt.days

    # Pick the shortest-duration (i.e. quarterly-most) record per group, breaking ties on filed/accn.
    dedup = orig.copy()
    dedup["_rank_duration"] = np.where(
        dedup["duration_days"].isna() | (dedup["duration_days"] <= 0),
        np.inf,
        dedup["duration_days"],
    )
    sort_keys = [*group_keys, "_rank_duration", "filed"]
    ascending = [True] * len(group_keys) + [True, False]
    if "accn" in dedup.columns:
        sort_keys.append("accn")
        ascending.append(False)

    cleaned = (
        dedup.sort_values(by=sort_keys, ascending=ascending)
        .groupby(list(group_keys), as_index=False)
        .nth(0)
        .reset_index(drop=True)
    )

    cleaned["fiscal_end_month"] = cleaned["ticker"].map(fisc_end_map)
    cleaned["fy_fiscal"] = [
        _fiscal_year(m, t) if pd.notna(m) else None
        for m, t in zip(cleaned["fiscal_end_month"], cleaned["date"])
    ]
    cleaned["fq"] = [
        _fiscal_quarter(m, t) if pd.notna(m) else None
        for m, t in zip(cleaned["fiscal_end_month"], cleaned["date"])
    ]
    cleaned["fq_label"] = _make_fq_label(cleaned["fq"])
    cleaned["fiscal_period_label"] = _make_fiscal_period_label(cleaned["fq_label"], cleaned["fy_fiscal"])
    cleaned["cal_year"] = cleaned["date"].dt.year.astype("Int64")
    cleaned["cal_quarter"] = (((cleaned["date"].dt.month - 1) // 3) + 1).astype("Int64")
    cleaned.drop(columns=["_rank_duration"], inplace=True, errors="ignore")

    if add_true_q4:
        cleaned = _derive_true_q4(orig, cleaned, fisc_end_map)

    if drop_fy_rows:
        cleaned = _drop_fy_rows_df(
            cleaned,
            duration_threshold=fy_duration_threshold,
            rule=fy_drop_rule,
        )

    return _finalise_dates_to_ymd(cleaned)


def _derive_true_q4(
    orig: pd.DataFrame,
    cleaned: pd.DataFrame,
    fisc_end_map: Dict[str, int],
) -> pd.DataFrame:
    """Synthesise Q4 rows as FY - (Q1+Q2+Q3) where no true quarterly Q4 exists."""
    orig_tmp = orig.copy()
    orig_tmp["fp_norm"] = (
        orig_tmp["fp"].astype(str).str.strip().str.upper() if "fp" in orig_tmp.columns else pd.NA
    )
    orig_tmp["fiscal_end_month"] = orig_tmp["ticker"].map(fisc_end_map)
    orig_tmp["fy_fiscal"] = [
        _fiscal_year(m, t) if pd.notna(m) else None
        for m, t in zip(orig_tmp["fiscal_end_month"], orig_tmp["date"])
    ]
    orig_tmp["fq"] = [
        _fiscal_quarter(m, t) if pd.notna(m) else None
        for m, t in zip(orig_tmp["fiscal_end_month"], orig_tmp["date"])
    ]

    # Best earnings_release per (ticker, fy) — prefer the Q4/FY release.
    er_cand = orig_tmp[orig_tmp["earnings_release"].notna()].copy()
    er_q4fy = er_cand[(er_cand["fq"] == 4) | (er_cand["fp_norm"].isin(["Q4", "FY"]))]
    er_source = er_q4fy if not er_q4fy.empty else er_cand
    er_best = (
        er_source.sort_values(["ticker", "fy_fiscal", "earnings_release"])
        .groupby(["ticker", "fy_fiscal"], as_index=False)
        .agg(earnings_release=("earnings_release", "max"))
    )

    fy_cand = orig_tmp[
        (orig_tmp["duration_days"] >= 300) | (orig_tmp["fp_norm"] == "FY")
    ].dropna(subset=["fy_fiscal"])
    fy_best = (
        fy_cand.sort_values(
            by=["ticker", "concept", "fy_fiscal", "unit", "duration_days", "filed"],
            ascending=[True, True, True, True, False, False],
        )
        .groupby(["ticker", "concept", "fy_fiscal", "unit"], as_index=False)
        .nth(0)
        .rename(columns={"value": "FY_value", "date": "FY_date", "filed": "FY_filed", "accn": "FY_accn"})
    )[["ticker", "concept", "fy_fiscal", "unit", "FY_value", "FY_date", "FY_filed", "FY_accn"]]

    q123 = (
        cleaned[cleaned["fq"].isin([1, 2, 3])]
        .groupby(["ticker", "concept", "fy_fiscal", "unit", "fq"], as_index=False)["value"]
        .sum()
    )
    qsum = (
        q123.pivot_table(
            index=["ticker", "concept", "fy_fiscal", "unit"],
            columns="fq",
            values="value",
            aggfunc="sum",
        )
        .rename(columns={1: "q1", 2: "q2", 3: "q3"})
        .reset_index()
    )

    cleaned_tmp = cleaned.copy()
    cleaned_tmp["fp_norm"] = (
        cleaned_tmp["fp"].astype(str).str.strip().str.upper() if "fp" in cleaned_tmp.columns else pd.NA
    )
    cleaned_tmp["has_quarterlike_duration"] = cleaned_tmp["duration_days"].between(60, 120, inclusive="both")
    real_q4 = (
        cleaned_tmp[
            (cleaned_tmp["fq"] == 4)
            & (cleaned_tmp["has_quarterlike_duration"] | (cleaned_tmp["fp_norm"] == "Q4"))
        ]
        .groupby(["ticker", "concept", "fy_fiscal", "unit"], as_index=False)
        .size()
        .rename(columns={"size": "real_q4_rows"})
    )

    base = (
        fy_best.merge(qsum, on=["ticker", "concept", "fy_fiscal", "unit"], how="left")
        .merge(real_q4, on=["ticker", "concept", "fy_fiscal", "unit"], how="left")
        .merge(er_best, on=["ticker", "fy_fiscal"], how="left")
    )

    need_true_q4 = base[
        base["real_q4_rows"].isna()
        & base[["q1", "q2", "q3"]].notna().all(axis=1)
        & base["FY_value"].notna()
    ].copy()
    if need_true_q4.empty:
        return cleaned

    need_true_q4["true_q4"] = need_true_q4["FY_value"] - (
        need_true_q4["q1"] + need_true_q4["q2"] + need_true_q4["q3"]
    )

    rows: List[dict] = []
    for _, r in need_true_q4.iterrows():
        end_month = fisc_end_map.get(str(r["ticker"]), np.nan)
        fy = int(r["fy_fiscal"])
        fq = 4
        fq_label = f"Q{fq}"
        rows.append(
            {
                "ticker": r["ticker"],
                "concept": r["concept"],
                "unit": r["unit"],
                "value": float(r["true_q4"]),
                "date": r["FY_date"],
                "start": pd.NaT,
                "filed": r.get("FY_filed", pd.NaT),
                "accepted": pd.NaT,
                "report_date": pd.NaT,
                "earnings_release": r.get("earnings_release", pd.NaT),
                "accn": r.get("FY_accn", pd.NA),
                "duration_days": np.nan,
                "fiscal_end_month": end_month,
                "fy_fiscal": fy,
                "fq": fq,
                "fq_label": fq_label,
                "fiscal_period_label": f"{fq_label} FY{fy}",
                "cal_year": (
                    pd.to_datetime(r["FY_date"]).year if pd.notna(r["FY_date"]) else pd.NA
                ),
                "cal_quarter": (
                    int(((pd.to_datetime(r["FY_date"]).month - 1) // 3) + 1)
                    if pd.notna(r["FY_date"])
                    else pd.NA
                ),
                "is_derived_q4": True,
            }
        )

    add_df = pd.DataFrame(rows)
    for c in cleaned.columns:
        if c not in add_df.columns:
            add_df[c] = pd.NA
    add_df = add_df[cleaned.columns]
    return pd.concat([cleaned, add_df], ignore_index=True)


def clean_sec_facts(
    path_in: str,
    path_out: Optional[str] = None,
    *,
    fisc_end_map: Dict[str, int] = DEFAULT_FISC_ENDS,
    add_true_q4: bool = True,
    drop_fy_rows: bool = True,
    fy_drop_rule: str = "both",
    fy_duration_threshold: int = 300,
) -> pd.DataFrame:
    """File-IO wrapper around :func:`clean_sec_facts_df`."""
    df = pd.read_csv(path_in, low_memory=False)
    cleaned = clean_sec_facts_df(
        df,
        fisc_end_map=fisc_end_map,
        add_true_q4=add_true_q4,
        drop_fy_rows=drop_fy_rows,
        fy_drop_rule=fy_drop_rule,
        fy_duration_threshold=fy_duration_threshold,
    )
    if path_out:
        cleaned.to_csv(path_out, index=False)
    return cleaned
