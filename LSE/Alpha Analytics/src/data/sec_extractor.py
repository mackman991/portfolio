"""SEC EDGAR XBRL extractor with 8-K Item 2.02 earnings-release detection."""
from __future__ import annotations

import logging
import time
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


class SecFinancialsExtractor:
    """Pull XBRL company-concept facts from EDGAR and tag each fact with the
    nearest 8-K Item 2.02 acceptance timestamp (the earnings release).
    """

    def __init__(
        self,
        cik_map: Dict[str, str],
        concepts: Dict[str, str],
        user_agent: str,
        polite_delay: float = 0.5,
        earnings_window_days: Tuple[int, int] = (0, 60),
        include_earnings_release: bool = True,
    ):
        if not user_agent or "@" not in user_agent:
            raise ValueError(
                "SEC requires a User-Agent identifying the requester (must include an email)."
            )
        self.cik_map = dict(cik_map)
        self.concepts = dict(concepts)
        self.headers = {
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov",
        }
        self.polite_delay = polite_delay
        self.earnings_window_days = earnings_window_days
        self.include_earnings_release = include_earnings_release

    # ------------ Internal helpers ------------
    def _get_company_concept(self, cik: str, concept: str) -> dict:
        url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{concept}.json"
        r = requests.get(url, headers=self.headers, timeout=30)
        r.raise_for_status()
        return r.json()

    def _get_submissions(self, cik: str) -> dict:
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        r = requests.get(url, headers=self.headers, timeout=30)
        r.raise_for_status()
        return r.json()

    @staticmethod
    def _submissions_recent_df(subs_json: dict) -> pd.DataFrame:
        recent = subs_json.get("filings", {}).get("recent", {})
        if not recent:
            return pd.DataFrame()
        df = pd.DataFrame(recent)
        for col in ["accessionNumber", "filingDate", "reportDate", "acceptanceDateTime", "form", "items"]:
            if col not in df.columns:
                df[col] = np.nan
        df["filingDate"] = pd.to_datetime(df["filingDate"], utc=True, errors="coerce")
        df["reportDate"] = pd.to_datetime(df["reportDate"], utc=True, errors="coerce")
        df["acceptanceDateTime"] = pd.to_datetime(df["acceptanceDateTime"], utc=True, errors="coerce")
        df["accessionNumber"] = df["accessionNumber"].astype(str).str.strip()
        return df

    @staticmethod
    def _map_accn_metadata(facts: pd.DataFrame, subs_df: pd.DataFrame) -> pd.DataFrame:
        if facts.empty or subs_df.empty:
            facts = facts.copy()
            facts["accepted"] = pd.NaT
            facts["report_date"] = pd.NaT
            return facts
        meta = subs_df[["accessionNumber", "acceptanceDateTime", "reportDate", "form"]].drop_duplicates().rename(
            columns={
                "accessionNumber": "accn",
                "acceptanceDateTime": "accepted",
                "reportDate": "report_date",
                "form": "form_submissions",
            }
        )
        out = facts.merge(meta, on="accn", how="left")
        out["form"] = np.where(out["form_submissions"].notna(), out["form_submissions"], out["form"])
        return out.drop(columns=["form_submissions"])

    def _infer_earnings_release_dates(
        self,
        subs_df: pd.DataFrame,
        period_end: pd.Timestamp,
        accepted_cutoff: Optional[pd.Timestamp],
    ) -> Optional[pd.Timestamp]:
        """Earliest 8-K with Item 2.02 within the configured window after period end."""
        if subs_df.empty or pd.isna(period_end):
            return pd.NaT

        low, high = self.earnings_window_days
        window_start = period_end + pd.Timedelta(days=low)
        window_end = period_end + pd.Timedelta(days=high)

        cand = subs_df.copy()
        cand = cand[cand["form"].astype(str).str.upper().eq("8-K")]
        cand = cand[cand["items"].astype(str).str.contains(r"\b2\.02\b", regex=True, na=False)]
        cand = cand[(cand["acceptanceDateTime"] >= window_start) & (cand["acceptanceDateTime"] <= window_end)]
        if pd.notna(accepted_cutoff):
            cand = cand[cand["acceptanceDateTime"] <= accepted_cutoff]
        if cand.empty:
            return pd.NaT
        return cand.sort_values("acceptanceDateTime").iloc[0]["acceptanceDateTime"]

    # ------------ Public API ------------
    def extract(
        self,
        tickers: Optional[Iterable[str]] = None,
        save_csv: Optional[str] = None,
        cutoff: Optional[str] = "2019-12-31",
    ) -> pd.DataFrame:
        tickers = list(tickers) if tickers else list(self.cik_map.keys())

        rows: List[dict] = []
        subs_cache: Dict[str, pd.DataFrame] = {}

        for ticker in tickers:
            cik = self.cik_map.get(ticker)
            if not cik:
                logger.warning("No CIK mapped for %s; skipping.", ticker)
                continue

            if cik not in subs_cache:
                try:
                    subs_cache[cik] = self._submissions_recent_df(self._get_submissions(cik))
                except Exception as exc:  # noqa: BLE001
                    logger.error("Submissions fetch failed for %s: %s", ticker, exc)
                    subs_cache[cik] = pd.DataFrame()
                time.sleep(self.polite_delay)

            logger.info("Extracting SEC concepts for %s", ticker)
            for label, usgaap in self.concepts.items():
                try:
                    data = self._get_company_concept(cik, usgaap)
                    units = data.get("units", {})
                    preferred = ("USD", "USD/shares", "shares")
                    if "share" in usgaap.lower():
                        preferred = ("shares", "USD/shares", "USD")
                    for unit_type in preferred:
                        if unit_type in units:
                            for item in units[unit_type]:
                                rows.append(
                                    {
                                        "ticker": ticker,
                                        "cik": cik,
                                        "concept": label,
                                        "us_gaap": usgaap,
                                        "value": item.get("val"),
                                        "fy": item.get("fy"),
                                        "fp": str(item.get("fp")).upper().strip() if item.get("fp") else np.nan,
                                        "form": item.get("form"),
                                        "date": item.get("end"),
                                        "start": item.get("start"),
                                        "filed": item.get("filed"),
                                        "accn": str(item.get("accn")).strip() if item.get("accn") else np.nan,
                                        "unit": unit_type,
                                    }
                                )
                            break
                    else:
                        logger.warning("No records for %s [%s]", label, usgaap)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Failed fetching %s [%s] for %s: %s", label, usgaap, ticker, exc)
                time.sleep(self.polite_delay)

        df = pd.DataFrame(rows)
        if df.empty:
            if save_csv:
                df.to_csv(save_csv, index=False)
            return df

        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df["start"] = pd.to_datetime(df["start"], utc=True, errors="coerce")
        df["filed"] = pd.to_datetime(df["filed"], utc=True, errors="coerce")
        df["fy"] = pd.to_numeric(df["fy"], errors="coerce").astype("Int64")
        df["fp"] = df["fp"].astype(str).str.upper().str.strip()

        # Enrich with submissions metadata
        enriched = []
        for ticker in tickers:
            cik = self.cik_map.get(ticker)
            if not cik:
                continue
            part = df[df["cik"] == cik].copy()
            part = self._map_accn_metadata(part, subs_cache.get(cik, pd.DataFrame()))
            enriched.append(part)
        df = pd.concat(enriched, ignore_index=True)

        if self.include_earnings_release:
            df["earnings_release"] = pd.NaT
            for (ticker, fy, fp), g in df.groupby(["ticker", "fy", "fp"], dropna=False):
                if g.empty:
                    continue
                period_end = g["date"].dropna().max() if g["date"].notna().any() else pd.NaT
                qk = g[g["form"].isin(["10-Q", "10-K"])]
                accepted_cutoff = qk["accepted"].min() if (not qk.empty and qk["accepted"].notna().any()) else pd.NaT
                subs_df = subs_cache.get(self.cik_map.get(ticker, ""), pd.DataFrame())
                er = self._infer_earnings_release_dates(subs_df, period_end, accepted_cutoff)
                if pd.notna(er):
                    df.loc[(df["ticker"] == ticker) & (df["fy"] == fy) & (df["fp"] == fp), "earnings_release"] = er

        df = df.sort_values(["ticker", "concept", "date", "filed"], ascending=[True, True, False, False]).reset_index(drop=True)

        if cutoff:
            cutoff_ts = pd.Timestamp(cutoff, tz="UTC")
            df = df[df["date"] >= cutoff_ts].reset_index(drop=True)

        if save_csv:
            df.to_csv(save_csv, index=False)
            logger.info("Saved SEC financials to %s", save_csv)
        return df
