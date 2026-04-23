"""Date-column normaliser shared across the price/EPS/SEC frames."""
from __future__ import annotations

import pandas as pd


class CleanDates:
    """Standardise date columns across loosely-formatted CSV-like DataFrames.

    - Lower-case & strip column names
    - Rename a primary date-like column to ``date``
      (looks at ``date`` / ``unnamed: 0`` / ``unnamed:0``)
    - Parse ``date`` to ``datetime.date``, drop unparsable rows, sort, set as index
    - Also parse ``filed``, ``accepted``, ``report_date``, ``earnings_release``
      if present (left as columns, no dropping)
    """

    PRIMARY_DATE_CANDIDATES = ("date", "unnamed: 0", "unnamed:0")
    EXTRA_DATE_COLS = ("filed", "accepted", "report_date", "earnings_release")

    @staticmethod
    def _parse_to_date(series: pd.Series) -> pd.Series:
        dt = pd.to_datetime(series, utc=True, errors="coerce")
        return dt.dt.date

    @classmethod
    def clean(cls, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or not isinstance(df, pd.DataFrame):
            raise TypeError("Please pass a pandas DataFrame.")

        df = df.copy()
        df.columns = df.columns.str.strip().str.lower()

        if "date" not in df.columns:
            for cand in cls.PRIMARY_DATE_CANDIDATES:
                if cand in df.columns and cand != "date":
                    df.rename(columns={cand: "date"}, inplace=True)
                    break

        if "date" in df.columns:
            df["date"] = cls._parse_to_date(df["date"])
            df = df.dropna(subset=["date"])
            df = df.sort_values("date").set_index("date")

        for col in cls.EXTRA_DATE_COLS:
            if col in df.columns:
                df[col] = cls._parse_to_date(df[col])

        return df
