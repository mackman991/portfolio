"""
Project configuration.

API keys are loaded from environment variables (use a .env file locally; never commit it).
See .env.example for the required variables.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict

from dotenv import load_dotenv

load_dotenv()


def _require_env(name: str) -> str:
    """Fetch an env var or raise — no silent fallbacks (which leak credentials)."""
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(
            f"Missing required environment variable: {name}. "
            "Copy .env.example to .env and populate it."
        )
    return value


# ---- API keys (loaded lazily so import doesn't fail in test/CI) ----
def fmp_api_key() -> str:
    return _require_env("FMP_API_KEY")


def fred_api_key() -> str:
    return _require_env("FRED_API_KEY")


def sec_user_agent() -> str:
    """SEC requires an identifying User-Agent (contact email)."""
    return os.getenv("SEC_USER_AGENT", "research-project example@example.com")


# ---- Universe & date range ----
TICKERS = ["AAPL", "NVDA", "GOOG"]

# Expanded 19-ticker mega/large-cap universe (used for the 19-stock backtest in
# data/strategy_summary_19.csv).
TICKERS_19 = [
    "AAPL", "AMD", "AMZN", "ASML", "AVGO", "AZN", "COST", "CSCO",
    "GOOG", "GOOGL", "INTU", "LIN", "META", "MSFT", "NFLX", "NVDA",
    "TMUS", "TSLA", "TXN",
]

# S&P 100 (OEX) constituents for the broader technical x earnings study.
# List snapshot as of early 2026. BRK-B uses FMP's hyphen convention
# (yfinance also accepts BRK-B). Adjust if constituents change.
SP100_TICKERS = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT", "AMZN",
    "AVGO", "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "BRK-B", "C",
    "CAT", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS",
    "CVX", "DE", "DHR", "DIS", "DUK", "EMR", "F", "FDX", "GD", "GE",
    "GILD", "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "INTU",
    "ISRG", "JNJ", "JPM", "KHC", "KO", "LIN", "LLY", "LMT", "LOW", "MA",
    "MCD", "MDLZ", "MDT", "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT",
    "NEE", "NFLX", "NKE", "NOW", "NVDA", "ORCL", "PEP", "PFE", "PG", "PLD",
    "PM", "PYPL", "QCOM", "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TGT",
    "TMO", "TMUS", "TSLA", "TXN", "UNH", "UNP", "UPS", "USB", "V", "VZ",
    "WBA", "WFC", "WMT", "XOM",
]

CIK_MAP: Dict[str, str] = {
    "AAPL": "0000320193",
    "NVDA": "0001045810",
    "GOOG": "0001652044",
}

# Map ticker -> fiscal year-end month (used by SEC cleaner)
FISCAL_END_MONTH: Dict[str, int] = {
    "AAPL": 9,
    "GOOG": 12,
    "NVDA": 1,
}

# us-gaap concepts to extract from EDGAR (display label -> us-gaap concept)
SEC_CONCEPTS: Dict[str, str] = {
    "Revenue": "Revenues",
    "NetIncome": "NetIncomeLoss",
    "Assets": "Assets",
    "CapEx": "PaymentsToAcquirePropertyPlantAndEquipment",
    "CashFlow_Operating": "NetCashProvidedByUsedInOperatingActivities",
    "EPS_Basic": "EarningsPerShareBasic",
    "EPS_Diluted": "EarningsPerShareDiluted",
    "DebtEquity": "LiabilitiesAndStockholdersEquity",
}

# FRED economic indicators
ECON_DATA_POINTS: Dict[str, str] = {
    "gdp": "GDPC1",
    "cpi": "CPIAUCSL",
    "interest_rate": "FEDFUNDS",
    "unemployment": "UNRATE",
    "payrolls": "PAYEMS",
}


@dataclass(frozen=True)
class DateRange:
    start: str = "2019-12-31"
    end: str = "2025-07-31"


DATES = DateRange()

# Event window defaults
DAYS_BEFORE_EPS = 10
DAYS_AFTER_EPS = 10

# Default technical-indicator set used by the FMP TechnicalIndicatorExtractor.
# Each entry is (type, period). Types must be in ALLOWED_TYPES of the extractor.
# MACD is *not* in this list because FMP does not serve it directly — it is
# derived locally from ema_12 and ema_26.
DEFAULT_INDICATORS: list[tuple[str, int]] = [
    ("rsi", 14),
    ("sma", 20),
    ("sma", 50),
    ("sma", 200),
    ("ema", 12),
    ("ema", 26),
]

# Output paths
DATA_DIR = os.getenv("DATA_DIR", "data")
