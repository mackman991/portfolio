"""Technical-indicator engine: SMAs, RSI, MACD, rolling volatility."""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class TechnicalAnalysis:
    """Compute and (optionally) plot a standard set of technical indicators per ticker."""

    def __init__(
        self,
        ma_windows=(20, 50, 200),
        rsi_window=14,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        vol_window=30,
        annualize_vol=False,
    ):
        self.ma_windows = tuple(ma_windows)
        self.rsi_window = int(rsi_window)
        self.macd_fast = int(macd_fast)
        self.macd_slow = int(macd_slow)
        self.macd_signal = int(macd_signal)
        self.vol_window = int(vol_window)
        self.annualize_vol = bool(annualize_vol)

    def add_indicators(self, price_df: pd.DataFrame) -> pd.DataFrame:
        if price_df is None or not isinstance(price_df, pd.DataFrame):
            raise TypeError("add_indicators expected a pandas DataFrame.")
        if "ticker" not in price_df.columns or "close" not in price_df.columns:
            raise ValueError("DataFrame must have 'ticker' and 'close' columns.")

        df = price_df.copy()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        return df.groupby("ticker", group_keys=False).apply(self._compute_indicators_one)

    def _compute_indicators_one(self, s: pd.DataFrame) -> pd.DataFrame:
        s = s.copy()

        if "date" in s.columns:
            s["date"] = pd.to_datetime(s["date"], errors="coerce")
            s = s.sort_values("date")
        else:
            try:
                if not pd.api.types.is_datetime64_any_dtype(s.index):
                    s.index = pd.to_datetime(s.index, errors="coerce")
                s = s.sort_index()
            except Exception:
                pass

        s["ret_d"] = s["close"].pct_change()

        for w in self.ma_windows:
            s[f"sma_{w}"] = s["close"].rolling(window=w, min_periods=w).mean()

        s["rsi"] = self._rsi(s["close"], self.rsi_window)

        macd, sig, hist = self._macd(s["close"], self.macd_fast, self.macd_slow, self.macd_signal)
        s["macd"] = macd
        s["macd_signal"] = sig
        s["macd_hist"] = hist

        vol = s["ret_d"].rolling(self.vol_window, min_periods=self.vol_window).std()
        if self.annualize_vol:
            vol *= np.sqrt(252)
        s[f"vol_{self.vol_window}"] = vol
        return s

    @staticmethod
    def _rsi(close: pd.Series, window: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _macd(close: pd.Series, fast: int, slow: int, signal: int):
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        return macd, macd_signal, macd - macd_signal

    def plot_indicators(self, df: pd.DataFrame, ticker: str | None = None, title_prefix: str = ""):
        if df is None or not isinstance(df, pd.DataFrame):
            raise TypeError("plot_indicators expected a pandas DataFrame.")
        if "ticker" not in df.columns:
            raise ValueError("plot_indicators expects a 'ticker' column.")

        if ticker is None:
            for tkr in df["ticker"].dropna().unique():
                self._plot_single(df[df["ticker"] == tkr].copy(), tkr, title_prefix)
        else:
            data = df[df["ticker"] == ticker].copy()
            if data.empty:
                raise ValueError(f"No rows for ticker '{ticker}'.")
            self._plot_single(data, ticker, title_prefix)

    def _plot_single(self, data: pd.DataFrame, ticker: str | None, title_prefix: str):
        if "date" in data.columns:
            x = pd.to_datetime(data["date"], errors="coerce")
        else:
            if not pd.api.types.is_datetime64_any_dtype(data.index):
                data.index = pd.to_datetime(data.index, errors="coerce")
            data = data.sort_index()
            x = data.index

        needed = ["close", "rsi", "macd", "macd_signal", "macd_hist"] + [f"sma_{w}" for w in self.ma_windows]
        missing = [c for c in needed if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns (run add_indicators first): {missing}")

        fig, axes = plt.subplots(
            nrows=4, ncols=1, sharex=True, figsize=(12, 10),
            gridspec_kw={"height_ratios": [3, 1, 1, 1]},
        )
        ax_price, ax_rsi, ax_macd, ax_vol = axes

        ax_price.plot(x, data["close"], label="Close", lw=1.4)
        for w in self.ma_windows:
            ax_price.plot(x, data[f"sma_{w}"], label=f"SMA {w}", lw=1.1)
        hdr = f"{title_prefix} {ticker}".strip() if ticker else title_prefix.strip()
        ax_price.set_title(f"{hdr} — Price & SMAs" if hdr else "Price & SMAs")
        ax_price.set_ylabel("Price")
        ax_price.legend(loc="upper left")
        ax_price.grid(True, alpha=0.3)

        ax_rsi.plot(x, data["rsi"], lw=1.1)
        ax_rsi.axhline(70, ls="--", lw=0.9, alpha=0.7)
        ax_rsi.axhline(30, ls="--", lw=0.9, alpha=0.7)
        ax_rsi.set_ylabel(f"RSI({self.rsi_window})")
        ax_rsi.set_ylim(0, 100)
        ax_rsi.grid(True, alpha=0.3)

        ax_macd.bar(x, data["macd_hist"], width=1.0, alpha=0.6)
        ax_macd.plot(x, data["macd"], lw=1.1, label="MACD")
        ax_macd.plot(x, data["macd_signal"], lw=1.1, label="Signal")
        ax_macd.set_ylabel("MACD")
        ax_macd.legend(loc="upper left")
        ax_macd.grid(True, alpha=0.3)

        vol_col = f"vol_{self.vol_window}"
        ax_vol.plot(x, data[vol_col], lw=1.1)
        ax_vol.set_ylabel(f"Vol {self.vol_window}d")
        ax_vol.grid(True, alpha=0.3)

        locator = mdates.AutoDateLocator()
        ax_vol.xaxis.set_major_locator(locator)
        ax_vol.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

        plt.tight_layout()
        plt.show()
