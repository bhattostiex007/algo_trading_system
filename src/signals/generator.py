"""
SignalGenerator
===============
Computes technical indicators and generates BUY / SELL / HOLD signals
for a universe of stocks using the `ta` library.

Indicators
----------
- EMA(20), EMA(50)          — trend filter
- MACD(12, 26, 9)           — momentum crossover entry
- RSI(14)                   — overbought / oversold confirmation
- Bollinger Bands(20, 2)    — mean reversion context
- ATR(14)                   — volatility sizing for trailing stop

Signal Logic
------------
BUY  : RSI < 35  AND  MACD line crosses above MACD signal  AND  Close > EMA(50)
SELL : RSI > 65  OR   price falls below trailing stop (6 % from recent high)
HOLD : everything else

Usage
-----
from src.data.loader import DataLoader
from src.signals.generator import SignalGenerator

loader = DataLoader()
ohlcv  = loader.load()

sg = SignalGenerator(ohlcv)
signals = sg.generate_all()     # dict[ticker -> DataFrame with signals]
summary = sg.summary()          # signal count table across all tickers
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
import ta
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SignalGenerator")

# ─── Signal constants ────────────────────────────────────────────────────────
BUY  =  1
HOLD =  0
SELL = -1

# ─── Default parameters ─────────────────────────────────────────────────────
DEFAULT_PARAMS = dict(
    ema_fast       = 20,
    ema_slow       = 50,
    macd_fast      = 12,
    macd_slow      = 26,
    macd_signal    = 9,
    rsi_period     = 14,
    rsi_buy        = 40,       # RSI below this → oversold → potential BUY (relaxed from 35)
    rsi_sell       = 65,       # RSI above this → overbought → potential SELL
    bb_period      = 20,
    bb_std         = 2.0,
    atr_period     = 14,
    trail_pct      = 0.06,     # 6 % trailing stop-loss
)


class SignalGenerator:
    """
    Computes indicators and generates trading signals for multiple tickers.

    Parameters
    ----------
    ohlcv : dict[ticker -> OHLCV DataFrame]
        Output of DataLoader.load().
    params : dict, optional
        Override any default parameter (see DEFAULT_PARAMS).
    """

    def __init__(
        self,
        ohlcv: Dict[str, pd.DataFrame],
        params: Optional[dict] = None,
    ) -> None:
        self.ohlcv  = ohlcv
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self._results: Dict[str, pd.DataFrame] = {}

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def generate_all(self) -> Dict[str, pd.DataFrame]:
        """
        Run indicator computation + signal generation for every ticker.

        Returns
        -------
        dict[ticker -> DataFrame]
            Each DataFrame contains all OHLCV columns plus:
            EMA_20, EMA_50, MACD, MACD_signal, MACD_diff,
            RSI, BB_upper, BB_mid, BB_lower, BB_pband, ATR,
            Trailing_Stop, Signal (1/0/-1), Signal_Label
        """
        logger.info("Generating signals for %d tickers …", len(self.ohlcv))
        for ticker, df in self.ohlcv.items():
            try:
                result = self._process_ticker(ticker, df)
                self._results[ticker] = result
            except Exception as exc:
                logger.error("%-6s — failed: %s", ticker, exc)
        logger.info("Done. %d tickers processed.", len(self._results))
        return self._results

    def get(self, ticker: str) -> pd.DataFrame:
        """Return the signal DataFrame for a single ticker."""
        if ticker not in self._results:
            raise KeyError(f"{ticker} not found. Call generate_all() first.")
        return self._results[ticker]

    def summary(self) -> pd.DataFrame:
        """Return a signal count / stats table across all tickers."""
        rows = []
        for ticker, df in self._results.items():
            s = df["Signal"]
            buys  = (s == BUY).sum()
            sells = (s == SELL).sum()
            holds = (s == HOLD).sum()
            rows.append({
                "Ticker":       ticker,
                "BUY  signals": buys,
                "SELL signals": sells,
                "HOLD signals": holds,
                "Total rows":   len(df),
                "BUY  rate %":  round(buys  / len(df) * 100, 1),
                "SELL rate %":  round(sells / len(df) * 100, 1),
                "Avg RSI":      round(df["RSI"].mean(), 1),
                "Avg ATR":      round(df["ATR"].mean(), 2),
            })
        return pd.DataFrame(rows).set_index("Ticker")

    # ──────────────────────────────────────────────────────────────────────
    # Core computation
    # ──────────────────────────────────────────────────────────────────────

    def _process_ticker(self, ticker: str, df: pd.DataFrame) -> pd.DataFrame:
        p   = self.params
        out = df.copy()

        close  = out["Close"]
        high   = out["High"]
        low    = out["Low"]

        # ── EMA ─────────────────────────────────────────────────────────
        out["EMA_20"] = EMAIndicator(close=close, window=p["ema_fast"]).ema_indicator()
        out["EMA_50"] = EMAIndicator(close=close, window=p["ema_slow"]).ema_indicator()

        # ── MACD ────────────────────────────────────────────────────────
        macd_obj        = MACD(
            close=close,
            window_fast=p["macd_fast"],
            window_slow=p["macd_slow"],
            window_sign=p["macd_signal"],
        )
        out["MACD"]        = macd_obj.macd()
        out["MACD_signal"] = macd_obj.macd_signal()
        out["MACD_diff"]   = macd_obj.macd_diff()   # histogram (MACD - signal)

        # ── RSI ─────────────────────────────────────────────────────────
        out["RSI"] = RSIIndicator(close=close, window=p["rsi_period"]).rsi()

        # ── Bollinger Bands ─────────────────────────────────────────────
        bb_obj           = BollingerBands(
            close=close,
            window=p["bb_period"],
            window_dev=p["bb_std"],
        )
        out["BB_upper"] = bb_obj.bollinger_hband()
        out["BB_mid"]   = bb_obj.bollinger_mavg()
        out["BB_lower"] = bb_obj.bollinger_lband()
        out["BB_pband"] = bb_obj.bollinger_pband()   # % position within bands

        # ── ATR ─────────────────────────────────────────────────────────
        out["ATR"] = AverageTrueRange(
            high=high, low=low, close=close, window=p["atr_period"]
        ).average_true_range()

        # ── Trailing Stop ────────────────────────────────────────────────
        out["Trailing_Stop"] = self._compute_trailing_stop(close, p["trail_pct"])

        # ── Signals ──────────────────────────────────────────────────────
        out["Signal"]       = self._compute_signals(out)
        out["Signal_Label"] = out["Signal"].map({BUY: "BUY", SELL: "SELL", HOLD: "HOLD"})

        # Drop warmup rows (NaN indicators in first ~50 rows)
        out = out.dropna(subset=["EMA_50", "MACD", "RSI", "ATR"])

        logger.info("  %-6s — %d rows | BUY: %d | SELL: %d | HOLD: %d",
                    ticker,
                    len(out),
                    (out["Signal"] == BUY).sum(),
                    (out["Signal"] == SELL).sum(),
                    (out["Signal"] == HOLD).sum())
        return out

    # ──────────────────────────────────────────────────────────────────────
    # Signal logic
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_trailing_stop(close: pd.Series, trail_pct: float) -> pd.Series:
        """
        Rolling 6 % trailing stop-loss.
        Uses a 60-day window so the stop tracks the recent peak
        without firing on every minor dip.
        For each day: stop = rolling_max(close, 60) * (1 - trail_pct)
        """
        rolling_max = close.rolling(window=60, min_periods=1).max()
        return rolling_max * (1 - trail_pct)

    def _compute_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Apply vectorised EMA crossover signal rules.

        BUY  : EMA_20 crosses above EMA_50 (Golden Cross) AND RSI < 70
               → Short-term trend turning bullish above long-term trend
        SELL : EMA_20 crosses below EMA_50 (Death Cross)
               OR Close < Trailing_Stop (6% stop-loss hit)
               OR RSI > 75 (extremely overbought)
        HOLD : between crossovers

        Interview note: This is a classic trend-following strategy.
        EMA crossovers give clear, spaced signals (3–6 per year per ticker).
        RSI filter avoids entering at overbought peaks.
        The trailing stop caps downside on sharp reversals.
        """
        p = self.params

        ema20 = df["EMA_20"]
        ema50 = df["EMA_50"]
        prev_ema20 = ema20.shift(1)
        prev_ema50 = ema50.shift(1)

        # Golden Cross: EMA20 crosses ABOVE EMA50 today, was below yesterday
        golden_cross = (ema20 > ema50) & (prev_ema20 <= prev_ema50)

        # Death Cross: EMA20 crosses BELOW EMA50 today, was above yesterday
        death_cross  = (ema20 < ema50) & (prev_ema20 >= prev_ema50)

        buy_condition = golden_cross & (df["RSI"] < 70)

        sell_condition = (
            death_cross |
            (df["Close"] < df["Trailing_Stop"]) |
            (df["RSI"] > 75)
        )

        signal = pd.Series(HOLD, index=df.index, dtype=int)
        signal[sell_condition] = SELL
        signal[buy_condition]  = BUY  # BUY overrides if both fire (edge case)

        return signal


# ─── CLI convenience ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.data.loader import DataLoader

    loader = DataLoader()
    ohlcv  = loader.load()

    sg      = SignalGenerator(ohlcv)
    results = sg.generate_all()

    print("\n=== Signal Summary ===")
    print(sg.summary().to_string())
