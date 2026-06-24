"""
BacktestEngine
==============
Vectorised backtesting engine that simulates trading based on pre-computed
BUY / SELL / HOLD signals from SignalGenerator.

Design decisions (important for interviews)
-------------------------------------------
1.  **No lookahead bias**: Signals are computed on day T's close price,
    but positions are entered at day T+1's open (implemented via .shift(1)).

2.  **Transaction costs**: 0.1% per trade (both entry and exit),
    deducted from daily returns when a position change occurs.

3.  **Position sizing**: Fixed fractional — each ticker gets equal capital.
    Within a ticker, the strategy is all-in (1 position at a time).

4.  **Multi-ticker aggregation**: Each ticker is backtested independently,
    then combined into an equal-weighted portfolio.

5.  **Benchmark**: Buy-and-hold (always fully invested) for comparison.

Usage
-----
from src.data.loader import DataLoader
from src.signals.generator import SignalGenerator
from src.backtest.engine import BacktestEngine

loader  = DataLoader()
ohlcv   = loader.load()
sg      = SignalGenerator(ohlcv)
signals = sg.generate_all()

engine  = BacktestEngine(initial_capital=100_000, commission=0.001)
results = engine.run_portfolio(signals)

print(results['portfolio_metrics'])
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.metrics import full_metrics, compute_drawdown_series

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("BacktestEngine")

# ─── Signal constants ────────────────────────────────────────────────────────
BUY  =  1
SELL = -1
HOLD =  0


class BacktestEngine:
    """
    Vectorised backtesting engine for signal-based strategies.

    Parameters
    ----------
    initial_capital : float
        Starting capital in USD. Default $100,000.
    commission : float
        Round-trip commission rate per trade (0.001 = 0.1%). Applied
        to both entry and exit.
    position_size : float
        Fraction of per-ticker capital to deploy per trade (1.0 = all-in).
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission: float      = 0.001,
        position_size: float   = 1.0,
    ) -> None:
        self.initial_capital = initial_capital
        self.commission      = commission
        self.position_size   = position_size

        # Populated after run_portfolio()
        self._ticker_results: Dict[str, dict] = {}

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def run_single(
        self,
        ticker: str,
        df: pd.DataFrame,
        capital: float | None = None,
    ) -> dict:
        """
        Run a vectorised backtest for a single ticker.

        Parameters
        ----------
        ticker  : str
        df      : DataFrame with columns: Close, Signal (from SignalGenerator)
        capital : Starting capital (defaults to self.initial_capital)

        Returns
        -------
        dict with keys: equity, returns, positions, trades, benchmark_equity,
                        trade_pnl, metrics, benchmark_metrics
        """
        cap = capital or self.initial_capital
        close  = df["Close"].copy()
        signal = df["Signal"].copy()

        # ── Step 1: Convert signals → held positions ──────────────────────
        # Mark 1.0 at BUY, 0.0 at SELL, NaN elsewhere → forward-fill
        raw_pos = pd.Series(np.nan, index=signal.index)
        raw_pos[signal == BUY]  = 1.0
        raw_pos[signal == SELL] = 0.0
        positions = raw_pos.ffill().fillna(0.0)

        # ── Step 2: Shift positions by 1 day (no lookahead bias) ──────────
        # Signal on close of day T → position entered at open of day T+1
        positions_shifted = positions.shift(1).fillna(0.0)

        # Scale by position_size
        positions_shifted = positions_shifted * self.position_size

        # ── Step 3: Daily returns ─────────────────────────────────────────
        price_returns = close.pct_change().fillna(0.0)
        strategy_returns_gross = price_returns * positions_shifted

        # ── Step 4: Transaction costs ─────────────────────────────────────
        # Cost fires when position changes (trade execution)
        trade_events = positions_shifted.diff().abs().fillna(0.0)
        cost_series  = trade_events * self.commission
        strategy_returns = strategy_returns_gross - cost_series

        # ── Step 5: Equity curve ──────────────────────────────────────────
        equity = (1 + strategy_returns).cumprod() * cap

        # ── Step 6: Benchmark (buy-and-hold) ─────────────────────────────
        bh_returns = price_returns
        benchmark_equity = (1 + bh_returns).cumprod() * cap

        # ── Step 7: Trade P&L series ──────────────────────────────────────
        trade_pnl = self._compute_trade_pnl(close, positions)

        # ── Step 8: Metrics ───────────────────────────────────────────────
        active_returns = strategy_returns[positions_shifted > 0]
        metrics    = full_metrics(strategy_returns, equity, trade_pnl)
        bh_metrics = full_metrics(bh_returns, benchmark_equity)

        logger.info(
            "  %-6s | Return: %+.1f%% | Sharpe: %.2f | MaxDD: %.1f%% | Trades: %d",
            ticker,
            metrics["Total Return (%)"],
            metrics["Sharpe Ratio"],
            metrics["Max Drawdown (%)"],
            metrics.get("Num Trades", 0),
        )

        return {
            "equity":           equity,
            "returns":          strategy_returns,
            "positions":        positions_shifted,
            "trade_pnl":        trade_pnl,
            "benchmark_equity": benchmark_equity,
            "metrics":          metrics,
            "bh_metrics":       bh_metrics,
        }

    def run_portfolio(
        self,
        signals: Dict[str, pd.DataFrame],
        custom_weights: Dict[str, float] | None = None,
    ) -> dict:
        """
        Run backtest across all tickers and aggregate into an
        equal-weighted portfolio.

        Parameters
        ----------
        signals : dict[ticker -> signals DataFrame]

        Returns
        -------
        dict with keys:
            ticker_results     — per-ticker result dicts
            portfolio_equity   — combined equity curve
            portfolio_returns  — combined daily returns
            portfolio_metrics  — aggregated performance metrics
            benchmark_equity   — equal-weight buy-and-hold
            metrics_table      — DataFrame comparing all tickers + portfolio
        """
        logger.info("Running portfolio backtest for %d tickers …", len(signals))

        all_equity     = {}
        all_returns    = {}
        all_bh_equity  = {}

        for ticker, df in signals.items():
            if custom_weights:
                capital = self.initial_capital * custom_weights.get(ticker, 0.0)
            else:
                capital = self.initial_capital / len(signals)
            
            # Skip if allocated $0
            if capital <= 0:
                continue

            result = self.run_single(ticker, df, capital=capital)
            self._ticker_results[ticker] = result
            all_equity[ticker]    = result["equity"]
            all_returns[ticker]   = result["returns"]
            all_bh_equity[ticker] = result["benchmark_equity"]

        # ── Portfolio equity = sum of per-ticker equity curves ─────────────
        equity_df  = pd.DataFrame(all_equity).dropna()
        returns_df = pd.DataFrame(all_returns).dropna()
        bh_df      = pd.DataFrame(all_bh_equity).dropna()

        portfolio_equity   = equity_df.sum(axis=1)
        portfolio_returns  = portfolio_equity.pct_change().fillna(0.0)
        benchmark_equity   = bh_df.sum(axis=1)

        # Total P&L across all tickers
        all_trade_pnl = pd.concat(
            [r["trade_pnl"] for r in self._ticker_results.values()
             if len(r["trade_pnl"]) > 0]
        ) if any(len(r["trade_pnl"]) > 0 for r in self._ticker_results.values()) else pd.Series(dtype=float)

        portfolio_metrics = full_metrics(portfolio_returns, portfolio_equity, all_trade_pnl)
        bh_metrics        = full_metrics(
            benchmark_equity.pct_change().fillna(0),
            benchmark_equity,
        )

        # ── Metrics comparison table ────────────────────────────────────────
        rows = {}
        for ticker, result in self._ticker_results.items():
            rows[ticker] = result["metrics"]
        rows["── PORTFOLIO ──"] = portfolio_metrics
        rows["── BUY & HOLD ──"] = bh_metrics
        metrics_table = pd.DataFrame(rows).T

        logger.info(
            "\nPORTFOLIO | Return: %+.1f%% | Sharpe: %.2f | MaxDD: %.1f%%",
            portfolio_metrics["Total Return (%)"],
            portfolio_metrics["Sharpe Ratio"],
            portfolio_metrics["Max Drawdown (%)"],
        )

        return {
            "ticker_results":   self._ticker_results,
            "portfolio_equity": portfolio_equity,
            "portfolio_returns":portfolio_returns,
            "portfolio_metrics":portfolio_metrics,
            "benchmark_equity": benchmark_equity,
            "bh_metrics":       bh_metrics,
            "metrics_table":    metrics_table,
            "equity_df":        equity_df,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_trade_pnl(
        close: pd.Series,
        positions: pd.Series,
    ) -> pd.Series:
        """
        Compute per-trade return (entry to exit) as a fraction.

        Returns a Series where each value is the return of one completed trade.
        """
        returns  = close.pct_change().fillna(0.0)
        pos_prev = positions.shift(1).fillna(0.0)

        entries = (positions > 0) & (pos_prev == 0)   # position opened
        exits   = (positions == 0) & (pos_prev > 0)   # position closed

        entry_dates = close.index[entries].tolist()
        exit_dates  = close.index[exits].tolist()

        trade_returns = []
        for entry in entry_dates:
            # Find the next exit after this entry
            future_exits = [e for e in exit_dates if e > entry]
            if not future_exits:
                # Position still open at end of period — use last close
                exit_date = close.index[-1]
            else:
                exit_date = future_exits[0]

            entry_price = close.loc[entry]
            exit_price  = close.loc[exit_date]
            trade_ret   = (exit_price / entry_price) - 1
            trade_returns.append(trade_ret)

        return pd.Series(trade_returns, dtype=float)


# ─── CLI convenience ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.data.loader import DataLoader
    from src.signals.generator import SignalGenerator

    loader  = DataLoader()
    ohlcv   = loader.load()
    sg      = SignalGenerator(ohlcv)
    signals = sg.generate_all()

    engine  = BacktestEngine(initial_capital=100_000, commission=0.001)
    results = engine.run_portfolio(signals)

    print("\n=== Metrics Table ===")
    print(results["metrics_table"][["Total Return (%)", "CAGR (%)", "Sharpe Ratio",
                                     "Max Drawdown (%)", "Ann. Volatility (%)"]].to_string())
