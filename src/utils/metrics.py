"""
metrics.py
==========
Standalone performance metric functions for evaluating strategy returns.

All functions accept a pd.Series of daily returns or an equity curve.

Functions
---------
compute_sharpe(returns, rf, periods)    → float
compute_sortino(returns, rf, periods)   → float
compute_max_drawdown(equity)            → float
compute_cagr(equity, periods)           → float
compute_calmar(cagr, max_dd)            → float
compute_win_rate(trade_returns)         → float
compute_profit_factor(trade_returns)    → float
full_metrics(returns, equity, trades)   → dict
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS = 252


# ─── Individual metric functions ─────────────────────────────────────────────

def compute_sharpe(
    returns: pd.Series,
    rf: float = 0.0,
    periods: int = TRADING_DAYS,
) -> float:
    """
    Annualised Sharpe Ratio.

    Parameters
    ----------
    returns : pd.Series  Daily net returns (after costs)
    rf      : float      Daily risk-free rate (default 0 for simplicity)
    periods : int        Trading days per year
    """
    excess = returns - rf
    if excess.std() == 0:
        return 0.0
    return float((excess.mean() / excess.std()) * np.sqrt(periods))


def compute_sortino(
    returns: pd.Series,
    rf: float = 0.0,
    periods: int = TRADING_DAYS,
) -> float:
    """
    Annualised Sortino Ratio — penalises only downside volatility.
    Better than Sharpe for skewed return distributions.
    """
    excess = returns - rf
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float((excess.mean() / downside.std()) * np.sqrt(periods))


def compute_max_drawdown(equity: pd.Series) -> float:
    """
    Maximum Drawdown as a positive percentage (e.g. 0.25 = 25% drawdown).

    Definition: largest peak-to-trough decline in the equity curve.
    """
    rolling_peak = equity.cummax()
    drawdown     = (equity - rolling_peak) / rolling_peak
    return float(abs(drawdown.min()))


def compute_drawdown_series(equity: pd.Series) -> pd.Series:
    """Returns the full drawdown series (values ≤ 0)."""
    rolling_peak = equity.cummax()
    return (equity - rolling_peak) / rolling_peak


def compute_cagr(
    equity: pd.Series,
    periods: int = TRADING_DAYS,
) -> float:
    """
    Compound Annual Growth Rate.

    CAGR = (end_value / start_value)^(1 / n_years) - 1
    """
    n_years = len(equity) / periods
    if n_years <= 0 or equity.iloc[0] <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1)


def compute_calmar(cagr: float, max_drawdown: float) -> float:
    """Calmar Ratio = CAGR / Max Drawdown."""
    if max_drawdown == 0:
        return 0.0
    return float(cagr / max_drawdown)


def compute_win_rate(trade_pnl: pd.Series) -> float:
    """Fraction of trades that were profitable."""
    if len(trade_pnl) == 0:
        return 0.0
    return float((trade_pnl > 0).sum() / len(trade_pnl))


def compute_profit_factor(trade_pnl: pd.Series) -> float:
    """
    Profit Factor = gross_profit / gross_loss.
    > 1.0 means the strategy made more than it lost.
    """
    gross_profit = trade_pnl[trade_pnl > 0].sum()
    gross_loss   = abs(trade_pnl[trade_pnl < 0].sum())
    if gross_loss == 0:
        return float('inf')
    return float(gross_profit / gross_loss)


def compute_volatility(
    returns: pd.Series,
    periods: int = TRADING_DAYS,
) -> float:
    """Annualised standard deviation of daily returns."""
    return float(returns.std() * np.sqrt(periods))


# ─── Combined metrics dict ────────────────────────────────────────────────────

def full_metrics(
    returns: pd.Series,
    equity: pd.Series,
    trade_pnl: pd.Series | None = None,
    rf: float = 0.0,
) -> dict:
    """
    Return a comprehensive performance metrics dictionary.

    Parameters
    ----------
    returns   : daily net returns series
    equity    : equity curve series (dollar values)
    trade_pnl : per-trade P&L series (optional — for win rate / profit factor)
    rf        : daily risk-free rate
    """
    cagr    = compute_cagr(equity)
    max_dd  = compute_max_drawdown(equity)
    sharpe  = compute_sharpe(returns, rf)
    sortino = compute_sortino(returns, rf)
    vol     = compute_volatility(returns)
    calmar  = compute_calmar(cagr, max_dd)

    metrics = {
        "Total Return (%)":   round((equity.iloc[-1] / equity.iloc[0] - 1) * 100, 2),
        "CAGR (%)":           round(cagr * 100, 2),
        "Sharpe Ratio":       round(sharpe, 3),
        "Sortino Ratio":      round(sortino, 3),
        "Calmar Ratio":       round(calmar, 3),
        "Max Drawdown (%)":   round(max_dd * 100, 2),
        "Ann. Volatility (%)":round(vol * 100, 2),
        "Start Value ($)":    round(equity.iloc[0], 2),
        "End Value ($)":      round(equity.iloc[-1], 2),
    }

    if trade_pnl is not None and len(trade_pnl) > 0:
        metrics["Num Trades"]    = len(trade_pnl)
        metrics["Win Rate (%)"]  = round(compute_win_rate(trade_pnl) * 100, 1)
        metrics["Profit Factor"] = round(compute_profit_factor(trade_pnl), 3)
        metrics["Avg Trade (%)"] = round(trade_pnl.mean() * 100, 3)

    return metrics
