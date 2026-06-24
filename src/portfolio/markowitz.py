"""
MarkowitzOptimizer
==================
Mean-Variance Optimization (MVO) using PyPortfolioOpt.

Key concepts (interview-ready)
-------------------------------
- **Efficient Frontier**: Set of portfolios with maximum return for a given risk.
- **Max Sharpe**: Tangency portfolio — highest return per unit of risk.
- **Min Volatility**: Minimum risk portfolio (leftmost point on frontier).
- **Ledoit-Wolf shrinkage**: Regularises the sample covariance matrix to reduce
  estimation error — critical because sample covariance is unstable with few assets.
- **Why BL beats pure MVO**: MVO is sensitive to small changes in expected returns
  (garbage-in-garbage-out). BL regularises inputs using market priors.

Usage
-----
from src.portfolio.markowitz import MarkowitzOptimizer

opt = MarkowitzOptimizer(prices)
opt.fit()

mvo_weights   = opt.max_sharpe(risk_free_rate=0.04)
minvol_weights = opt.min_volatility()

print(opt.portfolio_performance(mvo_weights))
frontier_df   = opt.efficient_frontier_points(n_points=100)
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.exceptions import OptimizationError

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MarkowitzOptimizer")


class MarkowitzOptimizer:
    """
    Wraps PyPortfolioOpt's EfficientFrontier for clean MVO workflows.

    Parameters
    ----------
    prices : pd.DataFrame  Shape (T, N) — daily close prices, columns = tickers
    frequency : int        Trading days per year (default 252)
    """

    def __init__(self, prices: pd.DataFrame, frequency: int = 252) -> None:
        self.prices    = prices
        self.frequency = frequency
        self.tickers   = list(prices.columns)

        # Set after fit()
        self.mu: Optional[pd.Series]     = None   # Expected annual returns
        self.S:  Optional[pd.DataFrame]  = None   # Shrunk covariance matrix

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def fit(self, returns_method: str = "mean_historical") -> "MarkowitzOptimizer":
        """
        Compute expected returns and covariance matrix.

        Parameters
        ----------
        returns_method : 'mean_historical' | 'ema_historical' | 'capm'
        """
        logger.info("Fitting Markowitz optimizer …")

        # ── Expected returns ──────────────────────────────────────────────
        if returns_method == "ema_historical":
            self.mu = expected_returns.ema_historical_return(
                self.prices, frequency=self.frequency
            )
        elif returns_method == "capm":
            self.mu = expected_returns.capm_return(
                self.prices, frequency=self.frequency
            )
        else:  # mean_historical (default)
            self.mu = expected_returns.mean_historical_return(
                self.prices, frequency=self.frequency
            )

        # ── Covariance matrix — Ledoit-Wolf shrinkage ─────────────────────
        # Ledoit-Wolf reduces estimation error vs sample covariance
        # (sample covariance is notoriously unstable with <30 assets × years of data)
        self.S = risk_models.CovarianceShrinkage(
            self.prices, frequency=self.frequency
        ).ledoit_wolf()

        logger.info("Expected returns (top 3): %s",
                    self.mu.nlargest(3).round(4).to_dict())
        logger.info("Covariance matrix shape: %s", self.S.shape)
        return self

    def max_sharpe(
        self,
        risk_free_rate: float = 0.04,
        weight_bounds: tuple  = (0.0, 0.40),
    ) -> Dict[str, float]:
        """
        Find the tangency portfolio — highest Sharpe ratio.

        Parameters
        ----------
        risk_free_rate : float  Annual risk-free rate (default 4% for US 2024)
        weight_bounds  : tuple  (min, max) weight per asset
        """
        self._check_fitted()
        ef = EfficientFrontier(self.mu, self.S, weight_bounds=weight_bounds)
        ef.max_sharpe(risk_free_rate=risk_free_rate)
        weights = ef.clean_weights()

        perf = ef.portfolio_performance(
            risk_free_rate=risk_free_rate, verbose=False
        )
        logger.info(
            "Max Sharpe: Return=%.2f%%  Vol=%.2f%%  Sharpe=%.3f",
            perf[0]*100, perf[1]*100, perf[2]
        )
        return dict(weights)

    def min_volatility(
        self,
        weight_bounds: tuple = (0.0, 0.40),
    ) -> Dict[str, float]:
        """Find the minimum variance portfolio."""
        self._check_fitted()
        ef = EfficientFrontier(self.mu, self.S, weight_bounds=weight_bounds)
        ef.min_volatility()
        weights = ef.clean_weights()

        perf = ef.portfolio_performance(verbose=False)
        logger.info(
            "Min Volatility: Return=%.2f%%  Vol=%.2f%%  Sharpe=%.3f",
            perf[0]*100, perf[1]*100, perf[2]
        )
        return dict(weights)

    def efficient_return(
        self,
        target_return: float,
        weight_bounds: tuple = (0.0, 0.40),
    ) -> Dict[str, float]:
        """Portfolio with minimum risk for a given target return."""
        self._check_fitted()
        ef = EfficientFrontier(self.mu, self.S, weight_bounds=weight_bounds)
        ef.efficient_return(target_return)
        return dict(ef.clean_weights())

    def portfolio_performance(
        self,
        weights: Dict[str, float],
        risk_free_rate: float = 0.04,
    ) -> dict:
        """
        Compute expected annual return, volatility, and Sharpe for given weights.
        """
        self._check_fitted()
        w = pd.Series(weights).reindex(self.tickers).fillna(0.0)
        ann_return = float(w @ self.mu)
        ann_vol    = float(np.sqrt(w @ self.S @ w))
        sharpe     = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0
        return {
            "Expected Return (%)": round(ann_return * 100, 2),
            "Expected Volatility (%)": round(ann_vol * 100, 2),
            "Sharpe Ratio": round(sharpe, 3),
        }

    def efficient_frontier_points(
        self,
        n_points: int = 80,
        risk_free_rate: float = 0.04,
    ) -> pd.DataFrame:
        """
        Generate a grid of (risk, return) points along the efficient frontier.
        Useful for plotting the frontier curve.
        """
        self._check_fitted()

        min_ret = float(self.mu.min()) * 0.9
        max_ret = float(self.mu.max()) * 0.95
        target_returns = np.linspace(min_ret, max_ret, n_points)

        frontier = []
        for target in target_returns:
            try:
                ef = EfficientFrontier(self.mu, self.S, weight_bounds=(0.0, 0.40))
                ef.efficient_return(target)
                ret, vol, sharpe = ef.portfolio_performance(
                    risk_free_rate=risk_free_rate, verbose=False
                )
                frontier.append({
                    "Return (%)": ret * 100,
                    "Volatility (%)": vol * 100,
                    "Sharpe": sharpe,
                })
            except (OptimizationError, Exception):
                pass

        return pd.DataFrame(frontier)

    def equal_weights(self) -> Dict[str, float]:
        """Naive equal-weight benchmark."""
        n = len(self.tickers)
        return {t: round(1.0 / n, 6) for t in self.tickers}

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if self.mu is None or self.S is None:
            raise RuntimeError("Call fit() before running optimizations.")
