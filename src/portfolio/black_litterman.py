"""
BlackLittermanOptimizer
=======================
Combines market equilibrium priors with subjective investor views to produce
stable, diversified expected returns (and optionally, weights).

Key concepts (interview-ready)
-------------------------------
- **Market Implied Returns (Prior)**: Extracted from market cap weights via
  reverse-optimization. "If the market is efficient, what returns is it expecting?"
- **Investor Views**: Absolute or relative views. E.g., "AAPL will return 12%",
  or "NVDA will outperform TSLA by 5%".
- **Omega (Confidence matrix)**: Uncertainty in the views. If highly confident,
  Omega is small, and BL returns pull strongly toward views.
- **Why use BL?** Markowitz MVO is notoriously sensitive to expected returns.
  Small changes in inputs → wildly different allocations. BL anchors the inputs
  to market reality, making MVO practically usable.

Usage
-----
from src.portfolio.black_litterman import BlackLittermanOptimizer

bl = BlackLittermanOptimizer(prices, mcap_weights)
view_dict = {"AAPL": 0.12, "MSFT": 0.10}  # Absolute views

# Compute BL expected returns
bl_returns = bl.compute_bl_returns(view_dict, confidences=[0.8, 0.6])

# Use them in Markowitz MVO
from src.portfolio.markowitz import MarkowitzOptimizer
opt = MarkowitzOptimizer(prices)
opt.fit()
opt.mu = bl_returns  # Override with BL returns
weights = opt.max_sharpe()
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from pypfopt import black_litterman, risk_models, expected_returns

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("BlackLitterman")


class BlackLittermanOptimizer:
    """
    Wrapper for PyPortfolioOpt's Black-Litterman model.

    Parameters
    ----------
    prices       : pd.DataFrame  Daily close prices (T, N)
    mcap_weights : dict          Market capitalization weights (ticker -> float)
                                 Must sum to ~1.0
    frequency    : int           Trading days per year
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        mcap_weights: Dict[str, float],
        frequency: int = 252,
    ) -> None:
        self.prices    = prices
        self.frequency = frequency
        self.tickers   = list(prices.columns)

        # Align weights with columns
        w_series = pd.Series(mcap_weights).reindex(self.tickers).fillna(0.0)
        self.market_weights = w_series / w_series.sum()

        # Compute shrunk covariance matrix for the BL prior
        self.S = risk_models.CovarianceShrinkage(
            prices, frequency=self.frequency
        ).ledoit_wolf()

        # Market implied risk aversion parameter (delta)
        # Using a standard heuristic: ~2.5 for broad equities
        self.delta = 2.5
        logger.info("Market-implied risk aversion (delta): %.3f", self.delta)

        # Market Prior expected returns (Pi)
        self.prior_returns = black_litterman.market_implied_prior_returns(
            self.market_weights, self.delta, self.S
        )

        # Populated after computing
        self.bl_returns: Optional[pd.Series] = None
        self.bl_cov:     Optional[pd.DataFrame] = None

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def compute_bl_returns(
        self,
        absolute_views: Dict[str, float],
        confidences: Optional[List[float]] = None,
        tau: float = 0.05,
    ) -> pd.Series:
        """
        Combine market prior with subjective views to get BL expected returns.

        Parameters
        ----------
        absolute_views : dict[ticker -> expected annual return]
        confidences    : list of confidence levels [0, 1] matching views
                         If None, uses uncertainty proportional to asset variance.
        tau            : scalar indicating uncertainty of the prior (usually 0.02 - 0.05)

        Returns
        -------
        pd.Series of Black-Litterman expected returns
        """
        logger.info("Computing Black-Litterman posterior returns …")

        bl_model = black_litterman.BlackLittermanModel(
            self.S,
            pi=self.prior_returns,
            absolute_views=absolute_views,
            omega="idzorek" if confidences else "default",
            view_confidences=confidences,
            tau=tau,
        )

        self.bl_returns = bl_model.bl_returns()
        self.bl_cov     = bl_model.bl_cov()

        logger.info(
            "Highest BL Returns: %s",
            self.bl_returns.nlargest(3).round(4).to_dict()
        )
        return self.bl_returns

    def get_bl_weights(self) -> Dict[str, float]:
        """
        Return the BL optimal portfolio weights directly derived from the model.
        (Alternatively, you can pass self.bl_returns into MarkowitzOptimizer).
        """
        if self.bl_returns is None:
            raise RuntimeError("Call compute_bl_returns() first.")

        bl_model = black_litterman.BlackLittermanModel(
            self.S, pi=self.prior_returns, tau=0.05
        )
        # Hack to re-init with computed returns
        bl_model.posterior_rets = self.bl_returns
        bl_model.posterior_cov  = self.bl_cov

        weights = bl_model.bl_weights(risk_aversion=self.delta)
        # Clean weights (round to 4 decimals, drop tiny values)
        clean_weights = {k: round(v, 4) for k, v in weights.items() if abs(v) > 1e-4}
        return clean_weights

    def get_prior_returns(self) -> pd.Series:
        """Return the market-implied expected returns (Pi)."""
        return self.prior_returns

    def get_posterior_cov(self) -> pd.DataFrame:
        """Return the BL updated covariance matrix."""
        if self.bl_cov is None:
            raise RuntimeError("Call compute_bl_returns() first.")
        return self.bl_cov


# ─── CLI convenience ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.data.loader import DataLoader
    from src.portfolio.markowitz import MarkowitzOptimizer

    loader = DataLoader()
    prices = loader.load()["Close"]

    # Dummy market cap weights (equal for testing)
    mcaps = {t: 1.0 for t in prices.columns}

    bl = BlackLittermanOptimizer(prices, mcaps)
    print("\n=== Market Prior Returns (Implied by weights) ===")
    print((bl.get_prior_returns() * 100).round(2))

    # Suppose we have a strong bullish view on AAPL and NVDA
    views = {"AAPL": 0.25, "NVDA": 0.40}
    conf  = [0.8, 0.9]

    bl_rets = bl.compute_bl_returns(views, confidences=conf)
    print("\n=== Black-Litterman Posterior Returns ===")
    print((bl_rets * 100).round(2))

    # Plug into MVO
    opt = MarkowitzOptimizer(prices)
    opt.fit()
    opt.mu = bl_rets  # override
    opt.S  = bl.get_posterior_cov()
    w = opt.max_sharpe()
    print("\n=== BL-MVO Tangency Portfolio Weights ===")
    for k, v in w.items():
        if v > 0.01:
            print(f"  {k:5s}: {v*100:5.1f}%")
