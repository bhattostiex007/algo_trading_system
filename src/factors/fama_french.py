"""
FamaFrench
==========
Downloads Fama-French 3-Factor (FF3) data and runs OLS regressions
to decompose strategy returns into systematic factor exposures + alpha.

The FF3 Model
-------------
R_strategy - Rf = α  +  β1·(Mkt-RF)  +  β2·SMB  +  β3·HML  +  ε

Where:
  α         = Jensen's alpha (excess return from skill, not factor exposure)
  Mkt-RF    = market excess return (systematic market risk)
  SMB       = Small Minus Big (size factor)
  HML       = High Minus Low (value factor)

Interview interpretation
------------------------
- High α (statistically significant) → strategy has genuine skill
- High β_Mkt → strategy is just levered market exposure
- Positive β_SMB → small-cap tilt; Negative → large-cap
- Positive β_HML → value tilt; Negative → growth/momentum

Usage
-----
from src.factors.fama_french import FamaFrenchAnalyzer

ff = FamaFrenchAnalyzer(start='2018-01-01', end='2024-12-31')
ff.download_factors()

result = ff.run_regression(strategy_returns, name='EMA Crossover')
print(result['summary'])
print(result['metrics'])

comparison = ff.compare_strategies(strategies_dict)
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
import pandas_datareader as pdr

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("FamaFrench")


class FamaFrenchAnalyzer:
    """
    Downloads FF3 factors and runs OLS regression on strategy returns.

    Parameters
    ----------
    start : str   Start date 'YYYY-MM-DD'
    end   : str   End date   'YYYY-MM-DD'
    """

    FF3_DATASET = "F-F_Research_Data_Factors_daily"

    def __init__(self, start: str = "2018-01-01", end: str = "2024-12-31") -> None:
        self.start = start
        self.end   = end
        self._factors: Optional[pd.DataFrame] = None   # Mkt-RF, SMB, HML, RF
        self._results: Dict[str, dict] = {}

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def download_factors(self) -> pd.DataFrame:
        """
        Download daily Fama-French 3 factors from Kenneth French's data library
        via pandas_datareader.

        Columns returned: Mkt-RF, SMB, HML, RF  (all as decimals, not %)
        """
        logger.info("Downloading FF3 factors from French data library …")
        try:
            raw = pdr.get_data_famafrench(
                self.FF3_DATASET,
                start=self.start,
                end=self.end,
            )
            # pandas_datareader returns a tuple; index 0 = daily data
            df = raw[0].copy()
            df.index = pd.to_datetime(df.index, format="%Y%m%d")
            df.index.name = "Date"

            # Convert from percent to decimal
            df = df / 100.0
            df.columns = [c.strip() for c in df.columns]

            # Filter to our date range
            df = df.loc[self.start : self.end]

            self._factors = df
            logger.info(
                "FF3 factors downloaded: %d rows | Columns: %s",
                len(df), list(df.columns),
            )
        except Exception as exc:
            logger.error("Failed to download FF3 data: %s", exc)
            logger.info("Falling back to synthetic factors …")
            self._factors = self._synthetic_factors()

        return self._factors

    def run_regression(
        self,
        strategy_returns: pd.Series,
        name: str = "Strategy",
        rf_col: str = "RF",
    ) -> dict:
        """
        Run OLS regression of strategy returns on FF3 factors.

        Parameters
        ----------
        strategy_returns : pd.Series  Daily strategy returns (not excess)
        name             : str        Label for this strategy
        rf_col           : str        Column name for risk-free rate in factors df

        Returns
        -------
        dict with: summary (str), metrics (dict), model (OLSResults), factors (df)
        """
        if self._factors is None:
            raise RuntimeError("Call download_factors() first.")

        # Align dates
        aligned = self._factors.join(
            strategy_returns.rename("Strategy"), how="inner"
        ).dropna()

        if len(aligned) < 100:
            logger.warning("%s: only %d aligned rows — results may be unreliable.",
                           name, len(aligned))

        # Excess strategy return = strategy_return - risk_free_rate
        excess_ret = aligned["Strategy"] - aligned[rf_col]

        # Factor matrix
        factors = aligned[["Mkt-RF", "SMB", "HML"]]
        X = sm.add_constant(factors)          # adds intercept (alpha)

        # OLS fit
        model = sm.OLS(excess_ret, X).fit(cov_type="HC3")  # robust std errors

        # Extract key stats
        alpha_daily    = model.params["const"]
        alpha_annual   = alpha_daily * 252
        beta_mkt       = model.params["Mkt-RF"]
        beta_smb       = model.params.get("SMB", np.nan)
        beta_hml       = model.params.get("HML", np.nan)
        r_squared      = model.rsquared
        adj_r_squared  = model.rsquared_adj
        t_alpha        = model.tvalues["const"]
        p_alpha        = model.pvalues["const"]
        information_ratio = alpha_annual / (excess_ret.std() * np.sqrt(252)) \
                           if excess_ret.std() > 0 else 0

        metrics = {
            "Strategy":           name,
            "Alpha (daily)":      round(alpha_daily, 6),
            "Alpha (annual %)":   round(alpha_annual * 100, 3),
            "t-stat (alpha)":     round(t_alpha, 3),
            "p-value (alpha)":    round(p_alpha, 4),
            "α significant":      "✓" if p_alpha < 0.05 else "✗",
            "β Market":           round(beta_mkt, 4),
            "β SMB (size)":       round(beta_smb, 4),
            "β HML (value)":      round(beta_hml, 4),
            "R²":                 round(r_squared, 4),
            "Adj. R²":            round(adj_r_squared, 4),
            "Info. Ratio":        round(information_ratio, 4),
            "Obs.":               len(aligned),
        }

        self._results[name] = {
            "model":   model,
            "metrics": metrics,
            "aligned": aligned,
            "excess_ret": excess_ret,
        }

        logger.info(
            "  %-20s | α_ann=%.3f%% | β_Mkt=%.3f | β_SMB=%.3f | β_HML=%.3f | R²=%.3f | p=%.4f",
            name, alpha_annual * 100, beta_mkt, beta_smb, beta_hml, r_squared, p_alpha,
        )

        return self._results[name]

    def compare_strategies(
        self,
        strategies: Dict[str, pd.Series],
    ) -> pd.DataFrame:
        """
        Run regression for multiple strategies and return a comparison DataFrame.

        Parameters
        ----------
        strategies : dict[name -> daily returns Series]
        """
        logger.info("Running FF3 regression for %d strategies …", len(strategies))
        rows = []
        for name, returns in strategies.items():
            result = self.run_regression(returns, name=name)
            rows.append(result["metrics"])

        comparison = pd.DataFrame(rows).set_index("Strategy")
        return comparison

    def get_factors(self) -> pd.DataFrame:
        """Return the downloaded factors DataFrame."""
        if self._factors is None:
            raise RuntimeError("Call download_factors() first.")
        return self._factors

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _synthetic_factors(self) -> pd.DataFrame:
        """
        Fallback: generate plausible synthetic FF3 factors if download fails.
        Values are calibrated to typical FF3 daily statistics.
        """
        logger.warning("Using SYNTHETIC FF3 factors — download failed.")
        np.random.seed(42)
        dates = pd.bdate_range(self.start, self.end)
        df = pd.DataFrame({
            "Mkt-RF": np.random.normal(0.0004, 0.0100, len(dates)),
            "SMB":    np.random.normal(0.0001, 0.0050, len(dates)),
            "HML":    np.random.normal(0.0000, 0.0050, len(dates)),
            "RF":     np.full(len(dates), 0.00015),   # ~4% annual / 252
        }, index=dates)
        df.index.name = "Date"
        return df.loc[self.start : self.end]


# ─── CLI convenience ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.data.loader import DataLoader
    from src.signals.generator import SignalGenerator
    from src.backtest.engine import BacktestEngine

    loader  = DataLoader()
    ohlcv   = loader.load()
    sg      = SignalGenerator(ohlcv)
    signals = sg.generate_all()
    engine  = BacktestEngine()
    results = engine.run_portfolio(signals)

    ff = FamaFrenchAnalyzer()
    ff.download_factors()

    port_returns = results["portfolio_returns"]
    result = ff.run_regression(port_returns, name="EMA Crossover Portfolio")
    print("\n=== FF3 Regression Results ===")
    for k, v in result["metrics"].items():
        print(f"  {k:<22}: {v}")
    print()
    print(result["model"].summary())
