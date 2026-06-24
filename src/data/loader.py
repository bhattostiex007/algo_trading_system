"""
DataLoader
==========
Pulls OHLCV data for a universe of tickers from yfinance,
cleans it, aligns all stocks to common trading dates, and
persists both per-ticker CSVs and a combined prices CSV.

Usage
-----
from src.data.loader import DataLoader

loader = DataLoader(
    tickers=["AAPL", "MSFT", "GOOGL"],
    start="2018-01-01",
    end="2024-12-31",
)
ohlcv = loader.load()          # dict[ticker -> DataFrame]
prices = loader.get_prices()   # DataFrame (date x ticker, Close prices)
loader.save()                  # persists to data/
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf
from tqdm import tqdm

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DataLoader")

# ─── Project root ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # algo-trading-system/
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


# ─── Constants ──────────────────────────────────────────────────────────────
DEFAULT_TICKERS: List[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN",
    "JPM",  "BAC",  "XOM",   "JNJ",
    "TSLA", "NVDA",
]

OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


class DataLoader:
    """
    Downloads and cleans OHLCV data for a list of tickers.

    Parameters
    ----------
    tickers : list of str
        Ticker symbols to download.
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str
        End date in 'YYYY-MM-DD' format.
    data_dir : Path, optional
        Directory to save CSV files.  Defaults to PROJECT_ROOT/data/.
    """

    def __init__(
        self,
        tickers: List[str] = DEFAULT_TICKERS,
        start: str = "2018-01-01",
        end: str = "2024-12-31",
        data_dir: Optional[Path] = None,
    ) -> None:
        self.tickers = [t.upper() for t in tickers]
        self.start = start
        self.end = end
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR

        # Populated by load()
        self._ohlcv: Dict[str, pd.DataFrame] = {}

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def load(self, force_download: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Load OHLCV data for all tickers.

        If a ticker's CSV already exists locally and force_download=False,
        it will be read from disk instead of hitting yfinance again.

        Returns
        -------
        dict[ticker -> cleaned OHLCV DataFrame]
        """
        logger.info(
            "Loading data for %d tickers | %s → %s",
            len(self.tickers), self.start, self.end,
        )

        for ticker in tqdm(self.tickers, desc="Downloading tickers"):
            csv_path = self.data_dir / f"ohlcv_{ticker}.csv"

            if csv_path.exists() and not force_download:
                logger.info("  %-6s — reading from cache: %s", ticker, csv_path)
                df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
            else:
                df = self._download(ticker)
                if df is None or df.empty:
                    logger.warning("  %-6s — no data returned, skipping.", ticker)
                    continue

            df = self._clean(df, ticker)
            if df is not None:
                self._ohlcv[ticker] = df

        logger.info("Loaded %d / %d tickers successfully.", len(self._ohlcv), len(self.tickers))
        return self._ohlcv

    def get_prices(self, price_col: str = "Close") -> pd.DataFrame:
        """
        Return a DataFrame of shape (trading_days, n_tickers)
        containing the specified price column for all loaded tickers.
        Columns are aligned to the common trading calendar (inner join).
        """
        if not self._ohlcv:
            raise RuntimeError("Call load() before get_prices().")

        frames = {
            ticker: df[price_col].rename(ticker)
            for ticker, df in self._ohlcv.items()
            if price_col in df.columns
        }
        prices = pd.concat(frames.values(), axis=1, join="inner")
        prices.index.name = "Date"
        logger.info(
            "Price matrix shape: %s  (trading days × tickers)", prices.shape
        )
        return prices

    def get_returns(self, price_col: str = "Close") -> pd.DataFrame:
        """Daily log-returns for all tickers."""
        prices = self.get_prices(price_col)
        returns = prices.pct_change().dropna()
        logger.info("Returns matrix shape: %s", returns.shape)
        return returns

    def save(self) -> None:
        """
        Persist per-ticker CSVs and a combined prices.csv to data_dir.
        """
        if not self._ohlcv:
            raise RuntimeError("Call load() before save().")

        for ticker, df in self._ohlcv.items():
            path = self.data_dir / f"ohlcv_{ticker}.csv"
            df.to_csv(path)
            logger.info("  Saved %s → %s", ticker, path)

        prices = self.get_prices()
        prices_path = self.data_dir / "prices.csv"
        prices.to_csv(prices_path)
        logger.info("Combined prices saved → %s", prices_path)

        returns = self.get_returns()
        returns_path = self.data_dir / "returns.csv"
        returns.to_csv(returns_path)
        logger.info("Daily returns saved  → %s", returns_path)

    def summary(self) -> pd.DataFrame:
        """
        Return a summary DataFrame with per-ticker data quality info.
        """
        rows = []
        for ticker, df in self._ohlcv.items():
            rows.append({
                "Ticker":     ticker,
                "Start":      df.index.min().date(),
                "End":        df.index.max().date(),
                "Rows":       len(df),
                "NaN_total":  df.isnull().sum().sum(),
                "Close_mean": round(df["Close"].mean(), 2),
                "Close_std":  round(df["Close"].std(), 2),
            })
        return pd.DataFrame(rows).set_index("Ticker")

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _download(self, ticker: str) -> Optional[pd.DataFrame]:
        """Download raw OHLCV from yfinance."""
        try:
            raw = yf.download(
                ticker,
                start=self.start,
                end=self.end,
                auto_adjust=True,      # Adj. for splits & dividends
                progress=False,
                threads=False,
            )
            if raw.empty:
                return None

            # yfinance sometimes returns MultiIndex columns
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(1)

            raw.index = pd.to_datetime(raw.index)
            raw.index.name = "Date"
            return raw[OHLCV_COLS]

        except Exception as exc:
            logger.error("  %-6s — download failed: %s", ticker, exc)
            return None

    def _clean(self, df: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
        """
        Clean a single ticker's OHLCV DataFrame:
          1. Keep only OHLCV columns
          2. Forward-fill up to 2 consecutive NaNs (weekend/holiday gaps)
          3. Drop rows that are still NaN after fill
          4. Filter to requested date range
          5. Drop duplicate dates (keep last)
        """
        # Select & reorder columns
        available = [c for c in OHLCV_COLS if c in df.columns]
        if len(available) < 4:
            logger.warning("  %-6s — insufficient columns: %s", ticker, available)
            return None

        df = df[available].copy()

        # Remove fully empty rows
        df = df.dropna(how="all")

        # Forward-fill short gaps (e.g., exchange holidays appear mid-range)
        df = df.ffill(limit=2)

        # Drop any remaining NaN rows
        before = len(df)
        df = df.dropna()
        if len(df) < before:
            logger.debug(
                "  %-6s — dropped %d NaN rows after ffill.", ticker, before - len(df)
            )

        # Filter to requested range
        df = df.loc[self.start : self.end]

        # Drop duplicate dates
        df = df[~df.index.duplicated(keep="last")]

        # Sort chronologically
        df = df.sort_index()

        if df.empty:
            logger.warning("  %-6s — empty after cleaning.", ticker)
            return None

        return df


# ─── CLI convenience ────────────────────────────────────────────────────────
if __name__ == "__main__":
    loader = DataLoader()
    loader.load()
    loader.save()
    print("\n=== Data Summary ===")
    print(loader.summary())
