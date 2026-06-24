<div align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue.svg" alt="Python Version"/>
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B.svg" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Finance-Quant-green.svg" alt="Quant"/>
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED.svg" alt="Docker"/>
</div>

# Quantitative Trading System & Portfolio Optimizer

An end-to-end algorithmic trading and portfolio optimization engine built in Python. This system downloads live market data, generates vectorised technical trading signals, executes a historically accurate backtest, attributes performance using the **Fama-French 3-Factor Model**, and optimizes capital allocation using **Markowitz Mean-Variance Optimization** and the **Black-Litterman Model**.

🚀 **[View Live Web Dashboard](https://algo-trading-dashboard-m36t.onrender.com)** 🚀

---

## 📈 Key Features

1. **Vectorised Backtesting Engine (`src/backtest/engine.py`)**
   - High-performance, Pandas-native backtesting architecture.
   - Built-in prevention of lookahead bias (Signal generated on T-Close, Position entered at T+1 Open).
   - Accurately models transaction costs (slippage/commission).

2. **Trend Following & Mean Reversion Signals (`src/signals/generator.py`)**
   - **Main Strategy:** EMA Golden/Death Cross combined with RSI overbought/oversold filtering.
   - Outputs discrete `+1` (Long) and `-1` (Short) signals across multiple assets.

3. **Fama-French 3-Factor Attribution (`src/factors/fama_french.py`)**
   - Decomposes strategy returns via OLS regression with HC3 robust standard errors.
   - Identifies genuine skill (Alpha, $\alpha$) versus systematic exposure to Market, Size (SMB), and Value (HML) factors.

4. **Advanced Portfolio Optimization (`src/portfolio/markowitz.py` & `black_litterman.py`)**
   - **Markowitz MVO:** Extracts the Efficient Frontier and Tangency Portfolio using Ledoit-Wolf shrinkage to regularize the covariance matrix (reducing estimation error).
   - **Black-Litterman Model:** Overcomes Markowitz's "error-maximizing" problem by blending market-implied prior returns with subjective investor views using Bayesian updating, yielding highly stable and tradable weights.

5. **Out-Of-Sample (OOS) Walk-Forward Testing**
   - In-Sample Training (2018–2022) for weight optimization.
   - Out-of-Sample Testing (2023–2024) to prove robustness against overfitting and data leakage.

6. **Live Execution Hook (`run_live.py`)**
   - Command-line script leveraging `yfinance` to fetch real-time market data and generate actionable, day-of trading signals.

---

## 📊 Live Cloud Application

The project is fully deployed to the cloud via Docker and Render. It features an interactive Streamlit web application allowing users to visualize equity curves, Fama-French metrics, and Black-Litterman allocations without interacting with the source code.

**🌐 Access the Live App:** [https://algo-trading-dashboard-m36t.onrender.com](https://algo-trading-dashboard-m36t.onrender.com)

---

## 🏗 System Architecture

```text
algo-trading-system/
│
├── src/
│   ├── data/
│   │   └── loader.py             # yfinance OHLCV extraction & caching
│   ├── signals/
│   │   └── generator.py          # Technical indicator logic (EMA, RSI, MACD)
│   ├── backtest/
│   │   └── engine.py             # Vectorised, zero-lookahead PnL simulator
│   ├── factors/
│   │   └── fama_french.py        # OLS regression & Alpha extraction
│   └── portfolio/
│       ├── markowitz.py          # PyPortfolioOpt Efficient Frontier
│       └── black_litterman.py    # Market Priors + Subjective Views
│
├── notebooks/                    # Jupyter notebooks mapping the 5-day build process
│   ├── day1_data_pipeline.ipynb
│   ├── day2_signals.ipynb
│   ├── day3_backtest.ipynb
│   ├── day4_factor_attribution.ipynb
│   └── day5_portfolio_optimization.ipynb
│
├── app.py                        # Streamlit Dashboard Entrypoint
├── run_live.py                   # Live terminal execution hook
├── Dockerfile                    # Containerization instructions
└── render.yaml                   # IaC Blueprint for cloud deployment
```

---

## 🧠 The Math: Black-Litterman Explained

While Markowitz Optimization is mathematically elegant, it is highly sensitive to the expected returns ($\mu$) inputted. 

This system implements **Black-Litterman** to solve this. It assumes the market portfolio is efficient, reverses the optimization to find the **Market-Implied Prior Returns** ($\Pi$), and updates them using the investor's subjective views ($Q$) and a confidence matrix ($\Omega$).

$$ E[R] = [(\tau \Sigma)^{-1} + P^T \Omega^{-1} P]^{-1} [(\tau \Sigma)^{-1} \Pi + P^T \Omega^{-1} Q] $$

The result is a posterior distribution of returns that leads to far more stable, intuitive, and less "cornered" portfolio weights than pure historical MVO.

---

## 📝 License
This project is for educational and portfolio demonstration purposes. Not financial advice. MIT License.
