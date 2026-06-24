import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Algo Trading System", layout="wide", page_icon="📈")

# --- Custom CSS for Glow Effects and Roman Fonts ---
st.markdown("""
<style>
    /* Roman Serif Font for Headings */
    h1, h2, h3 {
        font-family: "Times New Roman", Times, serif !important;
        text-shadow: 0px 0px 10px rgba(163, 44, 196, 0.4);
    }
    
    /* Glowing Gradient effects on buttons and interactive elements */
    .stButton>button {
        border: 1px solid #a32cc4;
        background: transparent;
        color: white;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, rgba(163,44,196,1) 0%, rgba(88,14,142,1) 100%);
        box-shadow: 0px 0px 15px 2px rgba(163,44,196,0.6);
        border: 1px solid transparent;
        transform: scale(1.02);
    }
    
    /* Glow effect on the sidebar navigation active item */
    .st-emotion-cache-17lntkn {
        transition: all 0.3s ease;
    }
    .st-emotion-cache-17lntkn:hover {
        box-shadow: 0px 0px 12px 1px rgba(163, 44, 196, 0.3);
        border-radius: 8px;
    }
    
    /* Global subtle gradient animation on the top bar */
    header {
        background: linear-gradient(90deg, #000000, #1a052b, #000000) !important;
        background-size: 200% 200% !important;
        animation: gradientBG 5s ease infinite !important;
    }
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* DataFrame styling overrides */
    .stDataFrame {
        border: 1px solid #333;
        border-radius: 8px;
        box-shadow: 0px 0px 15px rgba(163,44,196,0.15);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Data Loading
# ==========================================
@st.cache_data
def load_data():
    try:
        # Load from the results we saved in Day 4/5
        comp_df = pd.read_csv("data/day4_strategy_comparison.csv", index_col="Strategy")
        ff_factors = pd.read_csv("data/ff3_factors.csv", index_col="Date")
        ff_factors.index = pd.to_datetime(ff_factors.index)
        weights_df = pd.read_csv("data/portfolio/target_weights.csv", index_col=0)
        return comp_df, ff_factors, weights_df
    except Exception as e:
        return None, None, None

comp_df, ff_factors, weights_df = load_data()

# ==========================================
# UI Layout
# ==========================================
st.title("📈 Quantitative Trading System")
st.markdown("Interactive dashboard for Strategy Performance, Factor Attribution, and Portfolio Optimization.")

if comp_df is None:
    st.error("Data files not found. Please run the notebooks (Days 1-5) first to generate the required CSV files in `data/`.")
    st.stop()

# --- Sidebar ---
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", ["Strategy Performance", "Factor Attribution", "Portfolio Optimization"])

st.sidebar.markdown("---")
st.sidebar.info(
    "**System Architecture:**\n"
    "- Signals: EMA Crossovers, RSI\n"
    "- Factors: Fama-French 3-Factor (OLS)\n"
    "- Optimization: Markowitz MVO + Black-Litterman\n"
    "- Tech Stack: Pandas, PyPortfolioOpt, Statsmodels"
)

# ==========================================
# Page: Strategy Performance
# ==========================================
if page == "Strategy Performance":
    st.header("Multi-Strategy Performance Comparison (2018-2024)")
    
    st.dataframe(
        comp_df.style.format({
            "Total Return (%)": "{:+.1f}%",
            "CAGR (%)": "{:.2f}%",
            "Sharpe": "{:.2f}",
            "Sortino": "{:.2f}",
            "Max Drawdown (%)": "{:.1f}%",
            "Ann. Vol (%)": "{:.1f}%"
        }).background_gradient(subset=["Sharpe", "Total Return (%)"], cmap="Purples")
          .background_gradient(subset=["Max Drawdown (%)"], cmap="magma"),
        use_container_width=True,
        height=250
    )

    st.markdown("### Key Takeaways")
    st.markdown("""
    - **Trend Following (EMA Crossover)** spends ~80% of its time in cash, severely limiting drawdowns (6.9%) compared to Buy & Hold (48.6%).
    - Capital preservation is prioritized over total return.
    - To increase total return while maintaining high Sharpe, we blend uncorrelated strategies or use **Markowitz Optimization** (see Portfolio Optimization page).
    """)

# ==========================================
# Page: Factor Attribution
# ==========================================
elif page == "Factor Attribution":
    st.header("Fama-French 3-Factor (FF3) Attribution")
    st.markdown("""
    Decomposing returns into systematic market exposures vs. genuine skill (Alpha).
    $$R_{strategy} - R_f = \\alpha + \\beta_{Mkt} \\cdot (Mkt\\text{-}RF) + \\beta_{SMB} \\cdot SMB + \\beta_{HML} \\cdot HML$$
    """)

    st.subheader("Cumulative Factor Performance")
    # Cumulative returns
    cum_factors = (1 + ff_factors).cumprod()
    
    fig = go.Figure()
    for col in cum_factors.columns:
        if col != 'RF':
            fig.add_trace(go.Scatter(x=cum_factors.index, y=cum_factors[col], name=col))
    
    fig.update_layout(
        height=400, 
        template="plotly_dark", 
        yaxis_title="Cumulative Return",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# Page: Portfolio Optimization
# ==========================================
elif page == "Portfolio Optimization":
    st.header("Target Portfolio Allocation")
    st.markdown("Comparing Equal-Weight naive allocation vs. Mathematical Optimization.")
    
    st.dataframe(
        weights_df.style.format("{:.1f}%").background_gradient(cmap="Purples"),
        use_container_width=True
    )
    
    st.markdown("### Why Black-Litterman?")
    st.markdown("""
    - Pure **Historical MVO** (Max Sharpe) is "error-maximizing" and tends to place extreme 100% allocations on 1 or 2 assets (e.g. AAPL, NVDA) while zeroing out everything else.
    - **Black-Litterman (BL-MVO)** reverses market cap weights to establish a stable prior, then applies subjective views. The result is a much more balanced, diversified, and practically tradable portfolio.
    """)
