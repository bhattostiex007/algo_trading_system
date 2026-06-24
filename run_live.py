"""
Live Execution Hook for Algo Trading System

Downloads the latest market data (up to the current minute),
computes technical signals (EMA crossovers, RSI), and outputs
the current signal for today.
"""

import warnings
import pandas as pd
import yfinance as yf
from datetime import datetime
from rich.console import Console
from rich.table import Table

# Suppress warnings for clean terminal output
warnings.filterwarnings('ignore')

from src.signals.generator import SignalGenerator

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "BAC", "XOM", "JNJ", "TSLA", "NVDA"]

def main():
    console = Console()
    console.print(f"\n[bold blue]=== Algo Trading System | Live Execution Hook ===[/bold blue]")
    console.print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    console.print(f"Fetching latest data for {len(TICKERS)} tickers...\n")

    # Download recent data (last 150 days is enough to calculate EMA50 and ATR safely)
    data = yf.download(TICKERS, period="150d", progress=False, group_by="ticker")
    
    # Restructure like our DataLoader outputs
    ohlcv_dict = {}
    for ticker in TICKERS:
        df = data[ticker].copy()
        df.columns = [c.capitalize() for c in df.columns]  # open -> Open
        if "Adj close" in df.columns:
            df = df.rename(columns={"Adj close": "Adj Close"})
        ohlcv_dict[ticker] = df.dropna()

    # Generate signals
    sg = SignalGenerator(ohlcv_dict)
    signals = sg.generate_all()

    # Build terminal table
    table = Table(title="Today's Live Trading Signals", title_style="bold magenta")
    table.add_column("Ticker", style="cyan", no_wrap=True)
    table.add_column("Last Close ($)", justify="right")
    table.add_column("EMA 20", justify="right")
    table.add_column("EMA 50", justify="right")
    table.add_column("RSI", justify="right")
    table.add_column("SIGNAL", justify="center", style="bold")

    action_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}

    for ticker in TICKERS:
        df = signals[ticker]
        latest = df.iloc[-1]
        
        sig = int(latest["Signal"])
        if sig == 1:
            sig_str = "[bold green]BUY[/bold green]"
            action_counts["BUY"] += 1
        elif sig == -1:
            sig_str = "[bold red]SELL[/bold red]"
            action_counts["SELL"] += 1
        else:
            sig_str = "[white]HOLD[/white]"
            action_counts["HOLD"] += 1

        table.add_row(
            ticker,
            f"{latest['Close']:.2f}",
            f"{latest['EMA_20']:.2f}",
            f"{latest['EMA_50']:.2f}",
            f"{latest['RSI']:.1f}",
            sig_str
        )

    console.print(table)
    
    # Summary
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  [green]BUY:[/green]  {action_counts['BUY']}")
    console.print(f"  [red]SELL:[/red] {action_counts['SELL']}")
    console.print(f"  HOLD: {action_counts['HOLD']}")
    console.print("\n[dim]Note: Orders should be placed at the market open on the next trading day.[/dim]\n")

if __name__ == "__main__":
    main()
