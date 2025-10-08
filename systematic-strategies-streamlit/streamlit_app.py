import streamlit as st
import pandas as pd
import yfinance as yf
import requests

from src.data_loader import load_equities, load_crypto
from src.backtest import mean_reversion_strategy
from src.arbitrage import risk_arbitrage_signal
from src.risk_metrics import sharpe_ratio, sortino_ratio, max_drawdown, value_at_risk, conditional_var
from src.bayesian_eval import monte_carlo_metrics

# ------------------------
# Inline order book fetchers
# ------------------------
def fetch_order_book_yf(symbol="BTC-USD"):
    """Fallback order book from Yahoo Finance (only top bid/ask)."""
    ticker = yf.Ticker(symbol)
    info = ticker.info
    bids = pd.DataFrame([[info.get("bid"), info.get("bidSize")]], columns=["price","size"])
    asks = pd.DataFrame([[info.get("ask"), info.get("askSize")]], columns=["price","size"])
    return bids, asks

def fetch_order_book_coinbase(symbol="BTC-USD", depth=10):
    """Public Coinbase API order book (usually works on Streamlit Cloud)."""
    url = f"https://api.exchange.coinbase.com/products/{symbol}/book?level=2"
    resp = requests.get(url).json()
    bids = pd.DataFrame(resp.get("bids", [])[:depth], columns=["price","size","num_orders"])
    asks = pd.DataFrame(resp.get("asks", [])[:depth], columns=["price","size","num_orders"])
    return bids.astype(float), asks.astype(float)

# ------------------------
# Streamlit App
# ------------------------
st.title("Quantitative Trading Strategies Dashboard")

tabs = st.tabs(["Strategies", "Risk Metrics", "Monte Carlo", "Order Book"])

with tabs[0]:  # Strategies
    asset = st.sidebar.selectbox("Choose Asset", ["Equities", "Crypto"])
    if asset == "Equities":
        ticker = st.sidebar.selectbox(
            "Choose Equity Ticker",
            ["AAPL", "MSFT", "TSLA", "AMZN", "SPY", "QQQ", "GOOG", "NVDA"]
        )
        df = load_equities(ticker)

    else:  # Crypto
        symbol = st.sidebar.selectbox(
            "Choose Crypto Symbol",
            ["BTC-USD", "ETH-USD", "DOGE-USD", "SOL-USD", "BNB-USD"]
        )
        df = load_crypto(symbol)

    st.subheader(f"Data Preview: {asset}")
    st.dataframe(df.head())

    strategy = st.sidebar.selectbox("Choose Strategy", ["Mean Reversion", "Risk Arbitrage"])
    lookback = st.sidebar.slider("Lookback", 5, 30, 10)

    if strategy == "Mean Reversion":
        res = mean_reversion_strategy(df.copy(), lookback=lookback)
    else:
        res = risk_arbitrage_signal(df.copy())

    st.line_chart((1+res['strategy_returns']).cumprod())

with tabs[1]:  # Risk Metrics
    st.subheader("Risk Metrics")
    returns = res['strategy_returns'].dropna()
    st.write({
        "Sharpe": sharpe_ratio(returns),
        "Sortino": sortino_ratio(returns),
        "MaxDD": max_drawdown((1+returns).cumprod()),
        "VaR": value_at_risk(returns),
        "CVaR": conditional_var(returns)
    })

with tabs[2]:  # Monte Carlo
    st.subheader("Monte Carlo Stress Test")
    metrics = monte_carlo_metrics(returns, n_sim=200)
    df_mc = pd.DataFrame(metrics)
    if not df_mc.empty:
        st.line_chart(df_mc["sharpe"])
    else:
        st.info("Not enough return data yet.")

with tabs[3]:  # Order Book
    st.subheader("Live Order Book")
    symbol = st.text_input("Symbol (Coinbase style)", "BTC-USD")
    try:
        bids, asks = fetch_order_book_coinbase(symbol)
    except Exception:
        bids, asks = fetch_order_book_yf(symbol)

    st.write("Top Bids", bids)
    st.write("Top Asks", asks)

    if not bids.empty and not asks.empty:
        combined = pd.concat([
            bids.assign(side="bid"),
            asks.assign(side="ask")
        ])
        st.bar_chart(combined.set_index("price")["size"])
