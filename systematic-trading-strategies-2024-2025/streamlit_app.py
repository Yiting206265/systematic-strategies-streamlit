import streamlit as st
import pandas as pd
from src import data_loader, backtest, risk_metrics, bayesian_eval, arbitrage

st.title("📈 Systematic Trading Strategies 2024-2025")

asset = st.sidebar.selectbox("Choose Asset", ["Equities", "Crypto"])

if asset == "Equities":
    ticker = st.sidebar.text_input("Equity Ticker", "AAPL")
    df = data_loader.load_equities(ticker)
else:
    symbol = st.sidebar.text_input("Crypto Symbol", "BTC/USDT")
    df = data_loader.load_crypto(symbol)

st.subheader(f"Data Preview: {asset}")
st.dataframe(df.head())

strategy = st.sidebar.selectbox("Choose Strategy", ["Mean Reversion", "Risk Arbitrage"])
lookback = st.sidebar.slider("Lookback", 5, 30, 10)

if strategy == "Mean Reversion":
    res = backtest.mean_reversion_strategy(df.copy(), lookback=lookback)
else:
    res = arbitrage.risk_arbitrage_signal(df.copy())

st.line_chart((1+res['strategy_returns']).cumprod())

st.subheader("Risk Metrics")
returns = res['strategy_returns'].dropna()
st.write({
    "Sharpe": risk_metrics.sharpe_ratio(returns),
    "Sortino": risk_metrics.sortino_ratio(returns),
    "MaxDD": risk_metrics.max_drawdown((1+returns).cumprod()),
    "VaR": risk_metrics.value_at_risk(returns),
    "CVaR": risk_metrics.conditional_var(returns)
})

st.subheader("Monte Carlo Stress Test")
metrics = bayesian_eval.monte_carlo_metrics(returns, n_sim=200)
df_mc = pd.DataFrame(metrics)
st.line_chart(df_mc['sharpe'])
