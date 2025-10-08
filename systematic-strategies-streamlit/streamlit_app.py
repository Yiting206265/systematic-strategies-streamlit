import streamlit as st
import pandas as pd

from src.data_loader import load_equities, load_crypto
from src.backtest import mean_reversion_strategy
from src.arbitrage import risk_arbitrage_signal
from src.risk_metrics import sharpe_ratio, sortino_ratio, max_drawdown, value_at_risk, conditional_var
from src.bayesian_eval import monte_carlo_metrics

st.title("Quantitative Trading Strategies Dashboard")

asset = st.sidebar.selectbox("Choose Asset", ["Equities", "Crypto"])

if asset == "Equities":
    ticker = st.sidebar.text_input("Equity Ticker", "AAPL")
    df = load_equities(ticker)
else:
    symbol = st.sidebar.text_input("Crypto Symbol", "BTC/USDT")
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

st.subheader("Risk Metrics")
returns = res['strategy_returns'].dropna()
st.write({
    "Sharpe": sharpe_ratio(returns),
    "Sortino": sortino_ratio(returns),
    "MaxDD": max_drawdown((1+returns).cumprod()),
    "VaR": value_at_risk(returns),
    "CVaR": conditional_var(returns)
})

st.subheader("Monte Carlo Stress Test")
metrics = monte_carlo_metrics(returns, n_sim=200)
df_mc = pd.DataFrame(metrics)
st.line_chart(df_mc['sharpe'])
