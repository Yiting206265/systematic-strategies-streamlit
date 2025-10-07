# Systematic Trading Strategies 2024-2025

Systematic equity & crypto trading strategies with:
- Mean-reversion backtests
- Black-Litterman portfolio optimization
- Bayesian Monte Carlo stress testing
- Risk metrics (Sharpe, Sortino, VaR, CVaR)
- Streamlit dashboard

## Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Run with Docker
```bash
docker build -t trading-strategies .
docker run -p 8501:8501 trading-strategies
```
