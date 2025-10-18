import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm
from scipy.optimize import brentq
import altair as alt

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

def _bsm_d1(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan
    return (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))

@st.cache_data(show_spinner=False, ttl=300)
def get_last_close(ticker: str, period: str = "5d"):
    try:
        data = yf.Ticker(ticker).history(period=period).get("Close")
        if data is not None and len(data.dropna()) > 0:
            return float(data.dropna().iloc[-1])
    except Exception:
        pass
    return 100.0

def bsm_price(S, K, T, r, q, sigma, option_type="call"):
    d1 = _bsm_d1(S, K, T, r, q, sigma)
    d2 = d1 - sigma * sqrt(T)
    if option_type == "call":
        return S * exp(-q * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    else:
        return K * exp(-r * T) * norm.cdf(-d2) - S * exp(-q * T) * norm.cdf(-d1)

def bsm_delta(S, K, T, r, q, sigma, option_type="call"):
    d1 = _bsm_d1(S, K, T, r, q, sigma)
    if option_type == "call":
        return exp(-q * T) * norm.cdf(d1)
    else:
        return -exp(-q * T) * norm.cdf(-d1)

def bsm_gamma(S, K, T, r, q, sigma):
    d1 = _bsm_d1(S, K, T, r, q, sigma)
    return exp(-q * T) * norm.pdf(d1) / (S * sigma * sqrt(T))

def bsm_vega(S, K, T, r, q, sigma):
    d1 = _bsm_d1(S, K, T, r, q, sigma)
    return S * exp(-q * T) * norm.pdf(d1) * sqrt(T)

def bsm_theta(S, K, T, r, q, sigma, option_type="call"):
    d1 = _bsm_d1(S, K, T, r, q, sigma)
    d2 = d1 - sigma * sqrt(T)
    term1 = -S * exp(-q * T) * norm.pdf(d1) * sigma / (2 * sqrt(T))
    if option_type == "call":
        return term1 + q * S * exp(-q * T) * norm.cdf(d1) - r * K * exp(-r * T) * norm.cdf(d2)
    else:
        return term1 - q * S * exp(-q * T) * norm.cdf(-d1) + r * K * exp(-r * T) * norm.cdf(-d2)

def bsm_rho(S, K, T, r, q, sigma, option_type="call"):
    d1 = _bsm_d1(S, K, T, r, q, sigma)
    d2 = d1 - sigma * sqrt(T)
    if option_type == "call":
        return K * T * exp(-r * T) * norm.cdf(d2)
    else:
        return -K * T * exp(-r * T) * norm.cdf(-d2)

def implied_vol(price_mkt, S, K, T, r, q, option_type="call"):
    def f(sig):
        return bsm_price(S, K, T, r, q, sig, option_type) - price_mkt
    try:
        return brentq(f, 1e-6, 5.0, maxiter=100)
    except Exception:
        return np.nan

# ------------------------
# Additional Option Models
# ------------------------
def binomial_crr_price(S, K, T, r, q, sigma, N=200, option_type="call"):
    if T <= 0 or sigma <= 0 or N <= 0:
        # Immediate expiry or invalid -> intrinsic value
        return max(0.0, (S - K) if option_type == "call" else (K - S))
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)
    p = np.clip(p, 0.0, 1.0)
    # Terminal payoffs
    i = np.arange(N + 1)
    ST = S * (u ** (N - i)) * (d ** i)
    if option_type == "call":
        V = np.maximum(ST - K, 0.0)
    else:
        V = np.maximum(K - ST, 0.0)
    # Backward induction (European)
    for _ in range(N):
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
    return float(V[0])

def heston_mc_price(S, K, T, r, q, kappa, theta, v0, volvol, rho, option_type="call", n_paths=5000, n_steps=252, seed=42):
    if T <= 0:
        return max(0.0, (S - K) if option_type == "call" else (K - S))
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    S_t = np.full(n_paths, S, dtype=float)
    v_t = np.full(n_paths, max(v0, 1e-8), dtype=float)
    drift = (r - q)
    for _ in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        z2 = rho * z1 + np.sqrt(max(0.0, 1 - rho * rho)) * z2
        # Full truncation Euler for variance
        v_sqrt = np.sqrt(np.maximum(v_t, 0.0))
        v_next = v_t + kappa * (theta - np.maximum(v_t, 0.0)) * dt + volvol * v_sqrt * np.sqrt(dt) * z2
        v_next = np.maximum(v_next, 0.0)
        S_next = S_t * np.exp((drift - 0.5 * v_t) * dt + np.sqrt(v_t * dt) * z1)
        S_t, v_t = S_next, v_next
    if option_type == "call":
        payoff = np.maximum(S_t - K, 0.0)
    else:
        payoff = np.maximum(K - S_t, 0.0)
    price = np.exp(-r * T) * np.mean(payoff)
    return float(price)

# ------------------------
# Streamlit App
# ------------------------
st.set_page_config(page_title="Systematic Strategies", page_icon="📈", layout="wide")
st.title("Systematic Trading Strategies Platform (Equities & Crypto)")
st.markdown(
    """
    <style>
    .card {background: #111827; padding: 14px 16px; border-radius: 10px; border: 1px solid #1f2937;}
    .tight {margin-top: -10px;}
    </style>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(["Strategies", "Risk Metrics", "Monte Carlo", "Order Book", "Options", "Market Making", "Learn"])

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

    eq = (1 + res['strategy_returns']).cumprod()
    try:
        eq_df = eq.reset_index()
        eq_df.columns = ["Date", "Equity"] if len(eq_df.columns) == 2 else ["t", "Equity"]
        x_col = "Date" if "Date" in eq_df.columns else "t"
        chart = (
            alt.Chart(eq_df).mark_line().encode(
                x=alt.X(f"{x_col}:T" if x_col=="Date" else f"{x_col}:Q", title="Time"),
                y=alt.Y("Equity:Q", title="Cumulative Growth"),
                tooltip=list(eq_df.columns)
            ).properties(height=280)
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        st.line_chart(eq)

    c1, c2, c3 = st.columns(3)
    returns = res['strategy_returns'].dropna()
    with c1:
        st.metric("Last Return", f"{returns.tail(1).values[0]:.2%}" if not returns.empty else "NA")
    with c2:
        st.metric("Total Return", f"{eq.iloc[-1]-1:.2%}" if len(eq) else "NA")
    with c3:
        st.metric("Max Drawdown", f"{max_drawdown(eq):.2%}" if len(eq) else "NA")

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
    with st.spinner("Fetching order book..."):
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

with tabs[6]:  # Learn
    st.subheader("Learn")
    st.markdown("### Strategies")
    st.markdown("- Mean reversion: buy when price is below its recent mean, sell when above; expects reversion.")
    st.markdown("- Risk arbitrage: exploit temporary price gaps in related assets expecting convergence.")
    st.markdown("\n### Risk Metrics")
    st.markdown("- Sharpe: excess return per unit of volatility.  - Sortino: uses downside deviation only.  - Max Drawdown: worst peak-to-trough.")
    st.markdown("- VaR: loss threshold not exceeded with prob. (e.g., 95%).  - CVaR: average loss beyond VaR.")
    st.markdown("\n### Order Book")
    st.markdown("Top-of-book shows best bid/ask and spread; deeper books show depth of liquidity at each level.")
    st.markdown("\n### Options Basics")
    st.markdown("- Calls give the right to buy; puts the right to sell, at strike K by expiry T (European exercise here).")
    st.markdown("- Greeks: Delta (∂Price/∂S), Gamma (∂²Price/∂S²), Vega (∂Price/∂σ), Theta (∂Price/∂t), Rho (∂Price/∂r).")
    st.markdown("- Implied Volatility (IV): volatility that makes the model price match the observed price.")
    st.markdown("\n### Models: High-level intuition and derivations")
    st.markdown("- Black–Scholes–Merton (BSM): assumes geometric Brownian motion. Under risk-neutral measure, solve the diffusion PDE with terminal payoff; closed form uses N(d1), N(d2). Key step: replication removes drift.")
    st.latex(r"\frac{dS_t}{S_t} = (r - q)\,dt + \sigma\, dW_t")
    st.latex(r"d_1 = \frac{\ln(S/K) + (r-q + \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}},\quad d_2 = d_1 - \sigma\sqrt{T}")
    st.latex(r"C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2),\quad P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)")
    st.markdown("- Binomial (CRR): discretize time into N steps with up/down factors; price by backward induction under risk-neutral probability.")
    st.latex(r"u = e^{\sigma\sqrt{\Delta t}},\quad d = 1/u,\quad p = \frac{e^{(r-q)\Delta t} - d}{u - d}")
    st.markdown("- Heston: stochastic variance with correlation to price; priced via Monte Carlo or semi-analytical integration.")
    st.latex(r"\begin{aligned} dS_t &= (r-q) S_t\,dt + \sqrt{v_t}\, S_t\, dW_t^S, \\ dv_t &= \kappa(\theta - v_t)\,dt + \xi \sqrt{v_t}\, dW_t^v, \quad d\langle W^S, W^v \rangle_t = \rho\, dt. \end{aligned}")
    st.markdown("\n### Black–Litterman (portfolio model, not option pricing)")
    st.markdown("- Combines market-implied equilibrium returns with investor views into posterior expected returns; stabilizes mean-variance optimization.")
    st.latex(r"\mu_{BL} = \big( (\tau\Sigma)^{-1} + P^\top \Omega^{-1} P \big)^{-1} \big( (\tau\Sigma)^{-1} \mu^* + P^\top \Omega^{-1} Q \big)")

with tabs[4]:  # Options
    st.subheader("Options Analytics")
    colA, colB = st.columns(2)
    with colA:
        ticker_opt = st.text_input("Ticker", "AAPL")
        with st.spinner("Loading latest price..."):
            S_default = get_last_close(ticker_opt)
        # Presets
        preset = st.selectbox("Preset", ["Custom", "ATM 30D", "ATM 60D", "Straddle 30D (hi vol)"])
        apply = st.button("Apply Preset")
        if apply and preset != "Custom":
            from datetime import date, timedelta
            S_base = float(round(S_default, 2))
            if preset == "ATM 30D":
                st.session_state.setdefault("opt_K", S_base)
                st.session_state.setdefault("opt_sigma", 0.2)
                st.session_state.setdefault("opt_exp", pd.Timestamp.today().date() + timedelta(days=30))
                st.session_state.update({"opt_K": S_base, "opt_sigma": 0.2, "opt_exp": pd.Timestamp.today().date() + timedelta(days=30)})
            elif preset == "ATM 60D":
                st.session_state.update({"opt_K": S_base, "opt_sigma": 0.22, "opt_exp": pd.Timestamp.today().date() + timedelta(days=60)})
            elif preset == "Straddle 30D (hi vol)":
                st.session_state.update({"opt_K": S_base, "opt_sigma": 0.4, "opt_exp": pd.Timestamp.today().date() + timedelta(days=30)})
        S = st.number_input("Spot S", min_value=0.01, value=float(round(S_default, 2)), key="opt_S")
        K = st.number_input("Strike K", min_value=0.01, value=float(round(S_default, 2)), key="opt_K")
        expiry = st.date_input("Expiry", key="opt_exp")
        today = pd.Timestamp.today().tz_localize(None).date()
        T = max((pd.to_datetime(expiry) - pd.to_datetime(today)).days, 0) / 365.0
        r = st.number_input("Risk-free r", value=0.04, step=0.01, format="%.4f")
        q = st.number_input("Dividend yield q", value=0.0, step=0.01, format="%.4f")
        sigma = st.number_input("Volatility sigma", value=st.session_state.get("opt_sigma", 0.2), step=0.01, format="%.4f", key="opt_sigma")
    with colB:
        option_type = st.selectbox("Type", ["call", "put"])
        style = st.selectbox("Style", ["european"])  
        model = st.selectbox("Model", ["Black-Scholes-Merton", "Binomial (CRR)", "Heston (Monte Carlo)"])  
        # Model-specific parameters
        bin_N = None
        h_kappa = h_theta = h_v0 = h_volvol = h_rho = None
        h_paths = h_steps = None
        if model == "Binomial (CRR)":
            bin_N = st.slider("Binomial Steps (N)", 50, 1000, 200, step=50)
        elif model == "Heston (Monte Carlo)":
            h_kappa = st.number_input("kappa (mean reversion)", value=1.5, step=0.1, format="%.2f")
            h_theta = st.number_input("theta (long-run var)", value=0.04, step=0.01, format="%.4f")
            h_v0 = st.number_input("v0 (start var)", value=float(max(0.04, 1e-4)), step=0.01, format="%.4f")
            h_volvol = st.number_input("vol of vol", value=0.5, step=0.05, format="%.2f")
            h_rho = st.slider("rho (corr)", -0.99, 0.99, -0.6, 0.01)
            h_paths = st.slider("Monte Carlo paths", 1000, 20000, 5000, step=1000)
            h_steps = st.slider("Time steps", 50, 365, 252, step=10)
        calc = st.button("Calculate")
    if calc:
        # Compute price and greeks depending on model
        if model == "Black-Scholes-Merton":
            price = bsm_price(S, K, T, r, q, sigma, option_type)
            delta = bsm_delta(S, K, T, r, q, sigma, option_type)
            gamma = bsm_gamma(S, K, T, r, q, sigma)
            vega = bsm_vega(S, K, T, r, q, sigma)
            theta = bsm_theta(S, K, T, r, q, sigma, option_type)
            rho = bsm_rho(S, K, T, r, q, sigma, option_type)
        elif model == "Binomial (CRR)":
            price = binomial_crr_price(S, K, T, r, q, sigma, N=bin_N, option_type=option_type)
            # Finite diff greeks on price
            dS = max(1e-4, 0.01 * S)
            p_up = binomial_crr_price(S + dS, K, T, r, q, sigma, N=bin_N, option_type=option_type)
            p_dn = binomial_crr_price(S - dS, K, T, r, q, sigma, N=bin_N, option_type=option_type)
            delta = (p_up - p_dn) / (2 * dS)
            gamma = (p_up - 2 * price + p_dn) / (dS ** 2)
            vega = theta = rho = np.nan
        else:  # Heston MC
            price = heston_mc_price(S, K, T, r, q, h_kappa, h_theta, h_v0, h_volvol, h_rho, option_type, n_paths=h_paths, n_steps=h_steps)
            dS = max(1e-4, 0.01 * S)
            p_up = heston_mc_price(S + dS, K, T, r, q, h_kappa, h_theta, h_v0, h_volvol, h_rho, option_type, n_paths=h_paths, n_steps=h_steps)
            p_dn = heston_mc_price(S - dS, K, T, r, q, h_kappa, h_theta, h_v0, h_volvol, h_rho, option_type, n_paths=h_paths, n_steps=h_steps)
            delta = (p_up - p_dn) / (2 * dS)
            gamma = (p_up - 2 * price + p_dn) / (dS ** 2)
            vega = theta = rho = np.nan
        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("Price", f"{price:.4f}")
            st.metric("Delta", f"{delta:.4f}" if np.isfinite(delta) else "NA")
        with k2:
            st.metric("Gamma", f"{gamma:.6f}" if np.isfinite(gamma) else "NA")
            st.metric("Vega", f"{vega:.4f}" if np.isfinite(vega) else "NA")
        with k3:
            st.metric("Theta", f"{theta:.4f}" if np.isfinite(theta) else "NA")
            st.metric("Rho", f"{rho:.4f}" if np.isfinite(rho) else "NA")
        col1, col2 = st.columns(2)
        with col1:
            s_grid = np.linspace(max(0.01, S*0.5), S*1.5, 100)
            payoff = np.where(option_type=="call", np.maximum(s_grid - K, 0), np.maximum(K - s_grid, 0))
            df_pay = pd.DataFrame({"Spot": s_grid, "Payoff": payoff})
            ch = alt.Chart(df_pay).mark_line().encode(
                x=alt.X("Spot:Q"), y=alt.Y("Payoff:Q"), tooltip=["Spot","Payoff"]
            ).properties(height=260)
            st.altair_chart(ch, use_container_width=True)
        with col2:
            mode = st.selectbox("Sensitivity", ["Premium vs Strike", "Premium vs Volatility"]) 
            if mode == "Premium vs Strike":
                ks = np.linspace(max(0.01, S*0.5), S*1.5, 50)
                prem = [bsm_price(S, k_, T, r, q, sigma, option_type) for k_ in ks]
                df_sens = pd.DataFrame({"Strike": ks, "Premium": prem})
                st.altair_chart(alt.Chart(df_sens).mark_line().encode(
                    x="Strike:Q", y="Premium:Q", tooltip=["Strike","Premium"]
                ).properties(height=260), use_container_width=True)
            else:
                sigs = np.linspace(0.05, 1.0, 50)
                prem = [bsm_price(S, K, T, r, q, s_, option_type) for s_ in sigs]
                df_sens = pd.DataFrame({"Vol": sigs, "Premium": prem})
                st.altair_chart(alt.Chart(df_sens).mark_line().encode(
                    x="Vol:Q", y="Premium:Q", tooltip=["Vol","Premium"]
                ).properties(height=260), use_container_width=True)
        st.markdown("### Implied Volatility")
        price_mkt = st.number_input("Observed Option Price", value=float(round(price, 4)))
        if model == "Black-Scholes-Merton":
            if st.button("Solve IV"):
                iv = implied_vol(price_mkt, S, K, T, r, q, option_type)
                st.write({"implied_vol": iv})
        else:
            st.info("IV solve is available for Black-Scholes-Merton only.")

    with st.expander("Mini-Lab: IV Smile and Term Structure"):
        csmile, cterm = st.columns(2)
        with csmile:
            ks = np.linspace(max(0.01, S*0.5), S*1.5, 31)
            prem = [bsm_price(S, k_, T, r, q, sigma, option_type) for k_ in ks]
            df_smile = pd.DataFrame({"Strike": ks, "Premium": prem})
            st.altair_chart(alt.Chart(df_smile).mark_line().encode(x="Strike:Q", y="Premium:Q", tooltip=["Strike","Premium"]).properties(height=220), use_container_width=True)
        with cterm:
            Ts = np.linspace(max(1/365, T/10 if T>0 else 30/365), max(T, 180/365), 31)
            premT = [bsm_price(S, K, t_, r, q, sigma, option_type) for t_ in Ts]
            df_term = pd.DataFrame({"T_years": Ts, "Premium": premT})
            st.altair_chart(alt.Chart(df_term).mark_line().encode(x=alt.X("T_years:Q", title="T (years)"), y="Premium:Q", tooltip=["T_years","Premium"]).properties(height=220), use_container_width=True)
    st.markdown("---")
    st.subheader("Strategy Builder")
    use_builder = st.checkbox("Enable multi-leg strategy")
    if use_builder:
        n_legs = st.slider("Number of legs", 1, 4, 2)
        legs = []
        for i in range(n_legs):
            c1, c2, c3 = st.columns(3)
            with c1:
                t_i = st.selectbox(f"Type {i+1}", ["call", "put"], key=f"typ_{i}")
            with c2:
                k_i = st.number_input(f"Strike {i+1}", value=float(round(S, 2)), key=f"k_{i}")
            with c3:
                q_i = st.number_input(f"Qty {i+1}", value=1.0, step=1.0, key=f"q_{i}")
            legs.append((t_i, k_i, q_i))
        s_grid = np.linspace(max(0.01, S*0.5), S*1.5, 200)
        payoff = np.zeros_like(s_grid)
        for (t_i, k_i, q_i) in legs:
            payoff += q_i * np.where(t_i=="call", np.maximum(s_grid - k_i, 0), np.maximum(k_i - s_grid, 0))
        df_comb = pd.DataFrame({"Spot": s_grid, "Payoff": payoff})
        st.line_chart(df_comb.set_index("Spot"))

with tabs[5]:  # Market Making
    st.subheader("Market Making Simulation (2% Price Making + Skew)")
    colL, colR = st.columns(2)
    with colL:
        mm_ticker = st.text_input("Ticker", "AAPL", key="mm_ticker")
        mm_steps = st.slider("Number of requests", 5, 60, 20, key="mm_steps")
        mm_volume = st.number_input("Quote volume per request", min_value=1, value=100, step=10, key="mm_vol")
        spread_pct = st.number_input("Spread % (two-sided from ref)", min_value=0.0, value=2.0, step=0.1, format="%.1f", key="mm_spread")
        skew_pct = st.number_input("Inventory skew % (shift both sides)", min_value=0.0, value=1.0, step=0.1, format="%.1f", key="mm_skew")
    with colR:
        prob_buy = st.slider("Client Buy probability", 0.0, 1.0, 0.4, 0.05, key="mm_pbuy")
        prob_sell = st.slider("Client Sell probability", 0.0, 1.0, 0.4, 0.05, key="mm_psell")
        prob_ref = max(0.0, 1.0 - prob_buy - prob_sell)
        st.write({"p_buy": prob_buy, "p_sell": prob_sell, "p_refuse": round(prob_ref, 2)})
        run_sim = st.button("Run Simulation")

    class QuotedTrade:
        def __init__(self, ticker, trade_volume, ref_price, bid_price, offer_price, date):
            self.ticker = ticker
            self.trade_volume = int(trade_volume)
            self.ref_price = float(ref_price)
            self.bid_price = float(bid_price)
            self.offer_price = float(offer_price)
            self.date = date
        def __repr__(self):
            return f"QuotedTrade({self.ticker}, vol={self.trade_volume}, ref={self.ref_price:.2f}, bid={self.bid_price:.2f}, offer={self.offer_price:.2f}, date={self.date})"

    class CompletedTrade:
        def __init__(self, ticker, trade_volume, trade_price, mm_action, ref_price, bid_price, offer_price, date):
            self.ticker = ticker
            self.trade_volume = int(trade_volume)
            self.trade_price = float(trade_price)
            self.mm_action = mm_action
            self.ref_price = float(ref_price)
            self.bid_price = float(bid_price)
            self.offer_price = float(offer_price)
            self.date = date
        def __repr__(self):
            return f"CompletedTrade({self.ticker}, {self.mm_action} {self.trade_volume} @ {self.trade_price:.2f} on {self.date})"

    class MarketMakerLite:
        def __init__(self):
            self.current_positions = {}
            self.quoted_trades = []
            self.completed_trades = []
        def position_value(self, ticker):
            pos = self.current_positions.get(ticker, {"qty": 0, "avg": 0.0})
            return int(pos["qty"]), float(pos["avg"])
        def update_position(self, ticker, side, qty, price):
            q_old, avg_old = self.position_value(ticker)
            delta = qty if side == "Buy" else -qty
            q_new = q_old + delta
            if q_new == 0:
                avg_new = 0.0
            elif q_old == 0 or (q_old > 0 and q_new > 0) or (q_old < 0 and q_new < 0):
                avg_new = ((avg_old * q_old) + price * delta) / q_new
            else:
                avg_new = price
            self.current_positions[ticker] = {"qty": q_new, "avg": avg_new}

    if run_sim:
        try:
            hist = yf.Ticker(mm_ticker).history(period="6mo", interval="1d").reset_index()
            ref_series = hist["Close"].dropna()
            if len(ref_series) == 0:
                st.warning("No price data. Try another ticker.")
            else:
                ref_series = ref_series.tail(mm_steps)
                dates = list(range(len(ref_series)))
                mm_engine = MarketMakerLite()
                s_pct = spread_pct / 100.0
                k_pct = skew_pct / 100.0
                quotes, pos_hist, pnl_hist = [], [], []
                qty_hist, avg_hist = [], []
                pnl = 0.0
                for i, ref_px in enumerate(ref_series):
                    qty, avg = mm_engine.position_value(mm_ticker)
                    shift = (-k_pct if qty > 0 else (k_pct if qty < 0 else 0.0))
                    bid = ref_px * (1 - s_pct + shift)
                    offer = ref_px * (1 + s_pct + shift)
                    qt = QuotedTrade(mm_ticker, mm_volume, ref_px, bid, offer, dates[i])
                    mm_engine.quoted_trades.append(qt)
                    u = float(np.random.random())
                    action = "Refuse"
                    if u < prob_buy:
                        action = "Buy"
                    elif u < prob_buy + prob_sell:
                        action = "Sell"
                    if action != "Refuse":
                        if action == "Buy":
                            trade_px = offer
                            mm_side = "Sell"
                        else:
                            trade_px = bid
                            mm_side = "Buy"
                        ct = CompletedTrade(mm_ticker, mm_volume, trade_px, mm_side, ref_px, bid, offer, dates[i])
                        mm_engine.completed_trades.append(ct)
                        mm_engine.update_position(mm_ticker, mm_side, mm_volume, trade_px)
                    qty, avg = mm_engine.position_value(mm_ticker)
                    pos_hist.append(qty)
                    qty_hist.append(qty)
                    avg_hist.append(avg)
                    mtm = (ref_px - avg) * qty if qty != 0 else 0.0
                    pnl_hist.append(mtm)
                    quotes.append({"date": dates[i], "ref": ref_px, "bid": bid, "offer": offer})

                df_quotes = pd.DataFrame(quotes)
                st.markdown("### Quotes vs Reference Price")
                qlong = df_quotes.melt("date", var_name="series", value_name="value")
                chq = alt.Chart(qlong).mark_line().encode(
                    x=alt.X("date:Q", title="Step"),
                    y=alt.Y("value:Q", title="Price"),
                    color="series:N",
                    tooltip=["date","series","value"]
                ).properties(height=280)
                st.altair_chart(chq, use_container_width=True)

                st.markdown("### Position and PnL")
                df_pos = pd.DataFrame({"date": dates, "position": pos_hist, "pnl": pnl_hist})
                ch_pos = alt.Chart(df_pos).mark_line(color="#6366f1").encode(
                    x="date:Q", y=alt.Y("position:Q", title="Position"), tooltip=["date","position"]
                ).properties(height=220)
                ch_pnl = alt.Chart(df_pos).mark_line(color="#10b981").encode(
                    x="date:Q", y=alt.Y("pnl:Q", title="PnL"), tooltip=["date","pnl"]
                ).properties(height=220)
                st.altair_chart(ch_pos, use_container_width=True)
                st.altair_chart(ch_pnl, use_container_width=True)

                st.markdown("### Samples")
                st.write("Quoted Trades (first 10)")
                st.dataframe(pd.DataFrame([vars(x) for x in mm_engine.quoted_trades[:10]]))
                st.write("Completed Trades (first 10)")
                st.dataframe(pd.DataFrame([vars(x) for x in mm_engine.completed_trades[:10]]))
                qty_end, avg_end = mm_engine.position_value(mm_ticker)
                st.write({"ending_position_qty": qty_end, "ending_avg_price": avg_end})

                # Diagnostics and downloads
                fills = len(mm_engine.completed_trades)
                buys = sum(1 for t in mm_engine.completed_trades if t.mm_action == "Sell")
                sells = fills - buys
                total_quotes = len(mm_engine.quoted_trades)
                refuse = total_quotes - fills
                cA, cB, cC, cD = st.columns(4)
                cA.metric("Quotes", total_quotes)
                cB.metric("Fills", fills)
                cC.metric("Fill Rate", f"{(fills/max(1,total_quotes)):.1%}")
                cD.metric("Refused", refuse)

                qcsv = df_quotes.to_csv(index=False).encode()
                tdf = pd.DataFrame([vars(x) for x in mm_engine.completed_trades])
                tcsv = tdf.to_csv(index=False).encode() if not tdf.empty else b""
                d1, d2 = st.columns(2)
                with d1:
                    st.download_button("Download Quotes CSV", qcsv, file_name=f"{mm_ticker}_quotes.csv", mime="text/csv")
                with d2:
                    st.download_button("Download Trades CSV", tcsv, file_name=f"{mm_ticker}_trades.csv", mime="text/csv", disabled=tdf.empty)
        except Exception as e:
            st.error(f"Simulation error: {e}")
