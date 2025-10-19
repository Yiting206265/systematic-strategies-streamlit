import numpy as np
import pandas as pd

def risk_arbitrage_signal(
    df: pd.DataFrame,
    spread_col: str = 'close',
    lookback: int = 30,
    entry_z: float = 1.2,
    exit_z: float = 0.2,
    vol_target: float = 0.10,
    vol_ewm_halflife: int = 20,
    max_leverage: float = 3.0,
):
    # Use log-price as a proxy spread (for single series). In real risk-arb you'd use a pair spread.
    logp = np.log(df[spread_col].clip(lower=1e-8))
    mu = logp.rolling(lookback).mean()
    sig = logp.rolling(lookback).std()
    z = (logp - mu) / sig
    df['spread'] = z

    raw_pos = np.where(z > entry_z, -1, np.where(z < -entry_z, 1, np.nan))
    raw_pos = pd.Series(raw_pos, index=df.index).ffill()
    raw_pos = pd.Series(raw_pos, index=df.index).mask(z.abs() < exit_z, 0.0)

    # Vol targeting on returns
    ret = df['returns'].fillna(0.0)
    lam = np.log(2) / max(1, vol_ewm_halflife)
    ewm_var = ret.ewm(alpha=1 - np.exp(-lam)).var().fillna(method='bfill')
    daily_vol = np.sqrt(ewm_var)
    ann_vol = daily_vol * np.sqrt(252)
    leverage = (vol_target / ann_vol).clip(lower=0.0, upper=max_leverage)

    df['position'] = pd.Series(raw_pos, index=df.index).shift()
    df['leverage'] = leverage
    df['strategy_returns'] = df['position'] * df['leverage'] * df['returns']
    return df.dropna()
