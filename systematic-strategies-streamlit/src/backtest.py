import numpy as np
import pandas as pd

def mean_reversion_strategy(
    df: pd.DataFrame,
    lookback: int = 20,
    entry_z: float = 1.0,
    exit_z: float = 0.2,
    vol_target: float = 0.10,  # annualized target vol (10%)
    vol_ewm_halflife: int = 20,
    max_leverage: float = 3.0,
    momentum_lookback: int = 50,
    momentum_cutoff: float = 0.10,
):
    # Z-score vs rolling mean
    roll_mean = df['close'].rolling(lookback).mean()
    roll_std = df['close'].rolling(lookback).std()
    df['zscore'] = (df['close'] - roll_mean) / roll_std

    # Entry/exit bands
    raw_pos = np.where(df['zscore'] > entry_z, -1, np.where(df['zscore'] < -entry_z, 1, np.nan))
    # Hold until exit band is crossed
    raw_pos = pd.Series(raw_pos, index=df.index).ffill()
    exit_mask = df['zscore'].abs() < exit_z
    raw_pos = raw_pos.mask(exit_mask, 0.0)

    # Momentum filter to avoid strong trends
    mom = df['close'] / df['close'].shift(momentum_lookback) - 1.0
    allow_long = ~(mom < -momentum_cutoff)  # avoid long when strong downtrend
    allow_short = ~(mom > momentum_cutoff)  # avoid short when strong uptrend
    filt_pos = np.where(raw_pos > 0, np.where(allow_long, 1.0, 0.0), np.where(raw_pos < 0, np.where(allow_short, -1.0, 0.0), 0.0))

    # Volatility targeting (EWMA vol -> annualized)
    ret = df['returns'].fillna(0.0)
    lam = np.log(2) / max(1, vol_ewm_halflife)
    ewm_var = ret.ewm(alpha=1 - np.exp(-lam)).var().fillna(method='bfill')
    daily_vol = np.sqrt(ewm_var)
    ann_vol = daily_vol * np.sqrt(252)
    leverage = (vol_target / ann_vol).clip(lower=0.0, upper=max_leverage)

    df['position'] = pd.Series(filt_pos, index=df.index).shift()  # trade next bar
    df['leverage'] = leverage
    df['strategy_returns'] = df['position'] * df['leverage'] * df['returns']
    return df.dropna()
