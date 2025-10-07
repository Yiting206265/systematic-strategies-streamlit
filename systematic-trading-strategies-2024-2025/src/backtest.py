import numpy as np
import pandas as pd

def mean_reversion_strategy(df, lookback=5, entry_z=1.0):
    df['zscore'] = (df['close'] - df['close'].rolling(lookback).mean()) / df['close'].rolling(lookback).std()
    df['position'] = np.where(df['zscore'] > entry_z, -1, np.where(df['zscore'] < -entry_z, 1, 0))
    df['position'] = df['position'].ffill().shift()
    df['strategy_returns'] = df['position'] * df['returns']
    return df.dropna()
