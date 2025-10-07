import numpy as np

def risk_arbitrage_signal(df, spread_col='close'):
    df['spread'] = df[spread_col].pct_change().cumsum()
    threshold = df['spread'].std()
    df['position'] = np.where(df['spread'] > threshold, -1, np.where(df['spread'] < -threshold, 1, 0))
    df['strategy_returns'] = df['position'].shift() * df['returns']
    return df.dropna()
