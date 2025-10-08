import numpy as np
import pandas as pd

def simple_market_maker(df, spread=0.001, inventory_penalty=0.0005):
    """
    Simulates a toy market maker strategy.
    - df: DataFrame with 'close' prices
    - spread: half-spread around midprice
    - inventory_penalty: cost for carrying inventory
    """
    df = df.copy()
    df['mid'] = df['close']
    df['bid'] = df['mid'] * (1 - spread)
    df['ask'] = df['mid'] * (1 + spread)

    # Simulate fills: random side of the book
    np.random.seed(42)
    df['side'] = np.random.choice([-1, 1], size=len(df))  # -1 sell, +1 buy
    df['fill_price'] = np.where(df['side'] > 0, df['ask'], df['bid'])

    # Inventory PnL
    df['inventory'] = df['side'].cumsum()
    df['pnl'] = (df['fill_price'].diff().fillna(0) * -df['inventory']) \
                 - inventory_penalty * np.abs(df['inventory'])
    df['strategy_returns'] = df['pnl'] / df['mid']
    return df
