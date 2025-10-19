import pandas as pd
import yfinance as yf
import ccxt

def load_equities(ticker: str = 'AAPL', period: str = '2y', interval: str = '1d'):
    """Load equity data with a rolling window so the latest year (e.g., 2025) shows up.
    Uses yfinance period/interval for simplicity and freshness.
    """
    df = yf.download(ticker, period=period, interval=interval)
    df.reset_index(inplace=True)
    df['close'] = df['Close']
    df['returns'] = df['Close'].pct_change()
    return df.dropna()

def load_crypto(symbol: str = "BTC-USD", period: str = "1y", interval: str = "1d"):
    df = yf.download(symbol, period=period, interval=interval)
    df.reset_index(inplace=True)
    df["close"] = df["Close"]
    df["returns"] = df["Close"].pct_change()
    return df.dropna()
