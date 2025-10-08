import pandas as pd
import yfinance as yf
import ccxt

def load_equities(ticker='AAPL', start='2024-01-01', end='2024-12-31'):
    df = yf.download(ticker, start=start, end=end)
    df.reset_index(inplace=True)
    df['close'] = df['Close']
    df['returns'] = df['Close'].pct_change()
    return df.dropna()

def load_crypto(symbol="BTC-USD", period="1y", interval="1d"):
    df = yf.download(symbol, period=period, interval=interval)
    df.reset_index(inplace=True)
    df["close"] = df["Close"]
    df["returns"] = df["Close"].pct_change()
    return df.dropna()
