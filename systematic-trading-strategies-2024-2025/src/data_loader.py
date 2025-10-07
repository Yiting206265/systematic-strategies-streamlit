import pandas as pd
import yfinance as yf
import ccxt

def load_equities(ticker='AAPL', start='2024-01-01', end='2024-12-31'):
    df = yf.download(ticker, start=start, end=end)
    df.reset_index(inplace=True)
    df['close'] = df['Close']
    df['returns'] = df['Close'].pct_change()
    return df.dropna()

def load_crypto(symbol='BTC/USDT', exchange_name='binance', since=None, limit=500):
    exchange = getattr(ccxt, exchange_name)()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1d', since=since, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df['returns'] = df['close'].pct_change()
    return df.dropna()
