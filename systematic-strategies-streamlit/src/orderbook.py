import yfinance as yf
import pandas as pd

def fetch_order_book_yf(symbol="BTC-USD"):
    ticker = yf.Ticker(symbol)
    ob = ticker.info
    bids = pd.DataFrame([[ob.get("bid", None), ob.get("bidSize", None)]], columns=["price", "size"])
    asks = pd.DataFrame([[ob.get("ask", None), ob.get("askSize", None)]], columns=["price", "size"])
    return bids, asks
