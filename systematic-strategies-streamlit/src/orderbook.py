import ccxt
import pandas as pd

def fetch_order_book(symbol="BTC/USDT", exchange_name="binance", depth=10):
    """
    Fetch live order book from an exchange (via ccxt).
    Returns: bids and asks DataFrames with price/size
    """
    exchange = getattr(ccxt, exchange_name)()
    ob = exchange.fetch_order_book(symbol, limit=depth)
    bids = pd.DataFrame(ob['bids'], columns=['price','size'])
    asks = pd.DataFrame(ob['asks'], columns=['price','size'])
    return bids, asks
