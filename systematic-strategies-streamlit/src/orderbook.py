import requests, pandas as pd

def fetch_order_book_coinbase(symbol="BTC-USD", depth=10):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/book?level=2"
    resp = requests.get(url).json()
    bids = pd.DataFrame(resp["bids"][:depth], columns=["price","size","num_orders"])
    asks = pd.DataFrame(resp["asks"][:depth], columns=["price","size","num_orders"])
    return bids.astype(float), asks.astype(float)
