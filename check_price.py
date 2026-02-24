# check_price.py – vypíše aktuálne Last Price a Mark Price pre BTCUSDT

from binance.um_futures import UMFutures
import os
from dotenv import load_dotenv

def main():
    load_dotenv()
    FUTURES_BASE = os.getenv("FUTURES_BASE", "https://testnet.binancefuture.com")
    KEY = os.getenv("BINANCE_API_KEY")
    SEC = os.getenv("BINANCE_API_SECRET")

    client = UMFutures(key=KEY, secret=SEC, base_url=FUTURES_BASE)

    symbol = "BTCUSDT"

    # Last Price (posledná obchodná cena)
    ticker = client.ticker_price(symbol=symbol)
    last_price = float(ticker["price"])

    # Mark Price (referenčná cena)
    mark_info = client.mark_price(symbol=symbol)
    mark_price = float(mark_info["markPrice"])

    print(f"Symbol: {symbol}")
    print(f"  Last Price = {last_price:.2f}")
    print(f"  Mark Price = {mark_price:.2f}")

if __name__ == "__main__":
    main()