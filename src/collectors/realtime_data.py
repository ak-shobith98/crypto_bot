# File: src/collectors/realtime_data.py
# Purpose: Stream live market data (WebSocket)

# crypto_bot.src.collector.realtime_data.py
import requests
from src.config import REAL_TIME_BASE_URL


def fetch_realtime(symbol):
    """
    Fetch realtime product info for `symbol` from Delta Exchange.
    Returns dict: {'symbol', 'price', 'volume_24h', 'high_24h', 'low_24h'} or None.
    """
    url = f"{REAL_TIME_BASE_URL}/tickers/{symbol}"
    try:
        resp = requests.get(url, timeout=6)
        if resp.status_code != 200:
            print(f"[WARN] Realtime fetch failed {symbol}: {resp.status_code}")
            return None

        data = resp.json().get("result", {})
        print(f"[DEBUG] Realtime data for {symbol}: {data}")
        return {
            "symbol": symbol,
            # Use mark_price, volume, high, low fields from Delta v2 API
            "price": float(data.get("mark_price")) if data.get("mark_price") else None,
            "volume_24h": float(data.get("volume")) if data.get("volume") else None,
            "high_24h": float(data.get("high")) if data.get("high") else None,
            "low_24h": float(data.get("low")) if data.get("low") else None
        }

    except Exception as e:
        print(f"[ERROR] Realtime fetch exception for {symbol}: {e}")
        return None


def main():
    # Example test for BTCUSD
    symbol = "BTCUSD"
    info = fetch_realtime(symbol)
    if info:
        print(f"Realtime data for {symbol}:")
        print(f"  Price: {info['price']}")
        print(f"  24h Volume: {info['volume_24h']}")
        print(f"  24h High: {info['high_24h']}")
        print(f"  24h Low: {info['low_24h']}")
    else:
        print("Failed to fetch realtime data.")


if __name__ == "__main__":
    main()
