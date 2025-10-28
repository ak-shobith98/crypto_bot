# File: src/collectors/fetch_candles.py
# Purpose: Fetch historical candle data from exchange API and persist to disk

import json
import time
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from src.config import DATA_PATH
from src.collectors.products import get_all_products


def fetch_candles(symbol: str, resolution='1h', days=1, retries=3, delay=1, chunk_days=7) -> pd.DataFrame | None:
    """
    Fetch historical candles from Delta Exchange API with chunking to avoid per-request limits.
    Saves raw JSON for traceability.

    Parameters:
        symbol (str): Trading pair like BTCUSD
        resolution (str): Candle interval ('1m', '5m', '1h', '1d')
        days (int): Number of days of historical data to fetch
        retries (int): Retry attempts
        delay (int): Delay between retries
        chunk_days (int): Split into smaller requests
    """
    base_url = "https://api.india.delta.exchange/v2/history/candles"
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)
    res_minutes = {'1m': 1, '5m': 5, '1h': 60, '1d': 1440}.get(resolution, 60)
    expected = int(days * 24 * 60 / res_minutes)
    print(f"[INFO] Fetching {symbol} ({resolution}, {days}d, expected ~{expected} candles)")

    all_data = []
    cur_start = start_dt

    while cur_start < end_dt:
        cur_end = min(cur_start + timedelta(days=chunk_days), end_dt)
        start = int(time.mktime(cur_start.timetuple()))
        end = int(time.mktime(cur_end.timetuple()))
        params = {"symbol": symbol, "resolution": resolution, "start": start, "end": end}

        for attempt in range(1, retries + 1):
            try:
                resp = requests.get(base_url, params=params, timeout=10)
                print(f"  → Chunk {cur_start.date()} → {cur_end.date()} (Attempt {attempt}) [{resp.status_code}]")
                if resp.status_code != 200:
                    time.sleep(delay)
                    continue
                chunk = resp.json().get("result", [])
                if chunk:
                    all_data.extend(chunk)
                break
            except requests.RequestException as e:
                print(f"[WARN] Retry due to: {e}")
                time.sleep(delay)
        cur_start = cur_end

    if not all_data:
        print(f"[WARN] No data for {symbol}")
        return None

    # Save raw JSON
    raw_path = DATA_PATH / "raw" / f"candles_{symbol}_{resolution}_{int(time.time())}.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "w") as f:
        json.dump(all_data, f)
    print(f"[INFO] Raw data saved → {raw_path}")

    # Detect timestamp key
    first = all_data[0]
    ts_key = next((k for k in ["t", "timestamp", "time"] if k in first), None)
    if not ts_key:
        print("[ERROR] No recognizable timestamp key.")
        return None

    # Normalize and convert
    df = pd.DataFrame(all_data)
    ts_vals = pd.to_numeric(df[ts_key], errors="coerce")
    unit = "ms" if ts_vals.max() > 1_000_000_000_000 else "s"
    df["timestamp"] = pd.to_datetime(ts_vals, unit=unit, utc=True)
    df.set_index("timestamp", inplace=True)
    df["symbol"] = symbol
    df.sort_index(inplace=True)

    print(f"[INFO] Fetched {len(df)} candles for {symbol}")
    return df


def fetch_all_symbols(resolution='1h', days=30) -> pd.DataFrame:
    """
    Fetch candles for all tradable products listed by products.py
    """
    products_df = get_all_products()
    symbols = products_df["symbol"].tolist()
    print(f"[INFO] Starting candle collection for {len(symbols)} symbols...")

    all_dfs = []
    for symbol in symbols:
        df = fetch_candles(symbol, resolution=resolution, days=days)
        if df is not None and not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        print("[ERROR] No candles fetched.")
        return pd.DataFrame()

    combined = pd.concat(all_dfs)
    combined.sort_index(inplace=True)

    processed_path = DATA_PATH / "processed" / "historical_candles.parquet"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(processed_path)
    print(f"[✅] Processed candles saved → {processed_path}")
    return combined


# Standalone run
if __name__ == "__main__":
    print("🚀 Running candle collector standalone...\n")
    df = fetch_all_symbols(resolution="1h", days=10)
    if not df.empty:
        print(df.head())
        print(f"✅ Done. Total rows: {len(df)}")
    else:
        print("❌ No data fetched.")
