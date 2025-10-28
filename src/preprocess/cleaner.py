# File: src/preprocess/cleaner.py
# Purpose: Clean, resample, and standardize raw candle data.

import pandas as pd

def clean_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, nulls, and standardize timestamps."""
    df = df.copy()

    # 🧭 If index is 'timestamp', move it back to a column
    if df.index.name == "timestamp":
        df = df.reset_index()

    # 🕒 Handle both 'time' and 'timestamp' columns
    if "time" in df.columns and "timestamp" not in df.columns:
        df = df.rename(columns={"time": "timestamp"})

    # ✅ Ensure timestamp exists and is properly formatted
    if "timestamp" not in df.columns:
        raise KeyError("Candle data must contain a 'timestamp' or 'time' column.")

    # Convert UNIX timestamps (int) or string to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce", utc=True)

    # Drop bad or missing data
    df = df.drop_duplicates()
    df = df.dropna(subset=["timestamp", "close"])

    # Sort by symbol and timestamp if symbol exists
    if "symbol" in df.columns:
        df = df.sort_values(["symbol", "timestamp"])
    else:
        df = df.sort_values("timestamp")

    df = df.reset_index(drop=True)
    return df


def resample_candles(df: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
    """Resample candles to consistent frequency (default: 1 hour)."""
    all_resampled = []
    for symbol, grp in df.groupby("symbol"):
        grp = grp.set_index("timestamp").resample(freq).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna().reset_index()
        grp["symbol"] = symbol
        all_resampled.append(grp)

    return pd.concat(all_resampled, ignore_index=True)
