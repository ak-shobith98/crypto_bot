# File: src/preprocess/features.py
# Purpose: Compute advanced technical features for each trading symbol.

import pandas as pd
import numpy as np


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute common trading indicators for each symbol.

    Parameters
    ----------
    df : pd.DataFrame
        Candle data containing columns:
        ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']

    Returns
    -------
    pd.DataFrame
        Original DataFrame enriched with technical indicators.
    """

    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index (RSI)."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume (OBV)."""
        direction = np.sign(close.diff()).fillna(0)
        return (direction * volume).cumsum()

    def vwap(df: pd.DataFrame) -> pd.Series:
        """Volume Weighted Average Price (VWAP)."""
        return (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

    all_features = []

    for symbol, grp in df.groupby("symbol"):
        grp = grp.sort_values("timestamp").copy()

        # ---- Core price stats ----
        grp["return"] = grp["close"].pct_change()
        grp["sma_7"] = grp["close"].rolling(7).mean()
        grp["ema_14"] = grp["close"].ewm(span=14, adjust=False).mean()
        grp["volatility_14"] = grp["return"].rolling(14).std()

        # ---- RSI ----
        grp["rsi_14"] = rsi(grp["close"], 14)

        # ---- Bollinger Bands ----
        rolling_mean = grp["close"].rolling(20).mean()
        rolling_std = grp["close"].rolling(20).std()
        grp["bb_upper"] = rolling_mean + (2 * rolling_std)
        grp["bb_lower"] = rolling_mean - (2 * rolling_std)

        # ---- MACD (Moving Average Convergence Divergence) ----
        ema_12 = grp["close"].ewm(span=12, adjust=False).mean()
        ema_26 = grp["close"].ewm(span=26, adjust=False).mean()
        grp["macd"] = ema_12 - ema_26
        grp["macd_signal"] = grp["macd"].ewm(span=9, adjust=False).mean()
        grp["macd_hist"] = grp["macd"] - grp["macd_signal"]

        # ---- OBV (On-Balance Volume) ----
        grp["obv"] = obv(grp["close"], grp["volume"])

        # ---- VWAP ----
        grp["vwap"] = vwap(grp)

        all_features.append(grp)

    # Concatenate all symbols and clean
    features = pd.concat(all_features, ignore_index=True)
    features = features.dropna().reset_index(drop=True)
    return features


# Optional: quick test
if __name__ == "__main__":
    sample = pd.DataFrame({
        "symbol": ["BTC-USD"] * 40,
        "timestamp": pd.date_range("2025-01-01", periods=40, freq="H"),
        "open": np.random.rand(40) * 100,
        "high": np.random.rand(40) * 100,
        "low": np.random.rand(40) * 100,
        "close": np.random.rand(40) * 100,
        "volume": np.random.rand(40) * 1000
    })

    enriched = add_technical_features(sample)
    print(enriched.tail())
