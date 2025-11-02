# File: src/models/predictor.py
# Purpose: Load the trained model, prepare live candle features,
# predict BUY/SELL/HOLD signals consistently with training setup.

import time
import joblib
import requests
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.preprocess.features import add_technical_features
from src.config import REAL_TIME_BASE_URL, DATA_PATH
from src.collectors.realtime_data import fetch_realtime

# === Paths ===
MODEL_PATH = DATA_PATH / "models" / "xgboost_latest.model"
SIGNAL_LOG_PATH = DATA_PATH / "result" / "live_signals.parquet"

# === Constants ===
CANDLE_URL = f"{REAL_TIME_BASE_URL}/history/candles"
CONF_THRESHOLD = 0.55  # Probability threshold for taking action


# --------------------------------------------------
# 🕒 Fetch recent N candles
# --------------------------------------------------
def fetch_recent_candles(symbol: str, limit: int = 120):
    """Fetch recent historical candles for live prediction."""
    try:
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(minutes=limit)
        params = {
            "symbol": symbol,
            "resolution": "1m",
            "start": int(start_dt.timestamp()),
            "end": int(end_dt.timestamp()),
        }
        r = requests.get(CANDLE_URL, params=params, timeout=10)
        r.raise_for_status()
        data = r.json().get("result", [])
        if not data:
            print(f"⚠️ No candle data received for {symbol}.")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df = df.rename(columns={
            "time": "timestamp",
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume"
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df["symbol"] = symbol
        return df.sort_values("timestamp")

    except Exception as e:
        print(f"[ERROR] fetch_recent_candles failed for {symbol}: {e}")
        return pd.DataFrame()


# --------------------------------------------------
# ⚙️ Prepare features
# --------------------------------------------------
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add indicators, clean data, and ensure consistency."""
    if df.empty:
        return df

    df = add_technical_features(df)
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.dropna().reset_index(drop=True)
    return df


# --------------------------------------------------
# 🧠 Load trained model
# --------------------------------------------------
def load_trained_model():
    """Load the latest trained XGBoost model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"❌ Model not found → {MODEL_PATH}")

    try:
        model = xgb.XGBClassifier()
        model.load_model(str(MODEL_PATH))
        print(f"📦 Loaded XGBoost model → {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"[WARN] Native load failed, trying joblib: {e}")
        model = joblib.load(MODEL_PATH)
        print("✅ Loaded model via joblib.")
        return model


# --------------------------------------------------
# 🔮 Predict signal (multi-class)
# --------------------------------------------------
def predict_signal(model, features: pd.DataFrame):
    """
    Predict the latest signal using multi-class probabilities.
    Class mapping during training:
        0 → SELL
        1 → HOLD
        2 → BUY
    """
    if features.empty:
        return None

    # Ensure correct column order (same as trainer)
    feature_order = [
        "open", "high", "low", "close", "volume", "return",
        "sma_7", "ema_14", "volatility_14", "rsi_14",
        "bb_upper", "bb_lower", "macd", "macd_signal",
        "macd_hist", "obv", "vwap"
    ]
    X = features.reindex(columns=feature_order, fill_value=0)

    # Predict only the last row (most recent candle)
    probs = model.predict_proba(X)[-1]  # [p_sell, p_hold, p_buy]
    p_sell, p_hold, p_buy = probs
    max_prob = max(p_buy, p_sell)

    if max_prob < CONF_THRESHOLD:
        signal, result = 0, "HOLD"
    elif p_buy > p_sell:
        signal, result = 1, "BUY"
    else:
        signal, result = -1, "SELL"

    latest_row = features.iloc[-1]
    return {
        "timestamp": latest_row["timestamp"],
        "symbol": latest_row["symbol"],
        "p_sell": round(float(p_sell), 4),
        "p_hold": round(float(p_hold), 4),
        "p_buy": round(float(p_buy), 4),
        "signal": signal,
        "result": result,
        "confidence": round(float(max_prob), 4),
        "close": round(float(latest_row["close"]), 4)
    }


# --------------------------------------------------
# 🗂️ Save signal logs
# --------------------------------------------------
def save_signal(signal_row: dict):
    """Append new signal to live log."""
    SIGNAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    if SIGNAL_LOG_PATH.exists():
        df = pd.read_parquet(SIGNAL_LOG_PATH)
        df = pd.concat([df, pd.DataFrame([signal_row])], ignore_index=True)
    else:
        df = pd.DataFrame([signal_row])

    df.to_parquet(SIGNAL_LOG_PATH, index=False)
    print(f"📝 Logged {signal_row['result']} | Conf={signal_row['confidence']}")


# --------------------------------------------------
# 🚀 Main live prediction loop
# --------------------------------------------------
def main(symbol: str = "BTCUSD", interval: int = 60):
    """Run live prediction in loop every N seconds."""
    model = load_trained_model()
    print(f"🚀 Live prediction loop started for {symbol} | every {interval}s\n")

    while True:
        candles = fetch_recent_candles(symbol)
        if candles.empty:
            print("⚠️ Skipping due to empty candles.")
            time.sleep(interval)
            continue

        features = prepare_features(candles)
        signal_row = predict_signal(model, features)
        if signal_row:
            save_signal(signal_row)
            print(f"🤖 {symbol} → {signal_row['result']} | "
                  f"p_buy={signal_row['p_buy']} | p_sell={signal_row['p_sell']} | conf={signal_row['confidence']}")
        else:
            print("⚠️ No valid signal generated.")

        # Optionally print live price info
        price_info = fetch_realtime(symbol)
        if price_info:
            print(f"💰 {symbol}: {price_info['price']} | "
                  f"High 24h: {price_info['high_24h']} | Low 24h: {price_info['low_24h']}")

        print(f"⏳ Waiting {interval}s...\n")
        time.sleep(interval)


# --------------------------------------------------
# 🏁 Entry point
# --------------------------------------------------
if __name__ == "__main__":
    main("BTCUSD", interval=60)
