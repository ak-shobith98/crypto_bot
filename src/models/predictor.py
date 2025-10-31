# File: src/models/predictor.py
# Purpose: Load trained model, fetch live candles, predict BUY/SELL/HOLD with confidence,
# and log signals for live/paper trading workflows.

import os
import time
import joblib
import pandas as pd
import xgboost as xgb
import requests
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.collectors.realtime_data import fetch_realtime
from src.preprocess.features import add_technical_features
from src.config import REAL_TIME_BASE_URL, DATA_PATH

# === Paths ===
MODEL_PATH = DATA_PATH / "models" / "xgboost_latest.model"
SIGNAL_LOG_PATH = DATA_PATH / "result" / "live_signals.parquet"

# === Constants ===
DELTA_CANDLE_URL = f"{REAL_TIME_BASE_URL}/history/candles"


# --------------------------------------------------
# 🧱 Utility: Fetch recent N candles
# --------------------------------------------------
def fetch_recent_candles(symbol: str, limit: int = 100):
    """Fetch recent 1m candles from Delta Exchange."""
    try:
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(minutes=limit)
        params = {
            "symbol": symbol,
            "resolution": "1m",
            "start": int(start_dt.timestamp()),
            "end": int(end_dt.timestamp()),
        }

        resp = requests.get(DELTA_CANDLE_URL, params=params, timeout=8)
        resp.raise_for_status()

        data = resp.json().get("result", [])
        if not data:
            print("⚠️ Empty candle data received.")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df = df.rename(columns={
            "time": "timestamp", "t": "timestamp",
            "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df["symbol"] = symbol
        return df.sort_values("timestamp")

    except Exception as e:
        print(f"[ERROR] Failed to fetch recent candles: {e}")
        return pd.DataFrame()


# --------------------------------------------------
# 🧮 Feature Preparation
# --------------------------------------------------
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators & clean duplicates."""
    df = add_technical_features(df)
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.dropna().reset_index(drop=True)
    return df


# --------------------------------------------------
# 🧠 Load Model
# --------------------------------------------------
def load_trained_model():
    """Auto-detect and load XGBoost or Pickle model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")

    print(f"📦 Loading model from: {MODEL_PATH}")
    try:
        model = xgb.XGBClassifier()
        model.load_model(str(MODEL_PATH))
        print("✅ Loaded native XGBoost model.")
        return model
    except Exception:
        try:
            model = joblib.load(MODEL_PATH)
            print("✅ Loaded via joblib/pickle format.")
            return model
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load model: {e}")


# --------------------------------------------------
# 🔮 Predict Signal (Multi-class)
# --------------------------------------------------
def predict_signal(model, features: pd.DataFrame, prob_threshold: float = 0.55):
    """
    Predict trade signal (BUY / HOLD / SELL) using multi-class probabilities.
    Expects model.predict_proba() output in order: [BUY, HOLD, SELL].
    """
    if features.empty:
        return None

    # Keep only numeric columns
    X = features.select_dtypes(include=["number"]).fillna(0)

    # Ensure consistent feature order
    expected_order = [
        "open", "high", "low", "close", "volume", "return",
        "sma_7", "ema_14", "volatility_14", "rsi_14",
        "bb_upper", "bb_lower", "macd", "macd_signal",
        "macd_hist", "obv", "vwap"
    ]
    X = X.reindex(columns=expected_order, fill_value=0)

    try:
        probs = model.predict_proba(X)[-1]  # [p_buy, p_hold, p_sell]
    except Exception as e:
        print(f"[WARN] Predict_proba failed: {e}")
        return None

    p_buy, p_hold, p_sell = probs
    max_prob = max(p_buy, p_sell)

    # Decision logic
    if max_prob >= prob_threshold:
        if p_buy > p_sell:
            signal, label = 1, "BUY"
        else:
            signal, label = -1, "SELL"
    else:
        signal, label = 0, "HOLD"

    return {
        "timestamp": features.iloc[-1]["timestamp"],
        "symbol": features.iloc[-1]["symbol"],
        "p_buy": round(float(p_buy), 4),
        "p_hold": round(float(p_hold), 4),
        "p_sell": round(float(p_sell), 4),
        "confidence": round(float(max_prob), 4),
        "signal": signal,
        "result": label,
    }


# --------------------------------------------------
# 🗂️ Save Signal Log
# --------------------------------------------------
def save_signal(signal_row: dict):
    """Append new signal to a live log parquet."""
    SIGNAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SIGNAL_LOG_PATH.exists():
        existing = pd.read_parquet(SIGNAL_LOG_PATH)
        updated = pd.concat([existing, pd.DataFrame([signal_row])], ignore_index=True)
    else:
        updated = pd.DataFrame([signal_row])

    updated.to_parquet(SIGNAL_LOG_PATH, index=False)
    print(f"📝 Logged signal: {signal_row['result']} at {signal_row['timestamp']}")


# --------------------------------------------------
# 🚀 Main Live Prediction Loop
# --------------------------------------------------
def main(symbol: str = "BTCUSD", interval: int = 60):
    print(f"🚀 Starting live prediction loop for {symbol}")
    model = load_trained_model()

    while True:
        candles = fetch_recent_candles(symbol)
        if candles.empty:
            print("⚠️ No candle data. Retrying...")
            time.sleep(interval)
            continue

        features = prepare_features(candles)
        signal_row = predict_signal(model, features)
        if signal_row:
            save_signal(signal_row)
            print(f"🤖 Signal: {signal_row['result']} | Conf: {signal_row['confidence']} | p_buy={signal_row['p_buy']} | p_sell={signal_row['p_sell']}")
        else:
            print("⚠️ No signal generated.")

        # Fetch live price
        price_info = fetch_realtime(symbol)
        if price_info:
            print(f"💰 {symbol}: {price_info['price']} | 24h High: {price_info['high_24h']} | Low: {price_info['low_24h']}")

        print(f"⏱️ Waiting {interval}s before next prediction...\n")
        time.sleep(interval)


# --------------------------------------------------
# 🏁 Entry Point
# --------------------------------------------------
if __name__ == "__main__":
    main("BTCUSD", interval=60)
