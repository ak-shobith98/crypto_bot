# File: src/models/predictor.py
# Purpose: Load trained model and make live predictions on real-time candle data.

import os
import time
import json
import joblib
import pandas as pd
import xgboost as xgb
import requests
from datetime import datetime, timedelta, timezone
from src.collectors.realtime_data import fetch_realtime
from src.config import REAL_TIME_BASE_URL
from src.preprocess.features import add_technical_features

# === Paths ===
MODEL_PATH = os.path.join("data", "models", "xgboost_latest.model")
FEATURE_COLUMNS_PATH = os.path.join("data", "models", "feature_columns.json")

# === Constants ===
DELTA_CANDLE_URL = f"{REAL_TIME_BASE_URL}/history/candles"


# --------------------------------------------------
# 🧱 Utility: Fetch latest 100 candles
# --------------------------------------------------
def fetch_recent_candles(symbol: str, limit: int = 100):
    """Fetch recent candles from Delta Exchange using proper start/end params."""
    try:
        # Compute start and end times (approx 100 minutes back)
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(minutes=limit)

        start = int(start_dt.timestamp())
        end = int(end_dt.timestamp())

        params = {
            "symbol": symbol,
            "resolution": "1m",
            "start": start,
            "end": end,
        }

        resp = requests.get(DELTA_CANDLE_URL, params=params, timeout=8)
        resp.raise_for_status()

        data = resp.json().get("result", [])
        if not data:
            print("⚠️ Empty candle data received.")
            return None

        df = pd.DataFrame(data)
        df = df.rename(
            columns={
                "time": "timestamp",
                "t": "timestamp",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
            }
        )

        if "timestamp" not in df.columns:
            print("❌ No timestamp column found in response.")
            return None

        # Convert timestamp properly (Delta returns in seconds)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df["symbol"] = symbol

        return df

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to fetch recent candles: {e}")
        return None


# --------------------------------------------------
# 🧮 Feature Preparation
# --------------------------------------------------
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = add_technical_features(df)
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.dropna().reset_index(drop=True)
    return df


# --------------------------------------------------
# 🧠 Load Model (auto detect pickle vs native)
# --------------------------------------------------
def load_trained_model(model_path: str):
    """Auto-detect and load XGBoost or Pickle/Joblib model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"📦 Loading model from: {model_path}")

    try:
        # Try native XGBoost format first
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        print("✅ Loaded XGBoost native model format.")
        return model
    except Exception:
        # Fall back to Pickle/Joblib format
        try:
            model = joblib.load(model_path)
            print("✅ Loaded model via joblib/pickle format.")
            return model
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load model: {e}")


# --------------------------------------------------
# 🤖 Predict Signal
# --------------------------------------------------
def predict_signal(model, features: pd.DataFrame):
    if features.empty:
        print("⚠️ No features available for prediction.")
        return None

    # Drop non-numeric columns
    non_numeric = features.select_dtypes(exclude=["number"]).columns
    if len(non_numeric) > 0:
        features = features.drop(columns=non_numeric, errors="ignore")

    # ✅ Match the feature order used during training
    expected_order = [
        "open", "high", "low", "close", "volume", "return",
        "sma_7", "ema_14", "volatility_14", "rsi_14",
        "bb_upper", "bb_lower", "macd", "macd_signal",
        "macd_hist", "obv", "vwap"
    ]

    # Reorder safely
    features = features.reindex(columns=expected_order, fill_value=0)
    features = features.fillna(0)

    try:
        pred = model.predict(features)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0][1]
        else:
            proba = 0.5
        signal = "BUY" if pred[0] == 1 else "SELL"
        print(f"🤖 Predicted signal: {signal} (probability: {proba:.2f})")
        return signal
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return None

# --------------------------------------------------
# 🚀 Main Live Prediction Loop
# --------------------------------------------------
def main(symbol: str = "BTCUSD"):
    print(f"🚀 Starting live prediction loop for {symbol}")

    try:
        model = load_trained_model(MODEL_PATH)
    except Exception as e:
        print(e)
        return

    while True:
        candles = fetch_recent_candles(symbol)
        if candles is None or candles.empty:
            print("⚠️ No data fetched, retrying in 30s...")
            time.sleep(30)
            continue

        features = prepare_features(candles)
        latest_features = features.tail(1)

        # 🔧 Ensure correct feature alignment before prediction
        predict_signal(model, latest_features)

        # Fetch real-time price info
        price_info = fetch_realtime(symbol)
        if price_info:
            print(f"💰 Current {symbol} price: {price_info['price']}")
            print(f"📊 24h High: {price_info['high_24h']} | Low: {price_info['low_24h']}")
            print("🔄 Waiting 60 seconds before next prediction...\n")

        time.sleep(60)


# --------------------------------------------------
# 🏁 Entry Point
# --------------------------------------------------
if __name__ == "__main__":
    main("BTCUSD")
