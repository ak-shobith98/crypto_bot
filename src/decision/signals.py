# File: src/decision/signals.py
# Purpose: Generate BUY / SELL / HOLD signals using trained XGBoost model.

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from pathlib import Path
from src.config import DATA_PATH

MODEL_PATH = DATA_PATH / "models" / "xgboost_latest.model"
FEATURES_PATH = DATA_PATH / "processed" / "features.parquet"
OUTPUT_PATH = DATA_PATH / "result" / "signals.parquet"


# -----------------------------------------------------
# 🧠 Load Model
# -----------------------------------------------------
def load_model():
    """Load XGBoost model — supports both native and joblib formats."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")

    try:
        model = xgb.XGBClassifier()
        model.load_model(str(MODEL_PATH))
        print("✅ Loaded native XGBoost model.")
        return model
    except Exception:
        try:
            model = joblib.load(MODEL_PATH)
            print("✅ Loaded legacy joblib model.")
            return model
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load model: {e}")


# -----------------------------------------------------
# 🔮 Generate Signals
# -----------------------------------------------------
def generate_signals(df: pd.DataFrame):
    model = load_model()

    features = [
        "open", "high", "low", "close", "volume", "return",
        "sma_7", "ema_14", "volatility_14", "rsi_14",
        "bb_upper", "bb_lower", "macd", "macd_signal",
        "macd_hist", "obv", "vwap"
    ]
    X = df[features]

    # Predict probabilities
    probs = model.predict_proba(X)
    preds = np.argmax(probs, axis=1)

    df["p_sell"] = probs[:, 0]
    df["p_hold"] = probs[:, 1]
    df["p_buy"] = probs[:, 2]

    df["signal"] = preds
    df["result"] = df["signal"].map({0: "SELL", 1: "HOLD", 2: "BUY"})

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"✅ Saved {len(df)} signals across {df['symbol'].nunique()} symbols → {OUTPUT_PATH}")

    print("\n📊 Sample Signals Preview:")
    print(df.head())


# -----------------------------------------------------
# 🏁 Entry Point
# -----------------------------------------------------
if __name__ == "__main__":
    df = pd.read_parquet(FEATURES_PATH)
    generate_signals(df)
