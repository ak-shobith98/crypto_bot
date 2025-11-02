# =========================================================
# File: src/decision/signals.py
# Purpose: Batch prediction across all symbols using trained model.
# =========================================================

import pandas as pd
import xgboost as xgb
from pathlib import Path
from src.config import DATA_PATH

# === Paths ===
MODEL_PATH = DATA_PATH / "models" / "xgboost_latest.model"
FEATURES_PATH = DATA_PATH / "processed" / "features.parquet"
OUTPUT_PATH = DATA_PATH / "result" / "signals.parquet"


# --------------------------------------------------------
# 🧠 Load Model
# --------------------------------------------------------
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}")

    print(f"📦 Loading XGBoost model → {MODEL_PATH}")
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    print("✅ Model loaded successfully.")
    return model


# --------------------------------------------------------
# 🧩 Prepare Data
# --------------------------------------------------------
def prepare_features():
    """Load and clean processed features used for inference."""
    print("📂 Loading features...")
    df = pd.read_parquet(FEATURES_PATH)

    expected_features = [
        "open", "high", "low", "close", "volume", "return",
        "sma_7", "ema_14", "volatility_14", "rsi_14",
        "bb_upper", "bb_lower", "macd", "macd_signal",
        "macd_hist", "obv", "vwap"
    ]

    # Keep only valid numeric features
    df = df.dropna(subset=expected_features)
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    print(f"✅ Loaded {len(df)} rows from {df['symbol'].nunique()} symbols.")
    return df, expected_features


# --------------------------------------------------------
# 🔮 Generate Signals
# --------------------------------------------------------
def generate_signals():
    df, feature_cols = prepare_features()
    model = load_model()

    X = df[feature_cols]
    probs = model.predict_proba(X)

    df["p_sell"] = probs[:, 0]
    df["p_hold"] = probs[:, 1]
    df["p_buy"] = probs[:, 2]

    # Determine the most likely action
    df["signal"] = probs.argmax(axis=1)
    df["result"] = df["signal"].map({0: "SELL", 1: "HOLD", 2: "BUY"})

    # Keep relevant output columns
    output = df[[
        "timestamp", "symbol", "close",
        "p_sell", "p_hold", "p_buy", "signal", "result"
    ]].copy()

    # Save to disk
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT_PATH, index=False)
    print(f"\n✅ Saved {len(output)} signals across {df['symbol'].nunique()} symbols → {OUTPUT_PATH}")

    print("\n📊 Sample Signals Preview:")
    print(output.head())


# --------------------------------------------------------
# 🏁 Entry Point
# --------------------------------------------------------
if __name__ == "__main__":
    generate_signals()
