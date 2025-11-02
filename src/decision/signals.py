# File: src/decision/signals.py
# Purpose: Generate numeric trading signals (BUY=2, HOLD=1, SELL=0)
# with probability outputs for each class.

import joblib
import pandas as pd
from pathlib import Path
from src.config import DATA_PATH

MODEL_PATH = DATA_PATH / "models" / "xgboost_latest.model"
OUTPUT_PATH = DATA_PATH / "result" / "signals.parquet"


# --------------------------------------------------
# 🧠 Load trained model
# --------------------------------------------------
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")
    print(f"📦 Loading model from {MODEL_PATH}...")
    return joblib.load(MODEL_PATH)


# --------------------------------------------------
# 🔮 Generate signals from features
# --------------------------------------------------
def generate_signals(features: pd.DataFrame) -> pd.DataFrame:
    """
    Predict trade signals using all 3 class probabilities:
      0 → SELL
      1 → HOLD
      2 → BUY
    """
    model = load_model()

    if "symbol" not in features.columns:
        raise ValueError("❌ Missing 'symbol' column in features dataset")

    all_signals = []

    for symbol, group in features.groupby("symbol"):
        drop_cols = ["timestamp", "symbol", "target"]
        X = group.drop(columns=[c for c in drop_cols if c in group.columns], errors="ignore")

        # Predict probabilities for all classes
        probs = model.predict_proba(X)
        preds = probs.argmax(axis=1)  # 0/1/2

        for i in range(len(preds)):
            p_sell, p_hold, p_buy = probs[i]

            if preds[i] == 2:
                label, signal = "BUY", 1
            elif preds[i] == 0:
                label, signal = "SELL", -1
            else:
                label, signal = "HOLD", 0

            all_signals.append({
                "timestamp": group.iloc[i]["timestamp"],
                "symbol": symbol,
                "p_sell": round(float(p_sell), 4),
                "p_hold": round(float(p_hold), 4),
                "p_buy": round(float(p_buy), 4),
                "signal": signal,
                "result": label,
            })

    signals_df = pd.DataFrame(all_signals)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    signals_df.to_parquet(OUTPUT_PATH, index=False)

    print(f"✅ Saved {len(signals_df)} signals across {signals_df['symbol'].nunique()} symbols → {OUTPUT_PATH}")

    print("\n📊 Sample Signals Preview:")
    print(signals_df.head())

    return signals_df


# --------------------------------------------------
# 🏁 Entry point
# --------------------------------------------------
if __name__ == "__main__":
    features_path = DATA_PATH / "processed" / "features.parquet"
    if features_path.exists():
        df = pd.read_parquet(features_path)
        generate_signals(df)
    else:
        print("❌ Features file not found.")
