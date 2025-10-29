# File: src/decision/signals.py
# Purpose: Generate numeric trading signals (1 = BUY, -1 = SELL, 0 = HOLD) with readable labels and save to file

import joblib
import pandas as pd
from pathlib import Path
from src.config import DATA_PATH

MODEL_PATH = DATA_PATH / "models" / "xgboost_latest.model"
OUTPUT_PATH = DATA_PATH / "result" / "signals.parquet"  # ✅ where signals are saved


def load_model():
    """Load the trained XGBoost model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")
    print(f"📦 Loading model from {MODEL_PATH}...")
    return joblib.load(MODEL_PATH)


def generate_signals(features: pd.DataFrame, prob_threshold: float = 0.55) -> pd.DataFrame:
    """
    Generate numeric and textual trade signals:
        signal → numeric (1 = BUY, -1 = SELL, 0 = HOLD)
        result → text ("BUY", "SELL", "HOLD")
    """
    model = load_model()

    if "symbol" not in features.columns:
        raise ValueError("❌ 'symbol' column missing in features dataset")

    all_signals = []

    for symbol, group in features.groupby("symbol"):
        drop_cols = ["timestamp", "symbol", "target"]
        X = group.drop(columns=[c for c in drop_cols if c in group.columns], errors="ignore")

        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else [0.5] * len(preds)

        for i, (pred, prob) in enumerate(zip(preds, probs)):
            if prob > prob_threshold:
                signal, result = 1, "BUY"
            elif prob < (1 - prob_threshold):
                signal, result = -1, "SELL"
            else:
                signal, result = 0, "HOLD"

            all_signals.append({
                "timestamp": group.iloc[i].get("timestamp", None),
                "symbol": symbol,
                "probability": round(float(prob), 4),
                "raw_pred": int(pred),
                "signal": signal,
                "result": result
            })

    signals_df = pd.DataFrame(all_signals)

    # ✅ Save to file
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    signals_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"✅ Saved {len(signals_df)} signals across {signals_df['symbol'].nunique()} symbols to {OUTPUT_PATH}")

    return signals_df


if __name__ == "__main__":
    features_path = DATA_PATH / "processed" / "features.parquet"
    if features_path.exists():
        df = pd.read_parquet(features_path)
        generate_signals(df)
    else:
        print("❌ Features file not found.")
