# File: src/decision/signals.py
# Purpose: Generate numeric trading signals (1 = BUY, -1 = SELL, 0 = HOLD)
# Compatible with multi-class (BUY/HOLD/SELL) XGBoost model and ready for backtesting

import joblib
import pandas as pd
from pathlib import Path
from src.config import DATA_PATH

# === Paths ===
MODEL_PATH = DATA_PATH / "models" / "xgboost_latest.model"
OUTPUT_PATH = DATA_PATH / "result" / "signals.parquet"  # ✅ Output location


# --------------------------------------------------
# 🧠 Load Model
# --------------------------------------------------
def load_model():
    """Load the trained XGBoost model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")
    print(f"📦 Loading model from {MODEL_PATH}...")
    return joblib.load(MODEL_PATH)


# --------------------------------------------------
# ⚙️ Generate Trading Signals
# --------------------------------------------------
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

        # ✅ Ensure feature order matches training-time order
        if hasattr(model, "get_booster") and model.get_booster().feature_names:
            try:
                X = X[model.get_booster().feature_names]
            except KeyError:
                print(f"⚠️ Feature mismatch for {symbol}. Skipping missing features safely.")
                X = X.reindex(columns=model.get_booster().feature_names, fill_value=0)

        preds = model.predict(X)

        # ✅ Handle probabilities (multi-class safe)
        if hasattr(model, "predict_proba"):
            probs_all = model.predict_proba(X)
            probs = probs_all.max(axis=1)
        else:
            probs = [0.5] * len(preds)

        # ✅ Build signals
        for i, (pred, prob) in enumerate(zip(preds, probs)):
            if prob > prob_threshold and pred == 1:
                signal, result = 1, "BUY"
            elif prob > prob_threshold and pred == 2:
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

    # ✅ Create final DataFrame
    signals_df = pd.DataFrame(all_signals)

    # 🔍 Quick sanity check
    print("\n📊 Sample Signals Preview:")
    print(signals_df.head(5).to_string(index=False))

    # ✅ Save to file
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    signals_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\n✅ Saved {len(signals_df)} signals across {signals_df['symbol'].nunique()} symbols → {OUTPUT_PATH}")

    return signals_df


# --------------------------------------------------
# 🏁 Script Entry Point
# --------------------------------------------------
if __name__ == "__main__":
    features_path = DATA_PATH / "processed" / "features.parquet"

    if features_path.exists():
        df = pd.read_parquet(features_path)
        generate_signals(df)
    else:
        print("❌ Features file not found.")
