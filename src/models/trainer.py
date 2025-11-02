# File: src/models/trainer.py
# Purpose: Train XGBoost classifier with proper target logic, compatible across XGBoost versions.

import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from src.config import DATA_PATH

MODEL_PATH = DATA_PATH / "models" / "xgboost_latest.model"
SUMMARY_PATH = DATA_PATH / "models" / "training_summary.csv"
FEATURES_PATH = DATA_PATH / "processed" / "features.parquet"

# =========================
# 🧩 Load dataset
# =========================
def load_features():
    print("📂 Loading features...")
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"❌ Missing file: {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH)
    print(f"✅ Loaded {len(df)} samples from {df['symbol'].nunique()} assets")
    return df


# =========================
# 🎯 Create target variable
# =========================
def create_target(df: pd.DataFrame, up_thresh=0.003, down_thresh=-0.003):
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    df["future_close"] = df.groupby("symbol")["close"].shift(-1)
    df["future_return"] = (df["future_close"] - df["close"]) / df["close"]

    # Raw target (-1, 0, 1)
    df["target_raw"] = np.select(
        [
            df["future_return"] >= up_thresh,
            df["future_return"] <= down_thresh
        ],
        [1, -1],
        default=0
    )

    # Remap to (0, 1, 2) for XGBoost
    label_map = {-1: 0, 0: 1, 1: 2}
    df["target"] = df["target_raw"].map(label_map)

    df = df.dropna(subset=["future_return"])
    return df


# =========================
# 🧠 Train Model
# =========================
def train_model():
    df = load_features()
    df = create_target(df)

    # Drop non-numeric columns
    drop_cols = ["timestamp", "symbol", "future_close", "future_return"]
    X = df.drop(columns=drop_cols + ["target"], errors="ignore")
    y = df["target"]

    print("\n🎯 Label distribution:")
    print(y.value_counts())

    if X.select_dtypes(include=["number"]).shape[1] <= 6:
        print("⚠️ WARNING: Your features have only OHLCV columns.")
        print("👉 Run `python -m src.preprocess.features` to add technical indicators before training.")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    print(f"\n🧩 Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        learning_rate=0.05,
        n_estimators=400,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist"
    )

    # ✅ Compatible fit (works for all XGBoost versions)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"💾 Model saved → {MODEL_PATH}")

    # Save summary
    summary = pd.DataFrame({
        "metric": ["accuracy"],
        "value": [acc]
    })
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"📊 Metrics saved → {SUMMARY_PATH}")


# =========================
# 🏁 Entry Point
# =========================
if __name__ == "__main__":
    train_model()
