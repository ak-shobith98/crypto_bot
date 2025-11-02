# File: src/models/trainer.py
# Purpose: Train a reliable XGBoost classifier with realistic time-based splits and leakage-safe targets.

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path
from src.config import DATA_PATH

MODEL_PATH = DATA_PATH / "models" / "xgboost_latest.model"
SUMMARY_PATH = DATA_PATH / "models" / "training_summary.csv"
FEATURES_PATH = DATA_PATH / "processed" / "features.parquet"


# =========================================================
# 🎯 Generate Trading Target (Next Candle Return)
# =========================================================
def generate_targets(df: pd.DataFrame, up_th=0.002, down_th=-0.002):
    """
    Create target labels for each symbol:
        2 → BUY  (next close > +0.2%)
        1 → HOLD (neutral zone)
        0 → SELL (next close < -0.2%)
    """
    df = df.sort_values(["symbol", "timestamp"]).copy()

    # Future close (leakage-safe: uses next candle only)
    df["next_close"] = df.groupby("symbol")["close"].shift(-1)
    df["target_raw"] = (df["next_close"] - df["close"]) / df["close"]

    # Label encoding
    df["target"] = np.select(
        [df["target_raw"] > up_th, df["target_raw"] < down_th],
        [2, 0],
        default=1
    )

    # Drop last candle per symbol (no future target available)
    df = df.dropna(subset=["target_raw"]).reset_index(drop=True)
    return df


# =========================================================
# 🧩 Prepare Data for Model Training
# =========================================================
def prepare_data():
    print("📂 Loading features...")
    df = pd.read_parquet(FEATURES_PATH)
    df = generate_targets(df)

    print(f"✅ Loaded {len(df)} samples from {df['symbol'].nunique()} assets\n")
    print("🎯 Label distribution:")
    print(df["target"].value_counts(), "\n")

    # --- Feature set ---
    features = [
        "open", "high", "low", "close", "volume", "return",
        "sma_7", "ema_14", "volatility_14", "rsi_14",
        "bb_upper", "bb_lower", "macd", "macd_signal",
        "macd_hist", "obv", "vwap"
    ]

    # Ensure data is sorted chronologically to avoid look-ahead bias
    df = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)

    X_train, X_test = df.iloc[:split_idx][features], df.iloc[split_idx:][features]
    y_train, y_test = df.iloc[:split_idx]["target"], df.iloc[split_idx:]["target"]

    print(f"🕒 Data split by time → Train: {split_idx} | Test: {len(df) - split_idx}")
    return X_train, X_test, y_train, y_test


# =========================================================
# ⚙️ Train the Model
# =========================================================
def train_model():
    X_train, X_test, y_train, y_test = prepare_data()

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2,
        reg_alpha=1,
        min_child_weight=3,
        tree_method="hist",
        random_state=42,
        verbosity=0
    )

    print(f"\n🧩 Training samples: {len(X_train)} | Testing samples: {len(X_test)}")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc_test = accuracy_score(y_test, preds)
    acc_train = accuracy_score(y_train, model.predict(X_train))

    print(f"\n🧠 Train Accuracy: {acc_train:.4f} | Test Accuracy: {acc_test:.4f}\n")
    print(classification_report(y_test, preds))

    # Save model + summary
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    print(f"💾 Model saved → {MODEL_PATH}")

    # Save metrics
    report_df = pd.DataFrame(classification_report(y_test, preds, output_dict=True)).T
    report_df["train_acc"] = acc_train
    report_df["test_acc"] = acc_test
    report_df.to_csv(SUMMARY_PATH, index=True)
    print(f"📊 Metrics saved → {SUMMARY_PATH}")

    # Feature importance
    importances = model.get_booster().get_score(importance_type="gain")
    imp_df = (
        pd.DataFrame({"feature": list(importances.keys()), "importance": list(importances.values())})
        .sort_values("importance", ascending=False)
    )
    print("\n🏋️ Top 10 Important Features:")
    print(imp_df.head(10))


# =========================================================
# 🏁 Entry Point
# =========================================================
if __name__ == "__main__":
    train_model()
