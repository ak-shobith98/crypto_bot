# File: src/models/trainer.py
# Purpose: Train an ML model (XGBoost) on engineered features to predict next price movement.

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib
from src.config import DATA_PATH


def prepare_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create target variable:
      1  → BUY  (next close > current close)
     -1  → SELL (next close < current close)
      0  → HOLD (same close)
    """
    df = df.copy()
    df["next_close"] = df.groupby("symbol")["close"].shift(-1)

    df["target"] = np.select(
        [
            df["next_close"] > df["close"],
            df["next_close"] < df["close"]
        ],
        [1, -1],
        default=0
    )

    df = df.dropna(subset=["target"])
    return df.drop(columns=["next_close"])


def train_model():
    """
    Train an XGBoost model using preprocessed feature data.
    """
    features_path = DATA_PATH / "processed" / "features.parquet"
    model_path = DATA_PATH / "models"
    model_path.mkdir(exist_ok=True, parents=True)

    if not features_path.exists():
        raise FileNotFoundError(f"❌ Features file not found: {features_path}")

    print("📂 Loading features...")
    df = pd.read_parquet(features_path)
    df = prepare_labels(df)

    print(f"✅ Loaded {len(df)} samples from {df['symbol'].nunique()} assets")

    # Drop non-feature columns
    drop_cols = ["timestamp", "symbol", "target"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    y = df["target"]

    # Map -1 → 0, 0 → 1, 1 → 2 for XGBoost training
    label_map = {-1: 0, 0: 1, 1: 2}
    y_mapped = y.map(label_map).astype(int)

    print(f"🧠 Label distribution (mapped):\n{y_mapped.value_counts().sort_index()}")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_mapped, test_size=0.2, shuffle=False
    )

    print(f"🧩 Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    # Train XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss",
        tree_method="hist",
        num_class=3
    )

    print("🚀 Training model...")
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)

    # Map back to original -1, 0, 1
    reverse_map = {0: -1, 1: 0, 2: 1}
    preds_original = pd.Series(preds).map(reverse_map)
    y_test_original = pd.Series(y_test).map(reverse_map)

    acc = accuracy_score(y_test_original, preds_original)
    print(f"✅ Accuracy: {acc:.4f}")
    print(classification_report(y_test_original, preds_original, digits=3))

    # Save model
    model_file = model_path / "xgboost_latest.model"
    joblib.dump(model, model_file)
    print(f"💾 Model saved to: {model_file}")

    # Save metrics summary
    metrics_path = model_path / "training_summary.csv"
    pd.DataFrame({
        "accuracy": [acc],
        "train_size": [len(X_train)],
        "test_size": [len(X_test)],
        "unique_symbols": [df["symbol"].nunique()]
    }).to_csv(metrics_path, index=False)
    print(f"📊 Metrics saved → {metrics_path}")

    return model, acc


if __name__ == "__main__":
    train_model()
