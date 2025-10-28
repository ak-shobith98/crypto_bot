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
    Create target variable: 1 if next close > current close else 0.
    """
    df = df.copy()
    df["target"] = (df.groupby("symbol")["close"].shift(-1) > df["close"]).astype(int)
    df = df.dropna(subset=["target"])
    return df


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

    # Drop non-feature columns
    drop_cols = ["timestamp", "symbol", "target"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    y = df["target"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
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
        eval_metric="logloss",
        tree_method="hist"
    )

    print("🚀 Training model...")
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

    # Save model
    model_file = model_path / "xgboost_latest.model"
    joblib.dump(model, model_file)
    print(f"💾 Model saved to: {model_file}")

    return model, acc


if __name__ == "__main__":
    train_model()
