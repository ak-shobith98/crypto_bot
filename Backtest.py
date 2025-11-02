# File: src/backtest/run_backtest.py
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from pathlib import Path
from src.preprocess.features import add_technical_features
from src.config import DATA_PATH

MODEL_PATH = DATA_PATH / "models" / "xgboost_latest.model"
HISTORICAL_PATH = DATA_PATH / "processed" / "historical_candles.parquet"
RESULT_PATH = DATA_PATH / "result" / "backtest_results.parquet"

def load_model():
    """Load XGBoost or joblib model."""
    model = xgb.XGBClassifier()
    try:
        model.load_model(str(MODEL_PATH))
        print("✅ Loaded native XGBoost model.")
    except:
        model = joblib.load(MODEL_PATH)
        print("✅ Loaded via joblib.")
    return model

def run_backtest():
    print("🚀 Running backtest...")

    df = pd.read_parquet(HISTORICAL_PATH)
    df = add_technical_features(df)
    df = df.dropna().reset_index(drop=True)

    model = load_model()

    feature_cols = [
        "open", "high", "low", "close", "volume", "return",
        "sma_7", "ema_14", "volatility_14", "rsi_14",
        "bb_upper", "bb_lower", "macd", "macd_signal",
        "macd_hist", "obv", "vwap"
    ]

    X = df[feature_cols].fillna(0)
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else [0.5]*len(preds)

    df["signal"] = np.where(probs > 0.55, 1, np.where(probs < 0.45, -1, 0))
    df["prob"] = probs

    # 💰 Simple PnL calculation
    df["returns"] = df["close"].pct_change()
    df["strategy"] = df["signal"].shift(1) * df["returns"]
    df["cumulative_return"] = (1 + df["strategy"]).cumprod()

    # Save results
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RESULT_PATH, index=False)

    total_return = df["cumulative_return"].iloc[-1] - 1
    print(f"📈 Backtest complete. Total Return: {total_return:.2%}")
    print(f"✅ Results saved to: {RESULT_PATH}")

if __name__ == "__main__":
    run_backtest()
