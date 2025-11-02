# File: src/validate_forward_hourly.py
# Purpose: Validate model predictions on the most recent unseen hourly data (forward test).

import pandas as pd
from src.models.predictor import load_trained_model, prepare_features, predict_signal
from src.config import DATA_PATH

def run_forward_validation():
    """
    Load the trained model, apply it on forward-test data (latest candles),
    and print predicted BUY/SELL/HOLD signal with confidence.
    """
    print("🚀 Running Forward Test Validation...\n")

    model = load_trained_model()

    forward_path = DATA_PATH / "processed" / "forward_test.parquet"
    if not forward_path.exists():
        print(f"❌ Forward test data not found at: {forward_path}")
        return

    forward_test = pd.read_parquet(forward_path)
    if forward_test.empty:
        print("⚠️ Forward test file is empty.")
        return

    # ✅ Handle missing timestamp
    if "timestamp" not in forward_test.columns:
        if "time" in forward_test.columns:
            forward_test = forward_test.rename(columns={"time": "timestamp"})
        elif forward_test.index.name == "timestamp":
            forward_test = forward_test.reset_index()
        else:
            print("⚠️ No 'timestamp' column found, creating one from index.")
            forward_test = forward_test.reset_index().rename(columns={"index": "timestamp"})

    print(f"📦 Forward test samples loaded: {len(forward_test)}")

    # Prepare features
    ft_features = prepare_features(forward_test)

    if ft_features.empty:
        print("⚠️ No usable feature rows in forward test data.")
        return

    # Predict signal for latest row
    signal = predict_signal(model, ft_features)

    print("\n🧠 Forward Test Prediction Result:")
    if signal:
        print(f"Symbol:      {signal['symbol']}")
        print(f"Timestamp:   {signal['timestamp']}")
        print(f"Result:      {signal['result']}")
        print(f"Confidence:  {signal['confidence']}")
        print(f"p_buy:       {signal['p_buy']}")
        print(f"p_hold:      {signal['p_hold']}")
        print(f"p_sell:      {signal['p_sell']}")
    else:
        print("⚠️ No signal could be generated from forward test data.")


# 🏁 Entry Point
if __name__ == "__main__":
    run_forward_validation()
