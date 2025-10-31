import joblib
import pandas as pd

model = joblib.load("data/models/xgboost_latest.model")
features = pd.read_parquet("data/processed/features.parquet")

X = features.drop(columns=["timestamp", "symbol", "target"], errors="ignore")
probs = model.predict_proba(X)

print("Shape:", probs.shape)
print("Example row probabilities:")
print(probs[:5])
