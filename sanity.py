import pandas as pd

df = pd.read_parquet("data/result/signals.parquet")

print("Total signals:", len(df))
print("Unique symbols:", df['symbol'].nunique())
print("\nSignal distribution:")
print(df['result'].value_counts())

# See recent predictions for a few symbols
print("\nRecent predictions:")
print(df.groupby("symbol").tail(3).head(10))
