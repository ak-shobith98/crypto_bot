# File: src/preprocess/preprocess.py
# Purpose: Full preprocessing workflow for historical candles.

from pathlib import Path
import pandas as pd
from src.preprocess.cleaner import clean_candles, resample_candles
from src.preprocess.features import add_technical_features
from src.config import DATA_PATH

def run_preprocessing() -> pd.DataFrame:
    """Load, clean, resample, and feature-engineer the candle data."""
    processed_dir = DATA_PATH / "processed"
    raw_dir = DATA_PATH / "processed"

    # Load historical candles
    input_file = raw_dir / "historical_candles.parquet"
    if not input_file.exists():
        raise FileNotFoundError(f"Missing {input_file}")

    print("🧹 Loading historical candle data...")
    df = pd.read_parquet(input_file)

    print("🧼 Cleaning data...")
    df = clean_candles(df)

    print("⏱️ Resampling candles...")
    df = resample_candles(df, freq="1h")

    print("🧮 Generating technical features...")
    df = add_technical_features(df)

    # Save final features
    output_file = processed_dir / "features.parquet"
    df.to_parquet(output_file, index=False)
    print(f"✅ Features saved to: {output_file}")

    return df
