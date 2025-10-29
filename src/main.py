# File: src/pipeline/main.py
# Purpose: Orchestrate data collection, preprocessing, model, and decision layers.

from pathlib import Path
import pandas as pd
from src.collectors.products import get_all_products
from src.collectors.fetch_candles import fetch_all_symbols
from src.config import DATA_PATH
from src.preprocess.preprocess import run_preprocessing


def run_data_collection(resolution='1h', days=7) -> pd.DataFrame:
    """
    Run full data collection workflow:
    1. Fetch product metadata
    2. Fetch historical candles
    3. Save results
    """
    print("📦 Step 1: Fetching available products...")
    products_df = get_all_products()
    print(f"✅ {len(products_df)} products fetched.")

    print("\n🕓 Step 2: Fetching historical candles...")
    candle_df = fetch_all_symbols(resolution=resolution, days=days)

    if candle_df.empty:
        print("[ERROR] No candle data collected.")
        return pd.DataFrame()

    print(f"✅ Collected candles for {candle_df['symbol'].nunique()} symbols.")
    return candle_df


def main():
    """
    Main orchestrator pipeline.
    This is where all stages (collection → preprocessing → modeling → decisions) connect.
    """
    print("🚀 Starting main pipeline...\n")

    # # ===== STAGE 1: Data Collection =====
    candles = run_data_collection(resolution="1h", days=15)
    if candles.empty:
        print("❌ No data fetched, aborting pipeline.")
        return

    print("\n📊 Data Collection Summary:")
    print(f"Symbols collected: {candles['symbol'].nunique()}")
    print(f"Total candles: {len(candles)}")
    # Handle different possible time column names
    time_col = "timestamp" if "timestamp" in candles.columns else "time" if "time" in candles.columns else None
    if time_col:
        print(f"Data range: {candles[time_col].min()} → {candles[time_col].max()}")
    else:
        print("⚠️ No timestamp column found in candle data.")


    # ===== STAGE 2: Preprocessing =====
    print("\n🧹 Stage 2: Preprocessing & Feature Engineering...")
    features = run_preprocessing()
    if features is not None and not features.empty:
        print(f"✅ Feature dataset created. Shape: {features.shape}")
    else:
        print("⚠️ Feature generation failed or returned empty DataFrame.")

    # ===== (Future) STAGE 3: Modeling =====
    from src.models.trainer import train_model
    train_model()

    # ===== (Future) STAGE 4: Decision Layer =====
    from src.decision.signals import generate_signals

    print("\n📈 Stage 4: Generating numeric trade signals...")
    features_path = DATA_PATH / "processed" / "features.parquet"
    features = pd.read_parquet(features_path)

    signals_df = generate_signals(features)
    signals_path = DATA_PATH / "processed" / "signals.csv"
    signals_df.to_csv(signals_path, index=False)
    print(f"💾 Saved signals to: {signals_path}")

if __name__ == "__main__":
    main()
