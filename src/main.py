# File: src/pipeline/main.py
# Purpose: Orchestrate data collection, preprocessing, model training, and signal generation.

from pathlib import Path
import pandas as pd
from datetime import datetime, timezone

from src.collectors.products import get_all_products
from src.collectors.fetch_candles import fetch_all_symbols
from src.config import DATA_PATH
from src.preprocess.preprocess import run_preprocessing


# =========================================================
# 🧩 Data Collection Stage
# =========================================================
def run_data_collection(resolution='1h', days=7) -> pd.DataFrame:
    """
    Run full data collection workflow:
      1. Fetch product metadata
      2. Fetch historical candles
      3. Return combined DataFrame
    """
    print("📦 Step 1: Fetching available products...")
    products_df = get_all_products()
    print(f"✅ {len(products_df)} products fetched.")

    print("\n🕓 Step 2: Fetching historical candles...")
    candle_df = fetch_all_symbols(resolution=resolution, days=days)

    if candle_df.empty:
        print("❌ No candle data collected.")
        return pd.DataFrame()

    print(f"✅ Collected candles for {candle_df['symbol'].nunique()} symbols.")
    return candle_df


# =========================================================
# 🚀 Main Pipeline
# =========================================================
def main():
    """
    Main orchestrator pipeline:
      1. Data Collection
      2. Preprocessing & Feature Engineering
      3. Model Training
      4. Signal Generation
      5. Forward-test data isolation (latest hour)
    """
    print("🚀 Starting main pipeline...\n")

    # ===== STAGE 1: DATA COLLECTION =====
    candles = run_data_collection(resolution="1h", days=15)
    if candles.empty:
        print("❌ No data fetched, aborting pipeline.")
        return

    print("\n📊 Data Collection Summary:")
    print(f"Symbols collected : {candles['symbol'].nunique()}")
    print(f"Total candles     : {len(candles)}")

    # Detect and show date range
    time_col = "timestamp" if "timestamp" in candles.columns else "time"
    if time_col in candles.columns:
        print(f"Data range: {candles[time_col].min()} → {candles[time_col].max()}")
    else:
        print("⚠️ No timestamp column found in candle data.")

    # ===== STAGE 1.5: EXCLUDE LAST HOUR =====
    cutoff = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    print(f"\n⏳ Excluding current unfinished hour: {cutoff}")

    if "timestamp" in candles.columns:
        before_filter = len(candles)
        candles = candles[candles["timestamp"] < cutoff]
        removed = before_filter - len(candles)
        print(f"🧹 Removed {removed} recent rows (unfinished candles).")

    # Save forward-test data (last 30 candles per symbol for indicator context)
    print("\n📦 Creating forward-test dataset (last 30 candles per symbol)...")
    forward_test = (
        candles.groupby("symbol")
        .tail(30)  # 30 = enough lookback for RSI, EMA, Bollinger
        .reset_index(drop=True)
    )
    forward_path = DATA_PATH / "processed" / "forward_test.parquet"
    forward_test.to_parquet(forward_path, index=False)
    print(f"💾 Forward-test data saved → {forward_path} (shape={forward_test.shape})")

    # Save filtered candles
    processed_path = DATA_PATH / "processed" / "filtered_candles.parquet"
    candles.to_parquet(processed_path, index=False)
    print(f"💾 Filtered candles saved → {processed_path}")

    # ===== STAGE 2: PREPROCESSING =====
    print("\n🧹 Stage 2: Preprocessing & Feature Engineering...")
    features = run_preprocessing()
    if features is not None and not features.empty:
        print(f"✅ Feature dataset created. Shape: {features.shape}")
    else:
        print("⚠️ Feature generation failed or returned empty DataFrame.")

    # ===== STAGE 3: MODEL TRAINING =====
    print("\n🏋️ Stage 3: Training Model...")
    from src.models.trainer import train_model
    train_model()

    # ===== STAGE 4: SIGNAL GENERATION =====
    print("\n📈 Stage 4: Generating Numeric Trade Signals...")
    from src.decision.signals import generate_signals
    generate_signals()

    signals_path = DATA_PATH / "processed" / "signals.csv"
    print(f"💾 Signals saved → {signals_path}")

    print("\n✅ Pipeline completed successfully.")


# =========================================================
# 🏁 Entry Point
# =========================================================
if __name__ == "__main__":
    main()
