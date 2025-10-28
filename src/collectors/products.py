# File: src/collectors/products.py
# Purpose: Fetch and manage trading pairs or asset metadata

# src/collector/products.py
import os
import pandas as pd
from pathlib import Path
from src.utils.http_client import safe_get
from src.config import BASE_URL, QUOTE_CURRENCY, CONTRACT_TYPE_KEYWORDS_LIST, DATA_PATH

def get_all_products() -> pd.DataFrame:
    """
    Fetch products and filter by contract type keywords & quote currency.
    Automatically detects variations like 'perpetual', 'futures', etc.
    """
    url = f"{BASE_URL}/products"
    print(f"[DEBUG] Fetching products from URL: {url}")
    data = safe_get(url)

    products = data.get("result", [])
    print(f"[DEBUG] Number of products fetched: {len(products)}")

    df = pd.DataFrame(products)
    print(f"[DEBUG] DataFrame shape: {df.shape}")
    print(f"[DEBUG] DataFrame columns: {df.columns.tolist()}")

    if df.empty:
        print("[WARN] No products returned")
        return df

    # Extract settling asset symbol safely
    df["settling_asset_symbol"] = df["settling_asset"].apply(
        lambda x: x.get("symbol") if isinstance(x, dict) else None
    )

    # Print unique contract types and settling assets
    print("[DEBUG] Unique contract types in API:", df['contract_type'].unique())
    print("[DEBUG] Unique settling asset symbols in API:", df['settling_asset_symbol'].unique())

    # Filter for relevant contract types and quote currency
    mask_contract = df["contract_type"].str.lower().apply(
        lambda x: any(kw in x for kw in CONTRACT_TYPE_KEYWORDS_LIST)
    )
    mask_quote = df["settling_asset_symbol"].str.upper() == QUOTE_CURRENCY.upper()
    mask = mask_contract & mask_quote

    print(f"[DEBUG] Number of products after filter: {mask.sum()}")

    # Keep only useful columns (existing ones)
    columns_to_keep = [
        "symbol", "description", "underlying_asset", "status",
        "tick_size", "contract_type", "settling_asset_symbol",
        "maker_commission_rate", "taker_commission_rate",
        "initial_margin", "maintenance_margin"
    ]
    existing_columns = [c for c in columns_to_keep if c in df.columns]
    filtered = df.loc[mask, existing_columns].copy()

    # Normalize symbol for candle API
    if "symbol" in filtered.columns:
        filtered["symbol"] = filtered["symbol"].str.replace("USDT", "USD", regex=False)

    print(f"[INFO] Filtered products: {len(filtered)}")
    if not filtered.empty:
        print(f"[DEBUG] Sample filtered products:\n{filtered.head(5)}")
    return filtered


def save_products(df: pd.DataFrame, fmt="csv"):
    """
    Save filtered products to data/products_processed.(csv|parquet)
    """
    os.makedirs(DATA_PATH / "processed", exist_ok=True)
    path = DATA_PATH / "processed" / f"products_processed.{fmt}"

    print(f"[DEBUG] Saving products to path: {path} (format: {fmt})")

    if fmt.lower() == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)

    print(f"[INFO] Saved products → {path}")


# ===================== STANDALONE TEST =====================
if __name__ == "__main__":
    print("🚀 Fetching products from Delta Exchange...")
    products_df = get_all_products()

    if products_df.empty:
        print("[ERROR] No products fetched.")
    else:
        print(products_df.head(10))  # Show first 10 products
        save_products(products_df, fmt="csv")
        save_products(products_df, fmt="parquet")
