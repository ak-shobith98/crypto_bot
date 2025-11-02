# File: Backtest.py
# Purpose: Robust backtesting with safe price merge, clear diagnostics, and metrics

import pandas as pd
import numpy as np
from pathlib import Path
from src.config import DATA_PATH
import matplotlib.pyplot as plt

# === CONFIG ===
CONFIDENCE_THRESHOLD = 0.60
TAKE_PROFIT = 0.02
STOP_LOSS = -0.01
HOLD_EXIT_AFTER = 3

# === Paths ===
SIGNALS_PATH = DATA_PATH / "result" / "signals.parquet"
FEATURES_PATH = DATA_PATH / "processed" / "features.parquet"

print("📥 Loading signals and prices...")
signals = pd.read_parquet(SIGNALS_PATH)
features = pd.read_parquet(FEATURES_PATH)

print(f"🔹 Signals columns: {list(signals.columns)}")
print(f"🔹 Features columns: {list(features.columns)}")

# =====================================================
# ✅ Safe merge with auto-repair
# =====================================================
if "close" in features.columns:
    merged = signals.merge(
        features[["timestamp", "symbol", "close"]],
        on=["timestamp", "symbol"],
        how="left",
        suffixes=("", "_feat")
    )
    print("✅ Merged 'close' from features file.")
else:
    merged = signals.copy()
    print("⚠️ No 'close' in features — using signals as-is.")

# Handle possible suffixes or missing columns
possible_close_cols = [c for c in merged.columns if "close" in c.lower()]
if len(possible_close_cols) == 0:
    raise ValueError("❌ No column with 'close' found even after merge.")

# Prefer exact 'close' else use first match
if "close" not in merged.columns:
    merged.rename(columns={possible_close_cols[0]: "close"}, inplace=True)
    print(f"⚙️ Using '{possible_close_cols[0]}' as 'close' column.")

signals = merged.dropna(subset=["close"]).reset_index(drop=True)
print(f"✅ Using {len(signals)} valid rows for backtesting.\n")

# =====================================================
# 🚀 Backtest Logic
# =====================================================
def backtest(signals: pd.DataFrame):
    trades = []
    for symbol, df in signals.groupby("symbol"):
        df = df.sort_values("timestamp").reset_index(drop=True)
        position = None
        hold_count = 0

        for _, row in df.iterrows():
            ts, price = row["timestamp"], row["close"]
            sig = row["result"]
            p_buy, p_sell = row.get("p_buy", 0), row.get("p_sell", 0)

            # === ENTRY ===
            if position is None:
                if sig == "BUY" and p_buy > CONFIDENCE_THRESHOLD:
                    position = {"dir": "BUY", "entry_price": price, "entry_time": ts}
                elif sig == "SELL" and p_sell > CONFIDENCE_THRESHOLD:
                    position = {"dir": "SELL", "entry_price": price, "entry_time": ts}
                continue

            # === ACTIVE POSITION ===
            entry = position["entry_price"]
            direction = position["dir"]
            pnl = (price - entry) / entry if direction == "BUY" else (entry - price) / entry

            exit_trade = False

            # exit rules
            if pnl >= TAKE_PROFIT or pnl <= STOP_LOSS:
                exit_trade = True
            elif sig == "HOLD":
                hold_count += 1
                if hold_count >= HOLD_EXIT_AFTER:
                    exit_trade = True
            elif (sig == "SELL" and direction == "BUY") or (sig == "BUY" and direction == "SELL"):
                exit_trade = True
            else:
                hold_count = 0

            if exit_trade:
                trades.append({
                    "symbol": symbol,
                    "entry_time": position["entry_time"],
                    "exit_time": ts,
                    "entry_price": entry,
                    "exit_price": price,
                    "direction": direction,
                    "pnl_pct": pnl * 100
                })
                position = None
                hold_count = 0

    return pd.DataFrame(trades)

# =====================================================
# 🧮 Run Backtest
# =====================================================
print("🚀 Running backtest...")
results = backtest(signals)

if results.empty:
    print("⚠️ No trades executed based on thresholds.")
else:
    total = len(results)
    wins = (results["pnl_pct"] > 0).sum()
    losses = total - wins
    win_rate = wins / total * 100
    avg_pnl = results["pnl_pct"].mean()
    total_profit = results["pnl_pct"].sum()

    print("\n📊 Backtest Summary")
    print("-" * 60)
    print(f"Total Trades   : {total}")
    print(f"Wins           : {wins}")
    print(f"Losses         : {losses}")
    print(f"✅ Win Rate     : {win_rate:.2f}%")
    print(f"📊 Avg PnL      : {avg_pnl:.3f}%")
    print(f"💰 Total Profit : {total_profit:.2f}%")

    out = DATA_PATH / "result" / "backtest_trades.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(out, index=False)
    print(f"\n💾 Saved trade log → {out}")

    results = results.sort_values("exit_time")
    results["cum_pnl"] = results["pnl_pct"].cumsum()
    plt.figure(figsize=(10, 5))
    plt.plot(results["exit_time"], results["cum_pnl"], label="Cumulative PnL", linewidth=2)
    plt.title("📈 Cumulative Profit Curve")
    plt.xlabel("Time")
    plt.ylabel("Cumulative PnL (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
