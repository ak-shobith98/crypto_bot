# =========================================================
# File: Backtest.py
# Purpose: Realistic trade simulation with TP/SL, fees, and per-symbol sequential testing
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.config import DATA_PATH

# === CONFIG ===
TP_PCT = 0.02              # +2% take profit
SL_PCT = -0.01             # -1% stop loss
FEE_PCT = 0.0005           # 0.05% per trade (entry+exit)
SLIPPAGE_PCT = 0.0003      # 0.03% price slippage
MIN_CONF = 0.55            # Minimum confidence to enter a trade
MAX_HOLD_BARS = 3          # Exit after this many HOLDs if no exit yet

# === Paths ===
SIGNALS_PATH = DATA_PATH / "result" / "signals.parquet"
OUTPUT_PATH = DATA_PATH / "result" / "backtest_trades.parquet"

# =====================================================
# 🧮 Backtest per symbol
# =====================================================
def backtest_symbol(df: pd.DataFrame):
    trades = []
    position = None
    hold_count = 0

    for i in range(len(df)):
        row = df.iloc[i]
        price = row["close"]
        signal = row["result"]
        p_buy, p_sell = row["p_buy"], row["p_sell"]
        conf = max(p_buy, p_sell)

        # === ENTRY ===
        if position is None:
            if signal == "BUY" and p_buy >= MIN_CONF:
                position = {"side": "BUY", "entry": price * (1 + SLIPPAGE_PCT), "entry_time": row["timestamp"]}
                hold_count = 0
            elif signal == "SELL" and p_sell >= MIN_CONF:
                position = {"side": "SELL", "entry": price * (1 - SLIPPAGE_PCT), "entry_time": row["timestamp"]}
                hold_count = 0
            continue

        # === EXIT ===
        entry = position["entry"]
        side = position["side"]
        pnl = (price - entry) / entry if side == "BUY" else (entry - price) / entry

        exit_reason = None
        if pnl >= TP_PCT:
            exit_reason = "TP"
        elif pnl <= SL_PCT:
            exit_reason = "SL"
        elif signal in ["BUY", "SELL"] and signal != side:
            exit_reason = "Flip"
        elif signal == "HOLD":
            hold_count += 1
            if hold_count >= MAX_HOLD_BARS:
                exit_reason = "Timeout"
        else:
            hold_count = 0

        if exit_reason:
            # Deduct fees and slippage
            pnl_after_fees = pnl - 2 * FEE_PCT
            pnl_pct = pnl_after_fees * 100
            trades.append({
                "symbol": row["symbol"],
                "entry_time": position["entry_time"],
                "exit_time": row["timestamp"],
                "entry_price": entry,
                "exit_price": price,
                "side": side,
                "pnl_pct": pnl_pct,
                "exit_reason": exit_reason,
            })
            position = None
            hold_count = 0

    return trades


# =====================================================
# 🚀 Run Backtest
# =====================================================
def run_backtest():
    print("📥 Loading signals and preparing data...")
    signals = pd.read_parquet(SIGNALS_PATH)

    if signals.empty:
        raise ValueError("❌ No signal data found — run signals.py first.")

    results = []
    for symbol, df in signals.groupby("symbol"):
        df = df.sort_values("timestamp").reset_index(drop=True)
        results.extend(backtest_symbol(df))

    results_df = pd.DataFrame(results)
    if results_df.empty:
        print("⚠️ No trades executed based on thresholds.")
        return

    # === Summary ===
    total = len(results_df)
    wins = (results_df["pnl_pct"] > 0).sum()
    losses = (results_df["pnl_pct"] <= 0).sum()
    win_rate = wins / total * 100
    avg_pnl = results_df["pnl_pct"].mean()
    total_pnl = results_df["pnl_pct"].sum()

    print("\n📊 Backtest Summary")
    print("-" * 60)
    print(f"Total Trades   : {total}")
    print(f"Winning Trades : {wins}")
    print(f"Losing Trades  : {losses}")
    print(f"✅ Win Rate     : {win_rate:.2f}%")
    print(f"📊 Avg PnL      : {avg_pnl:.3f}%")
    print(f"💰 Total Profit : {total_pnl:.2f}%")

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\n💾 Results saved → {OUTPUT_PATH}")

    # === Equity curve ===
    results_df = results_df.sort_values("exit_time")
    results_df["cum_pnl"] = results_df["pnl_pct"].cumsum()

    plt.figure(figsize=(10, 5))
    plt.plot(results_df["exit_time"], results_df["cum_pnl"])
    plt.title("📈 Cumulative Profit Curve (after fees & slippage)")
    plt.xlabel("Time")
    plt.ylabel("Cumulative PnL (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =====================================================
# 🏁 Entry Point
# =====================================================
if __name__ == "__main__":
    run_backtest()
