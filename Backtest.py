# File: Backtest.py
# Purpose: Backtest multi-class BUY/SELL/HOLD signals using model confidence

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from src.config import DATA_PATH

# === CONFIG ===
SIGNALS_PATH = DATA_PATH / "result" / "signals.parquet"
FEATURES_PATH = DATA_PATH / "processed" / "features.parquet"

CONFIDENCE_THRESHOLD = 0.70   # minimum confidence to take a trade
HOLD_EXIT_AFTER = 3           # if HOLD seen for 3 periods, close trade
TAKE_PROFIT = 0.02            # +2% gain triggers exit
STOP_LOSS = -0.01             # -1% loss triggers exit


# =========================
# 1️⃣ Load and Prepare Data
# =========================
def load_data():
    print("📥 Loading signals and prices...")
    signals = pd.read_parquet(SIGNALS_PATH)
    features = pd.read_parquet(FEATURES_PATH)

    if "close" not in features.columns:
        raise ValueError("❌ Features must include 'close' prices")

    signals = signals.merge(
        features[["timestamp", "symbol", "close"]],
        on=["timestamp", "symbol"],
        how="left"
    ).dropna(subset=["close"])

    return signals


# =========================
# 2️⃣ Backtest Logic
# =========================
def backtest(signals: pd.DataFrame):
    trades = []
    for symbol, df in signals.groupby("symbol"):
        df = df.sort_values("timestamp").reset_index(drop=True)
        current_position = None
        hold_counter = 0

        for _, row in df.iterrows():
            ts, close = row["timestamp"], row["close"]
            p_buy, p_sell, signal = row["p_buy"], row["p_sell"], row["result"]

            # === ENTRY ===
            if current_position is None:
                if signal == "BUY" and p_buy > CONFIDENCE_THRESHOLD:
                    current_position = {"type": "BUY", "entry": close, "entry_time": ts}
                elif signal == "SELL" and p_sell > CONFIDENCE_THRESHOLD:
                    current_position = {"type": "SELL", "entry": close, "entry_time": ts}
                continue

            # === EXIT ===
            entry = current_position["entry"]
            direction = current_position["type"]
            pnl = (close - entry) / entry if direction == "BUY" else (entry - close) / entry

            exit_trade = False
            if pnl >= TAKE_PROFIT or pnl <= STOP_LOSS:
                exit_trade = True
            elif signal == "HOLD":
                hold_counter += 1
                if hold_counter >= HOLD_EXIT_AFTER:
                    exit_trade = True
            else:
                hold_counter = 0

            if exit_trade:
                trades.append({
                    "symbol": symbol,
                    "entry_time": current_position["entry_time"],
                    "exit_time": ts,
                    "entry_price": entry,
                    "exit_price": close,
                    "direction": direction,
                    "pnl_pct": pnl * 100
                })
                current_position = None
                hold_counter = 0

    return pd.DataFrame(trades)


# =========================
# 3️⃣ Run Backtest
# =========================
if __name__ == "__main__":
    signals = load_data()
    print("🚀 Running smart backtest...")
    trade_df = backtest(signals)

    if trade_df.empty:
        print("⚠️ No trades executed based on current confidence thresholds.")
    else:
        win_rate = (trade_df["pnl_pct"] > 0).mean() * 100
        avg_pnl = trade_df["pnl_pct"].mean()
        profit_trades = trade_df[trade_df["pnl_pct"] > 0].shape[0]
        loss_trades = trade_df[trade_df["pnl_pct"] <= 0].shape[0]

        print("\n📈 Backtest Summary")
        print("-" * 50)
        print(f"Total Trades       : {len(trade_df)}")
        print(f"Winning Trades     : {profit_trades}")
        print(f"Losing Trades      : {loss_trades}")
        print(f"✅ Win Rate         : {win_rate:.2f}%")
        print(f"📊 Avg PnL          : {avg_pnl:.3f}%")
        print(f"💰 Total Profit (%) : {trade_df['pnl_pct'].sum():.2f}%")

        print("\n🔍 Sample of last 5 trades:")
        print(trade_df.tail())

        # === Save trade log ===
        out_path = DATA_PATH / "result" / "backtest_trades.parquet"
        trade_df.to_parquet(out_path, index=False)
        print(f"\n💾 Saved trade log → {out_path}")

        # === Per-Symbol Summary ===
        perf = trade_df.groupby("symbol")["pnl_pct"].mean().sort_values()
        print("\n🔍 Top 10 Performing Symbols:")
        print(perf.tail(10))
        print("\n🔍 Bottom 10 Performing Symbols:")
        print(perf.head(10))

        # === Cumulative Profit Curve ===
        trade_df["cum_pnl"] = trade_df["pnl_pct"].cumsum()
        plt.figure(figsize=(10, 5))
        plt.plot(trade_df["exit_time"], trade_df["cum_pnl"], linewidth=2)
        plt.title("📈 Cumulative Profit Curve")
        plt.xlabel("Time")
        plt.ylabel("Cumulative PnL (%)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
