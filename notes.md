🧱 Step 1 — cleaner.py

    Basic cleaning and resampling logic:

    Remove nulls or duplicates

    Standardize timestamps

    Optionally, limit to top N symbols

🧮 Step 2 — features.py

    Compute classic technical indicators:

    Returns (close.pct_change())

    Moving averages (SMA, EMA)

    RSI, Bollinger Bands

    Volatility (rolling std)

    These will become model inputs.

| Category           | Feature                                 | Description                           | Typical Use                         |
| ------------------ | --------------------------------------- | ------------------------------------- | ----------------------------------- |
| **Trend**          | `macd`, `macd_signal`, `macd_hist`      | Moving Average Convergence Divergence | Detects trend changes               |
| **Momentum**       | `rsi_14`                                | Relative Strength Index               | Overbought / Oversold               |
| **Volatility**     | `volatility_14`, `bb_upper`, `bb_lower` | Std deviation, Bollinger Bands        | Range detection                     |
| **Volume-based**   | `obv`                                   | On-Balance Volume                     | Detects accumulation / distribution |
| **Price Strength** | `return`, `sma_7`, `ema_14`             | Core statistical features             | Smoothing & returns                 |
| **Liquidity**      | `vwap`                                  | Volume Weighted Average Price         | Realistic price anchor              |
    

⚙️ Step 3 — preprocess.py

    Integrate the cleaning + feature generation,
    save to data/processed/features.parquet.

    Then we’ll wire this into your pipeline:

    from src.preprocess.preprocess import run_preprocessing


    right after candle collection in main.py.

🧠 Stage 3 — Model Training

    Once features are ready:

    Create a simple ML model (e.g. Logistic Regression / RandomForest)

    Predict: next-hour trend (↑ / ↓)

    Save to data/predictions/model_output.parquet

🤖 Stage 4 — Decision Layer

    Generate trading signals:

    Define rules based on model confidence or indicators

    Log to console or file

    Later: Stream via WebSocket or Kafka for live trading

🧩 Stage 5 — Real-Time Integration

    Use FastAPI / Socket.IO to stream live updates,
    and Kafka (optional) for event-driven architecture.