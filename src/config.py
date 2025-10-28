# src/config.py
import os
from dotenv import load_dotenv
from pathlib import Path
from pathlib import Path
# Load .env
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data"

BASE_URL = os.getenv("DELTA_API_BASE_URL")
REAL_TIME_BASE_URL = os.getenv("DELTA_REALTIME_URL")
QUOTE_CURRENCY = os.getenv("DELTA_QUOTE_CURRENCY", "USDT")
CONTRACT_TYPE = os.getenv("DELTA_CONTRACT_TYPE", "perpetual")
CONTRACT_TYPE_KEYWORDS_LIST = os.getenv(
    "DELTA_CONTRACT_TYPE_KEYWORDS", "perpetual,perpetual_futures"
).split(",")

DATA_PATH = Path(os.getenv("DATA_PATH", "data"))
RESOLUTION = os.getenv("RESOLUTION", "1h")
HIST_DAYS = int(os.getenv("HIST_DAYS", 7))
TOP_N = int(os.getenv("TOP_N", 5))
MIN_LIQUIDITY = int(os.getenv("MIN_LIQUIDITY", 1_000_000))
