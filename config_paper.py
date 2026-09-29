"""Paper trading configuration — inherits from config.py, adds paper-specific settings."""
from config import *   # noqa: F401,F403

# ---------- Paper trading state ----------
PAPER_DB_PATH = "data/paper.db"

# ---------- Portfolio ----------
PAPER_STARTING_EQUITY = 1000.0
PAPER_CURRENCY = "USDT"

# ---------- Scheduler ----------
# How often to check for new 4H bar close (seconds)
CHECK_INTERVAL_SECONDS = 300   # 5 min

# ---------- Web UI ----------
WEB_REFRESH_SECONDS = 60       # auto-refresh page every 60s
