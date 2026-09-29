"""Trend Following v5 configuration — 10 coins.
This is the proven best version (Return +12.3%, Sharpe 0.93 on 365 days).
"""
import os

# ---------- API ----------
BASE_URL = "https://api.coinex.com/v2"
REQUEST_TIMEOUT = 20

# ---------- Universe (10 coins) ----------
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT",
    "SOLUSDT", "DOGEUSDT", "ADAUSDT",
    "LINKUSDT", "AVAXUSDT", "LTCUSDT",
]

# ---------- Timeframe ----------
PERIOD = "4hour"
BAR_HOURS = 4
BARS_PER_DAY = 6

# ---------- Cache ----------
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ---------- Lookback ----------
LOOKBACK_DAYS_DEFAULT = 365
LOOKBACK_DAYS_MAX = 500
FUNDING_LOOKBACK_DAYS = 365

# ---------- Trend parameters (v5 proven) ----------
MA_WINDOW = 200
DONCHIAN_ENTRY = 20
DONCHIAN_EXIT = 10
ATR_WINDOW = 14
ATR_STOP_MULT = 3.0
ATR_TRAIL_MULT = 3.0
TRAIL_ACTIVATION_ATR = 1.5
BTC_BIAS_MA = 100
MIN_ATR_PCT = 0.005

# ---------- Portfolio / Risk ----------
RISK_PER_TRADE = 0.0075
MAX_POSITION_NOTIONAL = 1.5
MAX_CONCURRENT_POSITIONS = 3
MAX_GROSS_EXPOSURE = 2.0
DAILY_STOP = -0.03
COOLDOWN_BARS_AFTER_LOSS = 6

# ---------- Entry filters ----------
MIN_FUNDING_ADVERSE = 0.0010
FUNDING_LOOKBACK_HOURS = 24
MIN_HOLD_BARS = 3
MAX_HOLD_BARS = 240

# ---------- Regime filter ----------
BTC_SHOCK_1H = 0.030
BTC_SHOCK_1D = 0.070

# ---------- Liquidity ----------
MIN_DAILY_VOLUME_USD = 20_000_000
LIQUIDITY_LOOKBACK_DAYS = 7

# ---------- Correlation ----------
MAX_CORR = 0.85

# ---------- Costs ----------
FEE_MAKER = float(os.getenv("FEE_MAKER", "0.0003"))
FEE_TAKER = float(os.getenv("FEE_TAKER", "0.0005"))
SLIPPAGE = float(os.getenv("SLIPPAGE", "0.0003"))
ROUND_TRIP_COST = 4 * (FEE_MAKER + SLIPPAGE)

# ---------- Backtest defaults ----------
STARTING_EQUITY = 1000.0
