"""SQLite persistence for paper trading state."""
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import List, Optional

from config_paper import PAPER_DB_PATH, PAPER_STARTING_EQUITY


SCHEMA = """
CREATE TABLE IF NOT EXISTS state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sym TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_ts TEXT NOT NULL,
    entry_price REAL NOT NULL,
    stop REAL NOT NULL,
    initial_stop REAL NOT NULL,
    atr REAL NOT NULL,
    notional REAL NOT NULL,
    bars INTEGER DEFAULT 0,
    pnl REAL DEFAULT 0.0,
    funding_pnl REAL DEFAULT 0.0,
    opened_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sym TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_ts TEXT NOT NULL,
    exit_ts TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL NOT NULL,
    notional REAL NOT NULL,
    pnl REAL NOT NULL,
    cost REAL NOT NULL,
    net_pnl REAL NOT NULL,
    reason TEXT NOT NULL,
    bars INTEGER NOT NULL,
    equity_after REAL NOT NULL,
    closed_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS equity_curve (
    ts REAL PRIMARY KEY,
    equity REAL NOT NULL
);
"""


def _ensure_dir():
    d = os.path.dirname(PAPER_DB_PATH)
    if d:
        os.makedirs(d, exist_ok=True)


@contextmanager
def get_conn():
    _ensure_dir()
    conn = sqlite3.connect(PAPER_DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as c:
        c.executescript(SCHEMA)
    # Initialize equity if missing
    if get_state("equity") is None:
        set_state("equity", str(PAPER_STARTING_EQUITY))
        set_state("peak_equity", str(PAPER_STARTING_EQUITY))
    if get_state("last_bar_ts") is None:
        set_state("last_bar_ts", "0")


# ---------- State (key-value) ----------
def get_state(key, default=None):
    with get_conn() as c:
        row = c.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default


def set_state(key, value):
    with get_conn() as c:
        c.execute("INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)",
                  (key, str(value)))


# ---------- Positions ----------
def get_open_positions() -> List[dict]:
    with get_conn() as c:
        rows = c.execute("SELECT * FROM positions ORDER BY opened_at").fetchall()
        return [dict(r) for r in rows]


def get_open_position(sym: str) -> Optional[dict]:
    with get_conn() as c:
        row = c.execute("SELECT * FROM positions WHERE sym=?", (sym,)).fetchone()
        return dict(row) if row else None


def insert_position(pos: dict):
    with get_conn() as c:
        c.execute("""
            INSERT INTO positions
                (sym, side, entry_ts, entry_price, stop, initial_stop,
                 atr, notional, bars, pnl, funding_pnl, opened_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 0.0, 0.0, ?)
        """, (pos["sym"], pos["side"], pos["entry_ts"], pos["entry_price"],
              pos["stop"], pos["initial_stop"], pos["atr"], pos["notional"],
              time.time()))


def update_position(sym: str, **fields):
    if not fields:
        return
    keys = list(fields.keys())
    set_clause = ", ".join(f"{k}=?" for k in keys)
    vals = [fields[k] for k in keys] + [sym]
    with get_conn() as c:
        c.execute(f"UPDATE positions SET {set_clause} WHERE sym=?", vals)


def delete_position(sym: str):
    with get_conn() as c:
        c.execute("DELETE FROM positions WHERE sym=?", (sym,))


# ---------- Trades ----------
def insert_trade(t: dict):
    with get_conn() as c:
        c.execute("""
            INSERT INTO trades
                (sym, side, entry_ts, exit_ts, entry_price, exit_price,
                 notional, pnl, cost, net_pnl, reason, bars, equity_after, closed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (t["sym"], t["side"], t["entry_ts"], t["exit_ts"],
              t["entry_price"], t["exit_price"], t["notional"],
              t["pnl"], t["cost"], t["net_pnl"], t["reason"],
              t["bars"], t["equity_after"], time.time()))


def get_trades(limit: int = 200) -> List[dict]:
    with get_conn() as c:
        rows = c.execute(
            "SELECT * FROM trades ORDER BY closed_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


# ---------- Equity curve ----------
def append_equity(ts: float, equity: float):
    with get_conn() as c:
        c.execute("INSERT OR REPLACE INTO equity_curve (ts, equity) VALUES (?, ?)",
                  (ts, equity))


def get_equity_curve(limit: int = 5000) -> List[dict]:
    with get_conn() as c:
        rows = c.execute(
            "SELECT ts, equity FROM equity_curve ORDER BY ts DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [dict(r) for r in reversed(rows)]
