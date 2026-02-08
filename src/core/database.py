# src/core/database.py

import sqlite3
from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "market_data.db"

# --- Ensure data directory exists ---
DATA_DIR.mkdir(exist_ok=True)

# --- Connection helper ---
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# --- Initialize database ---
def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            exchange TEXT NOT NULL,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            UNIQUE(exchange, symbol, timeframe, timestamp)
        );
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_ohlcv_lookup
        ON ohlcv(exchange, symbol, timeframe, timestamp);
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS backtest_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT NOT NULL,
            exchange TEXT NOT NULL,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            start_ts INTEGER,
            end_ts INTEGER,
            params_json TEXT NOT NULL,
            stats_json TEXT NOT NULL,
            chart_path TEXT,
            created_at TEXT NOT NULL
        );
    """)

    conn.commit()
    conn.close()


# --- Insert OHLCV batch ---
def insert_ohlcv_rows(rows):
    """
    rows: iterable of tuples
    (exchange, symbol, timeframe, timestamp, open, high, low, close, volume)
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executemany("""
        INSERT OR IGNORE INTO ohlcv (
            exchange, symbol, timeframe, timestamp,
            open, high, low, close, volume
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, rows)

    conn.commit()
    conn.close()
