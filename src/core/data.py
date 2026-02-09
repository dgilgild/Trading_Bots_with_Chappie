from src.core.database import get_connection
from datetime import datetime
import pandas as pd


def date_to_ms(date_str):
    if date_str is None:
        return None
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.timestamp() * 1000)


def fetch_ohlcv(exchange,symbol, timeframe, start_date=None, end_date=None, limit=50000, use_clean=True):
    table = "ohlcv_clean" if use_clean else "ohlcv"

    conn = get_connection()
    cur = conn.cursor()

    start_ts = date_to_ms(start_date)
    end_ts = date_to_ms(end_date)

    query = """
        SELECT timestamp, open, high, low, close, volume
        FROM ohlcv
        WHERE exchange = ?
          AND symbol = ?
          AND timeframe = ?
    """
    params = [exchange, symbol, timeframe]

    if start_ts is not None:
        query += " AND timestamp >= ?"
        params.append(start_ts)

    if end_ts is not None:
        query += " AND timestamp <= ?"
        params.append(end_ts)

    query += " ORDER BY timestamp ASC LIMIT ?"
    params.append(limit)

    cur.execute(query, tuple(params))
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)

    return df