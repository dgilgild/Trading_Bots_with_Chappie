# src/core/data_loader.py

import pandas as pd
from src.core.database import get_connection


def load_ohlcv(
    exchange: str,
    symbol: str,
    timeframe: str,
    start_ts: int = None,
    end_ts: int = None
) -> pd.DataFrame:
    """
    Loads OHLCV from SQLite and returns a pandas DataFrame.
    timestamps are in milliseconds UTC.

    start_ts, end_ts: optional (ms)
    """

    conn = get_connection()

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

    query += " ORDER BY timestamp ASC"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if df.empty:
        return df

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("datetime", inplace=True)

    return df
