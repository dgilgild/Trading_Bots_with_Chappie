from src.core.database import get_connection
from datetime import datetime, date
import pandas as pd

def date_to_ms(value):
    # None or empty/whitespace -> no filter
    if value is None:
        return None

    # If it's already a number, assume it's a timestamp
    if isinstance(value, (int, float)):
        v = int(value)
        # Heuristic: treat < 10**12 as seconds, convert to ms
        return v if v >= 10**12 else v * 1000

    # If it's a date/datetime object
    if isinstance(value, datetime):
        return int(value.timestamp() * 1000)
    if isinstance(value, date):
        dt = datetime.combine(value, datetime.min.time())
        return int(dt.timestamp() * 1000)

    # If it's a string, clean it up
    s = str(value).strip()
    if not s:
        return None

    # Try YYYY-MM-DD first
    try:
        dt = datetime.strptime(s, "%Y-%m-%d")
        return int(dt.timestamp() * 1000)
    except ValueError:
        pass

    # Try ISO-8601 (e.g., 2024-02-12T15:04:05 or with timezone)
    try:
        # Handle trailing 'Z' for UTC
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        return int(dt.timestamp() * 1000)
    except ValueError as e:
        raise ValueError(f"Invalid date '{value}'. Use 'YYYY-MM-DD' or an ISO-8601 datetime.") from e


def fetch_ohlcv(exchange, symbol, timeframe, start_date=None, end_date=None, limit=50000, use_clean=True):
    table = "ohlcv_clean" if use_clean else "ohlcv"

    with get_connection() as conn:
        cur = conn.cursor()

        start_ts = date_to_ms(start_date)
        end_ts = date_to_ms(end_date)

        # If end_date is provided as a plain date (YYYY-MM-DD), make it inclusive to end of that day
        if isinstance(end_date, str) and end_date.strip() and len(end_date.strip()) == 10:
            if end_ts is not None:
                end_ts += 24 * 60 * 60 * 1000 - 1  # add 23:59:59.999

        query = f"""
            SELECT timestamp, open, high, low, close, volume
            FROM {table}
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
        params.append(int(limit))

        cur.execute(query, tuple(params))
        rows = cur.fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df
