import os
import sqlite3
import pandas as pd
from datetime import datetime, timezone

DB_PATH = "data/market_data.db"

REPORT_DIR = "data/reports"
os.makedirs(REPORT_DIR, exist_ok=True)

TIMEFRAME_TO_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "1d": 86_400_000
}


def sanitize_data(exchange="binance", symbol="BTC/USDT", timeframe="15m"):

    step_ms = TIMEFRAME_TO_MS.get(timeframe)
    if not step_ms:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    report_lines = []
    report_lines.append("===========================================")
    report_lines.append(" BINANCE OHLCV SANITIZATION REPORT")
    report_lines.append("===========================================")
    report_lines.append(f"Exchange  : {exchange}")
    report_lines.append(f"Symbol    : {symbol}")
    report_lines.append(f"Timeframe : {timeframe}")
    report_lines.append(f"Generated at: {datetime.now(timezone.utc).isoformat()}")

    report_lines.append("")

    conn = sqlite3.connect(DB_PATH)

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_sql_query("""
        SELECT exchange, symbol, timeframe, timestamp, open, high, low, close, volume
        FROM ohlcv
        WHERE exchange = ?
          AND symbol = ?
          AND timeframe = ?
        ORDER BY timestamp ASC
    """, conn, params=(exchange, symbol, timeframe))

    original_count = len(df)
    report_lines.append(f"Original rows loaded: {original_count}")

    if df.empty:
        report_lines.append("ERROR: No data found. Aborting.")
        report_path = os.path.join(REPORT_DIR, "sanitize_report_EMPTY.txt")
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines))
        return

    # Ensure numeric columns are numeric
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # -----------------------------
    # 1) Remove NaN rows
    # -----------------------------
    nan_before = len(df)
    df = df.dropna()
    nan_removed = nan_before - len(df)

    report_lines.append("")
    report_lines.append("Step 1 - Remove NaN rows")
    report_lines.append(f"Removed rows with NaN: {nan_removed}")

    # -----------------------------
    # 2) Remove duplicates
    # -----------------------------
    dup_before = len(df)
    df = df.drop_duplicates(subset=["exchange", "symbol", "timeframe", "timestamp"])
    dup_removed = dup_before - len(df)

    report_lines.append("")
    report_lines.append("Step 2 - Remove duplicates")
    report_lines.append(f"Duplicates removed: {dup_removed}")

    # -----------------------------
    # 3) Sort timestamps
    # -----------------------------
    df = df.sort_values("timestamp").reset_index(drop=True)

    report_lines.append("")
    report_lines.append("Step 3 - Sort by timestamp")
    report_lines.append("Sorted ascending by timestamp.")

    # -----------------------------
    # 4) Validate OHLC structure
    # -----------------------------
    invalid_rows = df[
        (df["high"] < df["low"]) |
        (df["open"] < df["low"]) |
        (df["open"] > df["high"]) |
        (df["close"] < df["low"]) |
        (df["close"] > df["high"]) |
        (df["volume"] < 0)
    ]

    invalid_count = len(invalid_rows)

    report_lines.append("")
    report_lines.append("Step 4 - Validate OHLCV logic")
    report_lines.append(f"Invalid candles removed: {invalid_count}")

    if invalid_count > 0:
        report_lines.append("Examples of invalid candles (max 10):")
        for _, row in invalid_rows.head(10).iterrows():
            ts = datetime.utcfromtimestamp(row["timestamp"] / 1000).isoformat()
            report_lines.append(
                f"  {ts} | O={row['open']} H={row['high']} L={row['low']} C={row['close']} V={row['volume']}"
            )

    df = df.drop(invalid_rows.index).reset_index(drop=True)

    # -----------------------------
    # 5) Detect and fill gaps
    # -----------------------------
    report_lines.append("")
    report_lines.append("Step 5 - Detect and fill gaps")

    filled_rows = []
    gaps_summary = []
    gap_count = 0

    for i in range(1, len(df)):
        prev_row = df.iloc[i - 1]
        curr_row = df.iloc[i]

        expected_ts = prev_row["timestamp"] + step_ms

        if expected_ts < curr_row["timestamp"]:
            # Gap detected
            gap_start = expected_ts
            gap_end = curr_row["timestamp"]
            missing = 0

            while expected_ts < curr_row["timestamp"]:
                missing += 1
                gap_count += 1

                last_close = prev_row["close"]

                filled_rows.append({
                    "exchange": exchange,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": expected_ts,
                    "open": last_close,
                    "high": last_close,
                    "low": last_close,
                    "close": last_close,
                    "volume": 0.0
                })

                expected_ts += step_ms

            gaps_summary.append((gap_start, gap_end, missing))

    report_lines.append(f"Gaps detected: {len(gaps_summary)}")
    report_lines.append(f"Missing candles inserted: {len(filled_rows)}")

    if gaps_summary:
        report_lines.append("")
        report_lines.append("Gap details:")
        for gap_start, gap_end, missing in gaps_summary[:30]:
            start_str = datetime.fromtimestamp(gap_start / 1000, timezone.utc).isoformat()
            end_str   = datetime.fromtimestamp(gap_end   / 1000, timezone.utc).isoformat()
            report_lines.append(f"  Gap from {start_str} -> {end_str} | Missing candles: {missing}")

        if len(gaps_summary) > 30:
            report_lines.append(f"  ... ({len(gaps_summary) - 30} more gaps omitted)")

    # Insert filled candles
    if filled_rows:
        df_filled = pd.DataFrame(filled_rows)
        df = pd.concat([df, df_filled], ignore_index=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

    # -----------------------------
    # Final stats
    # -----------------------------
    final_count = len(df)
    report_lines.append("")
    report_lines.append("===========================================")
    report_lines.append(" FINAL SUMMARY")
    report_lines.append("===========================================")
    report_lines.append(f"Original candles : {original_count}")
    report_lines.append(f"Final candles    : {final_count}")
    report_lines.append(f"Removed NaNs     : {nan_removed}")
    report_lines.append(f"Removed dups     : {dup_removed}")
    report_lines.append(f"Removed invalid  : {invalid_count}")
    report_lines.append(f"Inserted gaps    : {len(filled_rows)}")

    start_ts = df["timestamp"].min()
    end_ts = df["timestamp"].max()

    start_dt = datetime.fromtimestamp(start_ts / 1000, timezone.utc)
    end_dt   = datetime.fromtimestamp(end_ts   / 1000, timezone.utc)

    report_lines.append("")
    report_lines.append("Dataset time range:")

    report_lines.append(f"Start: {start_dt.isoformat()}")
    report_lines.append(f"End  : {end_dt.isoformat()}")

    # -----------------------------
    # Save clean data to DB table
    # -----------------------------
    report_lines.append("")
    report_lines.append("Saving sanitized candles into table: ohlcv_clean")

    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS ohlcv_clean")

    cur.execute("""
        CREATE TABLE ohlcv_clean (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            exchange TEXT,
            symbol TEXT,
            timeframe TEXT,
            timestamp INTEGER,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL
        )
    """)

    conn.commit()

    df.to_sql("ohlcv_clean", conn, if_exists="append", index=False)

    report_lines.append(f"Inserted rows into ohlcv_clean: {len(df)}")

    conn.commit()
    conn.close()


    # -----------------------------
    # Write report to TXT
    # -----------------------------
    now_utc = datetime.now(timezone.utc)
    now_str = now_utc.strftime("%Y%m%d_%H%M%S")

    safe_symbol = symbol.replace("/", "_")

    report_path = os.path.join(
        REPORT_DIR,
        f"sanitize_report_{exchange}_{safe_symbol}_{timeframe}_{now_str}.txt"
    )

    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"[OK] Sanitization finished. Report saved to: {report_path}")



if __name__ == "__main__":
    sanitize_data(
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="15m"
    )
