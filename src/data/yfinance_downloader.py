# src/data/yfinance_downloader.py

from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from src.core.database import insert_ohlcv_rows


class YFinanceDownloader:
    def download_daily(self, ticker: str, symbol_label: str, start_date: str, end_date: str):
        df = yf.download(
            tickers=ticker,
            start=start_date,
            end=end_date,
            interval="1d",
            progress=False,
            auto_adjust=False,
        )

        if df is None or df.empty:
            return 0

        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]

        rows = []
        for _, row in df.iterrows():
            ts = int(
                pd.Timestamp(row["date"]).replace(tzinfo=timezone.utc).timestamp() * 1000
            )
            rows.append(
                (
                    "yfinance",
                    symbol_label,
                    "1d",
                    ts,
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row.get("volume", 0.0) or 0.0),
                )
            )

        insert_ohlcv_rows(rows)
        return len(rows)
