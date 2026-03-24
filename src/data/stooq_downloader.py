# src/data/stooq_downloader.py

import io
from datetime import datetime, timezone

import pandas as pd
import requests

from src.core.database import insert_ohlcv_rows


class StooqDownloader:
    BASE_URL = "https://stooq.com/q/d/l/"

    def download_daily(self, symbol_code: str, symbol_label: str, start_date: str, end_date: str):
        params = {
            "s": symbol_code.lower(),
            "i": "d",
        }

        response = requests.get(self.BASE_URL, params=params, timeout=30)
        response.raise_for_status()

        df = pd.read_csv(io.StringIO(response.text))
        if df.empty:
            return 0

        df.columns = [c.strip().lower() for c in df.columns]
        if "date" not in df.columns:
            return 0

        df["date"] = pd.to_datetime(df["date"], utc=True)

        start_dt = pd.to_datetime(start_date, utc=True)
        end_dt = pd.to_datetime(end_date, utc=True)

        df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)].copy()
        if df.empty:
            return 0

        if "volume" not in df.columns:
            df["volume"] = 0.0

        rows = []
        for _, row in df.iterrows():
            ts = int(row["date"].replace(tzinfo=timezone.utc).timestamp() * 1000)
            rows.append(
                (
                    "stooq",
                    symbol_label,
                    "1d",
                    ts,
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row["volume"]),
                )
            )

        insert_ohlcv_rows(rows)
        return len(rows)
