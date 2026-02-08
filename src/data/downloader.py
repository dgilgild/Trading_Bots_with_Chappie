# src/data/downloader.py

import ccxt
import time
from datetime import datetime, timezone

from src.core.database import insert_ohlcv_rows, init_db


class BinanceDownloader:
    def __init__(self):
        self.exchange = ccxt.binance({
            "enableRateLimit": True
        })
        self.exchange.load_markets()
        init_db()

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: int,
        limit: int = 1000
    ):
        return self.exchange.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=since,
            limit=limit
        )

    def download(
        self,
        symbol: str,
        timeframe: str,
        start_date: str
    ):
        """
        start_date: 'YYYY-MM-DD'
        """
        since = int(
            datetime.strptime(start_date, "%Y-%m-%d")
            .replace(tzinfo=timezone.utc)
            .timestamp() * 1000
        )

        now = self.exchange.milliseconds()
        all_rows = []

        print(f"Downloading {symbol} {timeframe} from {start_date}")

        while since < now:
            candles = self.fetch_ohlcv(symbol, timeframe, since)

            if not candles:
                break

            for c in candles:
                ts, o, h, l, cl, v = c
                all_rows.append((
                    "binance",
                    symbol,
                    timeframe,
                    ts,
                    o, h, l, cl, v
                ))

            since = candles[-1][0] + 1
            time.sleep(self.exchange.rateLimit / 1000)

        insert_ohlcv_rows(all_rows)
        print(f"Inserted {len(all_rows)} rows into SQLite")
