from core.data import fetch_ohlcv
from .strategy import check_signal
from core.clock import wait_for_new_candle
from . import config
import time

position = None

TIMEFRAME_SECONDS = 3600

def run():
    global position

    while True:
        wait_for_new_candle(TIMEFRAME_SECONDS)
        df = fetch_ohlcv(config.SYMBOL, config.TIMEFRAME)

        signal = check_signal(
            df,
            config.EMA_FAST,
            config.EMA_SLOW
        )

        if signal == "LONG" and position is None:
            position = "LONG"
            print("🟢 EMA CROSS LONG")

        elif signal == "EXIT" and position == "LONG":
            position = None
            print("🔴 EMA CROSS EXIT")

        time.sleep(60)
