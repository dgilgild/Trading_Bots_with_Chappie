import time

from src.core.clock import wait_for_new_candle
from src.core.data import fetch_ohlcv
from src.strategies.ema_cross import config
from src.strategies.ema_cross.strategy import check_signal

position = None


def timeframe_to_seconds(timeframe):
    unit = timeframe[-1]
    value = int(timeframe[:-1])

    if unit == "m":
        return value * 60
    if unit == "h":
        return value * 60 * 60
    if unit == "d":
        return value * 60 * 60 * 24

    raise ValueError(f"Unsupported timeframe: {timeframe}")

def run():
    global position

    timeframe_seconds = timeframe_to_seconds(config.TIMEFRAME)

    while True:
        wait_for_new_candle(timeframe_seconds)
        df = fetch_ohlcv(
            exchange=config.EXCHANGE,
            symbol=config.SYMBOL,
            timeframe=config.TIMEFRAME,
            use_clean=True,
        )

        if df is None or len(df) < 2:
            time.sleep(5)
            continue

        signal, trigger = check_signal(
            df,
            config.EMA_FAST,
            config.EMA_SLOW,
            current_side=position,
        )

        if signal == "LONG" and position != "LONG":
            position = "LONG"
            print("EMA CROSS LONG")

        elif signal == "SHORT" and position != "SHORT":
            position = "SHORT"
            print("EMA CROSS SHORT")

        elif signal == "EXIT":
            position = None
            print("EMA CROSS EXIT")
