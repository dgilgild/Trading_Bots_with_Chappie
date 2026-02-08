import time

def wait_for_new_candle(timeframe_seconds):
    now = int(time.time())
    sleep_time = timeframe_seconds - (now % timeframe_seconds)
    time.sleep(sleep_time + 1)
