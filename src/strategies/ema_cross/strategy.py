def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def check_signal(df, fast, slow):
    df["ema_fast"] = ema(df["close"], fast)
    df["ema_slow"] = ema(df["close"], slow)

    prev = df.iloc[-2]
    last = df.iloc[-1]

    if prev.ema_fast < prev.ema_slow and last.ema_fast > last.ema_slow:
        return "LONG"

    if prev.ema_fast > prev.ema_slow and last.ema_fast < last.ema_slow:
        return "EXIT"

    return None
