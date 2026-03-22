def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def check_signal(df, fast, slow, trend_period=200, current_side=None):
    if len(df) < max(slow, trend_period) + 1:
        return None, None

    df["ema_fast"] = ema(df["close"], fast)
    df["ema_slow"] = ema(df["close"], slow)
    df["ema_trend"] = ema(df["close"], trend_period)

    prev = df.iloc[-2]
    last = df.iloc[-1]

    if prev.ema_fast < prev.ema_slow and last.ema_fast > last.ema_slow:
        if last.close > last.ema_trend:
            return "LONG", f"EMA{fast} crossed ABOVE EMA{slow} + EMA{trend_period} filter"
        if current_side == "SHORT":
            return "EXIT", f"EMA{fast} crossed ABOVE EMA{slow} (exit short)"
        return None, f"Blocked LONG below EMA{trend_period}"

    if prev.ema_fast > prev.ema_slow and last.ema_fast < last.ema_slow:
        if last.close < last.ema_trend:
            return "SHORT", f"EMA{fast} crossed BELOW EMA{slow} + EMA{trend_period} filter"
        if current_side == "LONG":
            return "EXIT", f"EMA{fast} crossed BELOW EMA{slow} (exit long)"
        return None, f"Blocked SHORT above EMA{trend_period}"

    return None, None
