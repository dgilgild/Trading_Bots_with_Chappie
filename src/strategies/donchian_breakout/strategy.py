import pandas as pd


def compute_donchian(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    df = df.copy()
    df["donchian_high"] = df["high"].rolling(lookback).max().shift(1)
    df["donchian_low"] = df["low"].rolling(lookback).min().shift(1)
    return df


def check_signal(df, lookback: int, current_side=None):
    if len(df) < lookback + 1:
        return None, None

    last = df.iloc[-1]

    if last.close > last.donchian_high:
        return "LONG", f"Donchian breakout above {lookback} high"

    if current_side == "LONG" and last.close < last.donchian_low:
        return "EXIT", f"Donchian exit below {lookback} low"

    return None, None
