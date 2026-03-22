import pandas as pd


def compute_trend_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def check_signal(price, trend_value, trend_period, current_side=None):
    if trend_value is None:
        return None, None

    if price > trend_value:
        return "LONG", f"Price above EMA{trend_period}"

    if current_side == "LONG" and price < trend_value:
        return "EXIT", f"Price below EMA{trend_period}"

    return None, None
