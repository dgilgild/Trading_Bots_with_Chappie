import pandas as pd
import pandas_ta as ta


def keltner_reversion(
    df: pd.DataFrame,
    ema_length: int = 20,
    atr_length: int = 20,
    atr_mult: float = 1.5,
):
    ema = ta.ema(df["close"], length=int(ema_length))
    atr = ta.atr(df["high"], df["low"], df["close"], length=int(atr_length))

    if ema is None or atr is None or len(df) == 0:
        return None, None

    upper = ema + atr * float(atr_mult)
    lower = ema - atr * float(atr_mult)

    if upper.isna().iloc[-1] or lower.isna().iloc[-1] or ema.isna().iloc[-1]:
        return None, None

    price = df.iloc[-1]["close"]

    if price < lower.iloc[-1]:
        return "LONG", "KC_LONG"

    if price > upper.iloc[-1]:
        return "SHORT", "KC_SHORT"

    if price >= ema.iloc[-1]:
        return "EXIT", "KC_EXIT_LONG"

    if price <= ema.iloc[-1]:
        return "EXIT", "KC_EXIT_SHORT"

    return None, None
