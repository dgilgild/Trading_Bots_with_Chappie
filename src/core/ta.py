import pandas as pd
import pandas_ta as ta


def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    return ta.rsi(series, length=int(period))


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    return ta.atr(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        length=int(period),
    )


def compute_adx(df: pd.DataFrame, period: int) -> pd.Series:
    adx_df = ta.adx(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        length=int(period),
    )
    if adx_df is None:
        return None
    adx_col = f"ADX_{int(period)}"
    return adx_df[adx_col] if adx_col in adx_df.columns else None
