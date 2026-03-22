import pandas as pd


def compute_bmsb(df: pd.DataFrame, sma_period: int, ema_period: int) -> pd.DataFrame:
    df = df.copy()
    df["bmsb_sma"] = df["close"].rolling(int(sma_period)).mean()
    df["bmsb_ema"] = df["close"].ewm(span=int(ema_period), adjust=False).mean()
    df["bmsb"] = df[["bmsb_sma", "bmsb_ema"]].max(axis=1)
    return df


def compute_tensignal(df: pd.DataFrame, window: int) -> pd.Series:
    ten = (df["close"] > df["bmsb"]).astype(int).replace({0: -1})
    return ten.rolling(int(window)).sum()
