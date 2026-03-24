import pandas as pd


def compute_smas(df: pd.DataFrame, sma_fast: int, sma_slow: int) -> pd.DataFrame:
    df = df.copy()
    df["sma_fast"] = df["close"].rolling(int(sma_fast)).mean()
    df["sma_slow"] = df["close"].rolling(int(sma_slow)).mean()
    return df


def _has_positive_slope(series: pd.Series, bars: int) -> bool:
    if len(series) < bars + 1:
        return False
    recent = series.tail(bars + 1).values
    return all(recent[i] > recent[i - 1] for i in range(1, len(recent)))


def check_signal(df: pd.DataFrame, sma_fast: int, sma_slow: int, slope_bars: int, current_side=None):
    if len(df) < max(sma_fast, sma_slow, slope_bars) + 1:
        return None, None

    last = df.iloc[-1]
    slope_ok = _has_positive_slope(df["sma_fast"], int(slope_bars))

    if last.close > last.sma_slow and last.low <= last.sma_fast and last.close > last.sma_fast:
        if slope_ok:
            return "LONG", f"SMA{int(sma_fast)} touch + close above"

    if current_side == "LONG" and last.close < last.sma_fast:
        return "EXIT", f"Close below SMA{int(sma_fast)}"

    return None, None
