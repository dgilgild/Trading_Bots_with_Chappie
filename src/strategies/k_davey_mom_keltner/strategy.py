import pandas as pd
import pandas_ta as ta


def compute_keltner_stochastic(
    df: pd.DataFrame,
    length: int,
    atr_mult: float,
    mamode: str = "ema",
) -> pd.Series:
    kc = ta.kc(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        length=int(length),
        scalar=float(atr_mult),
        mamode=mamode,
    )
    if kc is None or kc.empty:
        return pd.Series(index=df.index, dtype=float)

    upper_col = next((c for c in kc.columns if c.startswith("KCU")), None)
    lower_col = next((c for c in kc.columns if c.startswith("KCL")), None)

    if upper_col is None or lower_col is None:
        return pd.Series(index=df.index, dtype=float)

    upper = kc[upper_col]
    lower = kc[lower_col]
    denom = (upper - lower).replace(0, pd.NA)
    if denom.isna().all():
        return pd.Series(index=df.index, dtype=float)

    kstoch = 100 * (df["close"] - lower) / denom
    return pd.to_numeric(kstoch, errors="coerce")


def compute_position_size(
    net_equity: float,
    base_equity: float,
    sizing_factor: float,
    max_contracts: int,
    use_position_sizing: bool,
) -> int:
    if not use_position_sizing:
        return 1

    ncons = sizing_factor * round(net_equity / base_equity, 0) + 1
    ncons = int(round(ncons))
    return min(max(ncons, 1), int(max_contracts))
