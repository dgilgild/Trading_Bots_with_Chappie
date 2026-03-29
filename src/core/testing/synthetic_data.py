from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_ohlcv_v1(
    rows: int = 420,
    freq: str = "D",
    start: str = "2021-01-01",
) -> pd.DataFrame:
    """Build the canonical deterministic synthetic OHLCV dataset (v1).

    Contract:
    - Output columns are exactly: timestamp, open, high, low, close, volume.
    - `timestamp` is a pandas datetime64 column generated from `start`, `rows`, and `freq`.
    - Price columns are float series with deterministic trend+wave values.
    - `volume` is a float series with a constant value of 1000.0.

    Deterministic guarantee:
    - No random source is used.
    - Given identical `rows`, `freq`, and `start`, output is byte-for-byte stable
      across calls within the same pandas/numpy behavior.
    """
    idx = pd.date_range(start, periods=rows, freq=freq)
    x = np.arange(rows, dtype=float)

    trend = 100.0 + 0.05 * x
    wave = 2.5 * np.sin(x / 6.0) + 1.0 * np.cos(x / 17.0)
    close = trend + wave
    open_ = close + 0.2 * np.sin(x / 3.0)
    high = np.maximum(open_, close) + 0.6
    low = np.minimum(open_, close) - 0.6
    volume = np.full(rows, 1000.0)

    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
