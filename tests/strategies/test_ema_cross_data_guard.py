from __future__ import annotations

import pandas as pd
import pytest

import src.strategies.ema_cross.backtest_ema_cross_v2 as ema_runner


@pytest.mark.parametrize(
    ("input_df", "expected_reason_fragment"),
    [
        (pd.DataFrame(), "No OHLCV data found"),
        (
            pd.DataFrame(
                {
                    "timestamp": ["2024-01-01"],
                    "open": [100.0],
                    "high": [101.0],
                    "low": [99.0],
                    "volume": [10.0],
                }
            ),
            "missing required column(s): close",
        ),
    ],
)
def test_ema_cross_returns_no_data_stats_on_invalid_ohlc(
    input_df: pd.DataFrame,
    expected_reason_fragment: str,
    common_backtest_kwargs: dict,
    monkeypatch,
) -> None:
    def fake_fetch(**_kwargs):
        return input_df.copy()

    monkeypatch.setattr(ema_runner, "fetch_ohlcv", fake_fetch)

    stats, chart_path, csv_path = ema_runner.run_backtest_ema_cross_v2(
        **{**common_backtest_kwargs, "ema_fast": 20, "ema_slow": 50}
    )

    assert chart_path is None
    assert csv_path is None
    assert stats.get("Status") == "No data"
    assert int(stats.get("Total trades", -1)) == 0
    assert expected_reason_fragment in str(stats.get("No Data Reason", ""))
