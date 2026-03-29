from __future__ import annotations

import pandas as pd
import pytest

from src.core.testing.contracts import validate_trade_csv_contract


pytestmark = pytest.mark.unit


def _base_trades_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_time": [
                "2024-01-01T00:00:00Z",
                "2024-01-02T00:00:00Z",
                "2024-01-03T00:00:00Z",
            ],
            "exit_time": [
                "2024-01-01T12:00:00Z",
                "2024-01-02T12:00:00Z",
                "2024-01-03T12:00:00Z",
            ],
            "entry_trigger": ["e1", "e2", "e3"],
            "exit_trigger": ["x1", "x2", "x3"],
            "net_pnl": [1.0, -2.0, 3.0],
            "position_size": [1.0, 1.0, 1.0],
            "bars_in_trade": [4, 5, 6],
            "pyramid_level": [0, 0, 1],
        }
    )


def test_trade_csv_contract_timeline_valid_passes() -> None:
    trades_df = _base_trades_df()
    validate_trade_csv_contract(trades_df, context="timeline valid")


def test_trade_csv_contract_timeline_entry_after_exit_fails() -> None:
    trades_df = _base_trades_df()
    trades_df.loc[1, "entry_time"] = "2024-01-02T18:00:00Z"
    trades_df.loc[1, "exit_time"] = "2024-01-02T12:00:00Z"

    with pytest.raises(ValueError, match="timeline invalid: entry_time must be <= exit_time"):
        validate_trade_csv_contract(trades_df, context="timeline invalid")


def test_trade_csv_contract_exit_time_monotonic_toggle() -> None:
    trades_df = _base_trades_df()
    trades_df.loc[0, "exit_time"] = "2024-01-03T12:00:00Z"
    trades_df.loc[1, "exit_time"] = "2024-01-02T12:00:00Z"

    with pytest.raises(ValueError, match="timeline monotonic: exit_time must be monotonically"):
        validate_trade_csv_contract(
            trades_df,
            context="timeline monotonic",
            validate_exit_time_monotonic=True,
        )

    validate_trade_csv_contract(
        trades_df,
        context="timeline monotonic disabled",
        validate_exit_time_monotonic=False,
    )
