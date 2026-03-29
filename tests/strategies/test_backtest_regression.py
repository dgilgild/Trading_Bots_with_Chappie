from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.core.testing.contracts import validate_trade_csv_contract_path

pytestmark = pytest.mark.regression


def test_backtest_regression(backtest_jobs, synthetic_ohlcv, regression_expected, monkeypatch) -> None:
    def fake_fetch(**_kwargs):
        return synthetic_ohlcv.copy()

    for strategy_name, module, run_fn, kwargs in backtest_jobs:
        monkeypatch.setattr(module, "fetch_ohlcv", fake_fetch)
        stats, _chart_path, csv_rel_path = run_fn(**kwargs)

        spec = regression_expected[strategy_name]
        expected_trades = int(spec["total_trades"])

        if expected_trades == 0:
            assert stats == {}, f"{strategy_name}: expected empty stats when no trades"
            assert csv_rel_path is None, f"{strategy_name}: expected no CSV when no trades"
            continue

        assert "Total trades" in stats, f"{strategy_name}: missing Total trades stat"
        assert "Total Net Profit" in stats, f"{strategy_name}: missing Total Net Profit stat"
        assert int(stats["Total trades"]) == expected_trades, (
            f"{strategy_name}: expected {expected_trades} trades, got {stats['Total trades']}"
        )

        expected_net = float(spec["total_net_profit"])
        got_net = float(stats["Total Net Profit"])
        assert math.isclose(got_net, expected_net, rel_tol=0.0, abs_tol=1e-3), (
            f"{strategy_name}: expected net {expected_net}, got {got_net}"
        )

        assert csv_rel_path is not None, f"{strategy_name}: expected CSV path"
        csv_abs_path = Path(kwargs["base_path"]) / "static" / csv_rel_path
        assert csv_abs_path.exists(), f"{strategy_name}: CSV file missing on disk"

        trades_df = validate_trade_csv_contract_path(
            csv_abs_path,
            context=f"{strategy_name} trade CSV '{csv_abs_path.name}'",
            validate_exit_time_monotonic=True,
        )
        assert len(trades_df) == expected_trades, f"{strategy_name}: CSV rows mismatch expected trades"

        first_trigger = str(trades_df.iloc[0]["entry_trigger"])
        last_exit = str(trades_df.iloc[-1]["exit_trigger"])

        assert spec["first_entry_trigger_contains"] in first_trigger, (
            f"{strategy_name}: first entry trigger mismatch; "
            f"expected contains '{spec['first_entry_trigger_contains']}', got '{first_trigger}'"
        )
        assert spec["last_exit_trigger_contains"] in last_exit, (
            f"{strategy_name}: last exit trigger mismatch; "
            f"expected contains '{spec['last_exit_trigger_contains']}', got '{last_exit}'"
        )
