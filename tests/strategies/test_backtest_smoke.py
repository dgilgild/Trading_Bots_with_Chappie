from __future__ import annotations

import pytest

pytestmark = pytest.mark.smoke


def test_backtest_smoke(backtest_jobs, synthetic_ohlcv, monkeypatch) -> None:
    def fake_fetch(**_kwargs):
        return synthetic_ohlcv.copy()

    for strategy_name, module, run_fn, kwargs in backtest_jobs:
        monkeypatch.setattr(module, "fetch_ohlcv", fake_fetch)
        stats, chart_path, csv_path = run_fn(**kwargs)

        assert isinstance(stats, dict), f"{strategy_name}: backtest should return stats dict"
        assert chart_path is None, f"{strategy_name}: chart_path should be None when generate_equity=False"
        assert csv_path is None or isinstance(csv_path, str), (
            f"{strategy_name}: csv_path should be None or string"
        )
