from __future__ import annotations

import pytest

from src.core.analysis.walk_forward import (
    SUPPORTED_WALK_FORWARD_STRATEGIES,
    WINDOW_OUTPUT_COLUMNS,
    load_walk_forward_data,
    plan_walk_forward_windows,
    resolve_window_lengths,
    run_walk_forward,
)


def test_percentage_lengths_and_validation() -> None:
    mode, train_len, test_len, step_len = resolve_window_lengths(
        total_bars=200,
        train_pct=50,
        test_pct=25,
        step_pct=25,
    )

    assert mode == "percentage"
    assert train_len == 100
    assert test_len == 50
    assert step_len == 50

    with pytest.raises(ValueError, match=r"train_pct \+ test_pct must be <= 100"):
        resolve_window_lengths(
            total_bars=200,
            train_pct=80,
            test_pct=30,
            step_pct=10,
        )

    with pytest.raises(ValueError, match="train_pct must be > 0"):
        resolve_window_lengths(
            total_bars=200,
            train_pct=0,
            test_pct=25,
            step_pct=25,
        )


def test_no_overlap_enforcement() -> None:
    with pytest.raises(ValueError, match="step_len must be >= test_len"):
        resolve_window_lengths(
            total_bars=200,
            train_pct=50,
            test_pct=30,
            step_pct=10,
        )


def test_window_boundaries_deterministic() -> None:
    windows = plan_walk_forward_windows(total_bars=20, train_len=10, test_len=5, step_len=5)

    assert len(windows) == 2

    first = windows[0]
    second = windows[1]

    assert (first.train_start, first.train_end, first.test_start, first.test_end) == (0, 10, 10, 15)
    assert (second.train_start, second.train_end, second.test_start, second.test_end) == (5, 15, 15, 20)


def test_summary_output_shape_small_deterministic_run() -> None:
    data_df = load_walk_forward_data(mode="deterministic", deterministic_rows=120)

    _mode, train_len, test_len, step_len = resolve_window_lengths(
        total_bars=len(data_df),
        train_pct=50,
        test_pct=25,
        step_pct=25,
    )

    windows_df, summary = run_walk_forward(
        strategy="ema_cross",
        data_mode="deterministic",
        data_df=data_df,
        train_len=train_len,
        test_len=test_len,
        step_len=step_len,
        window_mode="percentage",
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1d",
        ema_fast=5,
        ema_slow=10,
    )

    assert not windows_df.empty
    assert {
        "window_id",
        "train_start_idx",
        "train_end_idx",
        "test_start_idx",
        "test_end_idx",
        "total_trades",
        "total_net_profit",
        "profit_factor",
        "max_drawdown_pct",
    }.issubset(set(windows_df.columns))

    assert summary["window_count"] == len(windows_df)
    assert summary["strategy"] == "ema_cross"
    assert summary["data_mode"] == "deterministic"
    assert "aggregate_net_profit" in summary
    assert "total_trades" in summary


@pytest.mark.parametrize("strategy", SUPPORTED_WALK_FORWARD_STRATEGIES)
def test_all_supported_strategies_run_in_deterministic_mode(strategy: str) -> None:
    data_df = load_walk_forward_data(mode="deterministic", deterministic_rows=120)

    windows_df, summary = run_walk_forward(
        strategy=strategy,
        data_mode="deterministic",
        data_df=data_df,
        train_len=60,
        test_len=30,
        step_len=30,
        window_mode="fixed",
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1d",
        ema_fast=5,
        ema_slow=10,
    )

    assert not windows_df.empty
    assert list(windows_df.columns) == WINDOW_OUTPUT_COLUMNS
    assert summary["window_count"] == len(windows_df)
    assert summary["strategy"] == strategy
    assert summary["data_mode"] == "deterministic"


def test_output_shape_consistent_across_strategies() -> None:
    data_df = load_walk_forward_data(mode="deterministic", deterministic_rows=100)

    per_strategy_columns = []
    per_strategy_window_counts = []

    for strategy in SUPPORTED_WALK_FORWARD_STRATEGIES:
        windows_df, summary = run_walk_forward(
            strategy=strategy,
            data_mode="deterministic",
            data_df=data_df,
            train_len=50,
            test_len=25,
            step_len=25,
            window_mode="fixed",
            exchange="binance",
            symbol="BTC/USDT",
            timeframe="1d",
        )

        per_strategy_columns.append(list(windows_df.columns))
        per_strategy_window_counts.append(summary["window_count"])

    assert all(columns == WINDOW_OUTPUT_COLUMNS for columns in per_strategy_columns)
    assert len(set(per_strategy_window_counts)) == 1


def test_zero_trade_windows_are_handled_gracefully() -> None:
    data_df = load_walk_forward_data(mode="deterministic", deterministic_rows=90)

    windows_df, summary = run_walk_forward(
        strategy="donchian_breakout",
        data_mode="deterministic",
        data_df=data_df,
        train_len=45,
        test_len=15,
        step_len=15,
        window_mode="fixed",
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1d",
    )

    assert not windows_df.empty
    assert (windows_df["total_trades"] == 0).all()
    assert (windows_df["total_net_profit"] == 0.0).all()
    assert (windows_df["max_drawdown_pct"] == 0.0).all()
    assert summary["windows_with_trades"] == 0
    assert summary["total_trades"] == 0
    assert summary["aggregate_net_profit"] == 0.0
