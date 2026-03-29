from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from src.core.analysis.trade_explainability import (
    generate_trade_explainability,
    generate_trade_explainability_for_strategy,
)


pytestmark = pytest.mark.regression


FULL_MODE_STRATEGIES = ("ema_cross", "bmsb", "rsi_reversion")
FALLBACK_STRATEGY = "ema_trend_hold"
ZERO_TRADE_EDGE_STRATEGY = "donchian_breakout"


def _run_strategy(strategy: str, tmp_path: Path, rows: int = 430, max_trades: int = 2) -> dict:
    output_dir = tmp_path / strategy
    return generate_trade_explainability_for_strategy(
        strategy=strategy,
        rows=rows,
        context_bars=20,
        max_trades=max_trades,
        timeframe_label="1d",
        high_contrast=False,
        output_dir=output_dir,
    )


def _load_indexes(result: dict) -> tuple[pd.DataFrame, dict]:
    csv_path = Path(result["csv_index"])
    json_path = Path(result["json_index"])

    assert csv_path.exists(), f"missing csv index: {csv_path}"
    assert json_path.exists(), f"missing json index: {json_path}"

    return pd.read_csv(csv_path), json.loads(json_path.read_text(encoding="utf-8"))


def _assert_explanation_contract(explanation_path: Path, expected_mode: str) -> None:
    text = explanation_path.read_text(encoding="utf-8")

    assert f"mode: {expected_mode}" in text
    assert "entry_indicator_values" in text
    assert "entry_conditions" in text
    assert "exit_indicator_values" in text
    assert "exit_conditions" in text


def _assert_index_rows_map_to_files(output_dir: Path, index_df: pd.DataFrame) -> None:
    for row in index_df.to_dict(orient="records"):
        image_path = output_dir / str(row["image_file"])
        explanation_path = output_dir / str(row["explanation_file"])
        assert image_path.exists(), f"missing image artifact: {image_path}"
        assert explanation_path.exists(), f"missing explanation artifact: {explanation_path}"


@pytest.mark.parametrize("strategy", FULL_MODE_STRATEGIES)
def test_full_mode_strategy_contracts(strategy: str, tmp_path: Path) -> None:
    result = _run_strategy(strategy=strategy, tmp_path=tmp_path)
    index_df, payload = _load_indexes(result)
    output_dir = Path(result["output_dir"])

    assert not index_df.empty, f"expected at least one trade for strategy={strategy}"
    _assert_index_rows_map_to_files(output_dir=output_dir, index_df=index_df)

    fallback_series = index_df["fallback_mode"].astype(str).str.lower()
    assert (fallback_series == "false").all(), f"full-mode strategy unexpectedly used fallback={strategy}"

    for explanation_file in index_df["explanation_file"]:
        _assert_explanation_contract(output_dir / str(explanation_file), expected_mode="deterministic_full")

    assert int(payload["generated_trades"]) == len(index_df)
    assert int(payload["fallback_trades"]) == 0


def test_fallback_strategy_contracts_when_trades_exist(tmp_path: Path) -> None:
    result = _run_strategy(strategy=FALLBACK_STRATEGY, tmp_path=tmp_path)
    index_df, payload = _load_indexes(result)
    output_dir = Path(result["output_dir"])

    assert not index_df.empty, "expected at least one trade for fallback strategy"
    _assert_index_rows_map_to_files(output_dir=output_dir, index_df=index_df)

    fallback_series = index_df["fallback_mode"].astype(str).str.lower()
    assert (fallback_series == "true").all()

    for explanation_file in index_df["explanation_file"]:
        _assert_explanation_contract(output_dir / str(explanation_file), expected_mode="fallback")

    assert int(payload["generated_trades"]) == len(index_df)
    assert int(payload["fallback_trades"]) == len(index_df)


def test_donchian_zero_trade_contract_is_graceful(tmp_path: Path) -> None:
    result = _run_strategy(strategy=ZERO_TRADE_EDGE_STRATEGY, tmp_path=tmp_path)
    index_df, payload = _load_indexes(result)
    output_dir = Path(result["output_dir"])

    _assert_index_rows_map_to_files(output_dir=output_dir, index_df=index_df)

    generated_trades = int(payload["generated_trades"])
    assert generated_trades == len(index_df)

    if generated_trades == 0:
        assert int(payload["fallback_trades"]) == 0
        assert list(index_df.columns) == [
            "trade_number",
            "side",
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "net_pnl",
            "entry_trigger",
            "exit_trigger",
            "fallback_mode",
            "image_file",
            "explanation_file",
        ]
        assert list(output_dir.glob("trade_*.png")) == []
        assert list(output_dir.glob("trade_*.txt")) == []
        return

    fallback_series = index_df["fallback_mode"].astype(str).str.lower()
    assert (fallback_series == "false").all()
    for explanation_file in index_df["explanation_file"]:
        _assert_explanation_contract(output_dir / str(explanation_file), expected_mode="deterministic_full")


def test_run_level_summary_contract_for_all_strategies(tmp_path: Path) -> None:
    run_output_dir = tmp_path / "run_summary"
    result = generate_trade_explainability(
        strategy="all",
        rows=430,
        context_bars=30,
        max_trades=10,
        timeframe_label="1d",
        high_contrast=True,
        run_output_dir=run_output_dir,
    )

    summary_json_path = Path(result["summary_json"])
    summary_report_path = Path(result["summary_report"])

    assert summary_json_path.exists(), f"missing summary json: {summary_json_path}"
    assert summary_report_path.exists(), f"missing summary report: {summary_report_path}"

    payload = json.loads(summary_json_path.read_text(encoding="utf-8"))

    assert payload["run_metadata"] == {
        "rows": 430,
        "timeframe_label": "1d",
        "context_bars": 30,
        "max_trades": 10,
        "high_contrast": True,
    }
    assert payload["run_output_dir"] == str(run_output_dir)

    processed = payload["strategies_processed"]
    assert processed == [item["strategy"] for item in result["strategies"]]

    per_strategy = payload["per_strategy_counts"]
    assert len(per_strategy) == len(result["strategies"])
    assert payload["total_generated_trades"] == sum(int(item["generated_trades"]) for item in per_strategy)
    assert isinstance(payload["warnings"], list)

    for item in per_strategy:
        assert isinstance(item["requested_trades"], int)
        assert isinstance(item["generated_trades"], int)
        assert isinstance(item["fallback_trades"], int)
        assert isinstance(item["warnings"], list)

    report_text = summary_report_path.read_text(encoding="utf-8")
    assert "trade_explainability_run_summary" in report_text
    assert "total_generated_trades:" in report_text
    for strategy_name in processed:
        assert strategy_name in report_text
