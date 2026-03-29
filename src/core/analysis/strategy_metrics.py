from __future__ import annotations

import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.testing.synthetic_data import make_synthetic_ohlcv_v1
import src.strategies.basic_keltner_reversion.backtest_basic_keltner_reversion_v2 as bk_runner
import src.strategies.bmsb.backtest_bmsb_v2 as bmsb_runner
import src.strategies.donchian_breakout.backtest_donchian_breakout_v2 as don_runner
import src.strategies.ema_cross.backtest_ema_cross_v2 as ema_runner
import src.strategies.ema_trend_hold.backtest_ema_trend_hold_v2 as trend_runner
import src.strategies.emalyarovich_smas.backtest_emalyarovich_smas_v2 as sma_runner
import src.strategies.k_davey_mom_keltner.backtest_k_davey_mom_keltner_v2 as kd_runner
import src.strategies.rsi_reversion.backtest_rsi_reversion_v2 as rsi_runner


DRAWDOWN_PROXY_FIELD = "drawdown_proxy"
DRAWDOWN_PROXY_DEFINITION = (
    "Approximation from per-trade net PnL only: min(cumulative_net_pnl - running_peak)"
)


@contextmanager
def patched_attr(module, attr_name, replacement):
    original = getattr(module, attr_name)
    setattr(module, attr_name, replacement)
    try:
        yield
    finally:
        setattr(module, attr_name, original)


def build_strategy_jobs(base_path: Path) -> list[tuple[str, object, object, dict[str, Any]]]:
    common = {
        "exchange": "binance",
        "symbol": "BTC/USDT",
        "timeframe": "1d",
        "start_date": "2021-01-01",
        "end_date": "2022-12-31",
        "use_clean": True,
        "run_id": "pdfgen",
        "generate_report": False,
        "generate_plots": False,
        "generate_equity": False,
        "base_path": str(base_path),
    }
    return [
        ("ema_cross", ema_runner, ema_runner.run_backtest_ema_cross_v2, {**common, "ema_fast": 20, "ema_slow": 50}),
        (
            "rsi_reversion",
            rsi_runner,
            rsi_runner.run_backtest_rsi_reversion_v2,
            {**common, "rsi_period": 14, "rsi_entry": 30, "rsi_exit": 50},
        ),
        (
            "donchian_breakout",
            don_runner,
            don_runner.run_backtest_donchian_breakout_v2,
            {**common, "donchian_lookback": 20},
        ),
        (
            "ema_trend_hold",
            trend_runner,
            trend_runner.run_backtest_ema_trend_hold_v2,
            {**common, "trend_ema": 200},
        ),
        (
            "bmsb",
            bmsb_runner,
            bmsb_runner.run_backtest_bmsb_v2,
            {**common, "sma_period": 20, "ema_period": 21, "tensignal_window": 3},
        ),
        (
            "emalyarovich_smas",
            sma_runner,
            sma_runner.run_backtest_emalyarovich_smas_v2,
            {**common, "sma_fast": 20, "sma_slow": 200, "slope_bars": 3},
        ),
        (
            "k_davey_mom_keltner",
            kd_runner,
            kd_runner.run_backtest_k_davey_mom_keltner_v2,
            {**common, "symbol": "MES", "position_mode": "fixed", "trade_size": 1.0},
        ),
        (
            "basic_keltner_reversion",
            bk_runner,
            bk_runner.run_backtest_basic_keltner_reversion_v2,
            {**common, "kc_ema_length": 20, "kc_atr_length": 20, "kc_atr_mult": 1.5},
        ),
    ]


def collect_trade_csvs(temp_runs_dir: Path, csv_dir: Path) -> dict[str, Path | None]:
    temp_runs_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    synthetic_df = make_synthetic_ohlcv_v1(rows=430, freq="D")

    def fake_fetch(**_kwargs):
        return synthetic_df.copy()

    csv_by_strategy: dict[str, Path | None] = {}
    for strategy_name, module, fn, kwargs in build_strategy_jobs(temp_runs_dir):
        with patched_attr(module, "fetch_ohlcv", fake_fetch):
            _stats, _chart, csv_rel = fn(**kwargs)

        dst = csv_dir / f"{strategy_name}.csv"
        if not csv_rel:
            if dst.exists():
                dst.unlink()
            csv_by_strategy[strategy_name] = None
            continue

        csv_abs = temp_runs_dir / "static" / csv_rel
        if not csv_abs.exists():
            if dst.exists():
                dst.unlink()
            csv_by_strategy[strategy_name] = None
            continue

        shutil.copyfile(csv_abs, dst)
        csv_by_strategy[strategy_name] = dst

    return csv_by_strategy


def _round_metric(value: float) -> float:
    return round(float(value), 6)


def compute_strategy_metrics(csv_path: Path | None) -> dict[str, float | int]:
    zero = {
        "trades": 0,
        "total_net_pnl": 0.0,
        "win_rate": 0.0,
        "avg_pnl": 0.0,
        "avg_bars_in_trade": 0.0,
        DRAWDOWN_PROXY_FIELD: 0.0,
    }
    if not csv_path or not csv_path.exists():
        return zero

    trades = pd.read_csv(csv_path)
    if trades.empty:
        return zero

    pnl = pd.to_numeric(trades.get("net_pnl", 0.0), errors="coerce").fillna(0.0)
    bars = pd.to_numeric(trades.get("bars_in_trade", 0.0), errors="coerce").fillna(0.0)
    cumulative = pnl.cumsum()
    running_peak = cumulative.cummax()
    drawdowns = cumulative - running_peak

    trade_count = int(len(trades))
    return {
        "trades": trade_count,
        "total_net_pnl": _round_metric(pnl.sum()),
        "win_rate": _round_metric((pnl > 0).mean() * 100.0),
        "avg_pnl": _round_metric(pnl.mean()),
        "avg_bars_in_trade": _round_metric(bars.mean()),
        DRAWDOWN_PROXY_FIELD: _round_metric(drawdowns.min()),
    }


def build_comparison_payload(csv_by_strategy: dict[str, Path | None]) -> dict[str, Any]:
    strategies: dict[str, dict[str, float | int]] = {}
    for strategy_name in sorted(csv_by_strategy):
        strategies[strategy_name] = compute_strategy_metrics(csv_by_strategy[strategy_name])

    ranked = sorted(
        strategies.items(),
        key=lambda item: (
            -float(item[1]["total_net_pnl"]),
            -float(item[1]["win_rate"]),
            item[0],
        ),
    )

    ranking = []
    for idx, (strategy_name, metric) in enumerate(ranked, start=1):
        ranking.append(
            {
                "rank": idx,
                "strategy": strategy_name,
                "total_net_pnl": metric["total_net_pnl"],
                "win_rate": metric["win_rate"],
                "trades": metric["trades"],
            }
        )

    return {
        "baseline_source": "src.core.testing.synthetic_data.make_synthetic_ohlcv_v1(rows=430, freq='D')",
        "metric_definitions": {
            DRAWDOWN_PROXY_FIELD: DRAWDOWN_PROXY_DEFINITION,
        },
        "strategies": strategies,
        "ranking": ranking,
    }


def write_comparison_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_comparison_json(json_path: Path) -> dict[str, Any] | None:
    if not json_path.exists():
        return None
    return json.loads(json_path.read_text(encoding="utf-8"))


def render_comparison_report(payload: dict[str, Any], json_path: Path) -> str:
    repo_root = Path(__file__).resolve().parents[3]
    try:
        display_json_path = json_path.relative_to(repo_root)
    except ValueError:
        display_json_path = json_path

    lines = [
        "Strategy Comparison Report",
        "",
        f"Source JSON: {display_json_path}",
        f"Deterministic baseline: {payload.get('baseline_source', 'n/a')}",
        "",
        "Metric definitions",
        f"- {DRAWDOWN_PROXY_FIELD}: {payload.get('metric_definitions', {}).get(DRAWDOWN_PROXY_FIELD, 'n/a')}",
        "",
        "Per-strategy summary",
    ]

    strategies = payload.get("strategies", {})
    for strategy_name in sorted(strategies):
        metric = strategies[strategy_name]
        lines.append(
            "- "
            f"{strategy_name}: trades={metric.get('trades', 0)}, "
            f"total_net_pnl={float(metric.get('total_net_pnl', 0.0)):.6f}, "
            f"win_rate={float(metric.get('win_rate', 0.0)):.6f}%, "
            f"avg_pnl={float(metric.get('avg_pnl', 0.0)):.6f}, "
            f"avg_bars_in_trade={float(metric.get('avg_bars_in_trade', 0.0)):.6f}, "
            f"{DRAWDOWN_PROXY_FIELD}={float(metric.get(DRAWDOWN_PROXY_FIELD, 0.0)):.6f}"
        )

    lines.append("")
    lines.append("Ranking (by total_net_pnl desc, win_rate desc, strategy name)")
    for row in payload.get("ranking", []):
        lines.append(
            "- "
            f"#{int(row.get('rank', 0))} {row.get('strategy', 'n/a')}: "
            f"total_net_pnl={float(row.get('total_net_pnl', 0.0)):.6f}, "
            f"win_rate={float(row.get('win_rate', 0.0)):.6f}%, "
            f"trades={int(row.get('trades', 0))}"
        )

    return "\n".join(lines) + "\n"


def write_comparison_report_from_json(json_path: Path, report_path: Path) -> None:
    payload = load_comparison_json(json_path)
    if payload is None:
        raise FileNotFoundError(f"Comparison JSON not found: {json_path}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_comparison_report(payload, json_path), encoding="utf-8")


def generate_metrics_and_report(
    temp_runs_dir: Path,
    csv_dir: Path,
    metrics_json_path: Path,
    metrics_report_path: Path,
) -> tuple[dict[str, Path | None], dict[str, Any]]:
    csv_by_strategy = collect_trade_csvs(temp_runs_dir=temp_runs_dir, csv_dir=csv_dir)
    payload = build_comparison_payload(csv_by_strategy)
    write_comparison_json(payload, metrics_json_path)
    write_comparison_report_from_json(metrics_json_path, metrics_report_path)
    return csv_by_strategy, payload
