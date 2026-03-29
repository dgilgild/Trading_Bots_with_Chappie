from __future__ import annotations

import math
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.core.data import fetch_ohlcv
from src.core.testing.synthetic_data import make_synthetic_ohlcv_v1
import src.strategies.basic_keltner_reversion.backtest_basic_keltner_reversion_v2 as basic_keltner_runner
import src.strategies.bmsb.backtest_bmsb_v2 as bmsb_runner
import src.strategies.donchian_breakout.backtest_donchian_breakout_v2 as donchian_runner
import src.strategies.ema_cross.backtest_ema_cross_v2 as ema_runner
import src.strategies.ema_trend_hold.backtest_ema_trend_hold_v2 as ema_trend_hold_runner
import src.strategies.emalyarovich_smas.backtest_emalyarovich_smas_v2 as emalyarovich_runner
import src.strategies.k_davey_mom_keltner.backtest_k_davey_mom_keltner_v2 as k_davey_runner
import src.strategies.rsi_reversion.backtest_rsi_reversion_v2 as rsi_runner


SUPPORTED_WALK_FORWARD_STRATEGIES = [
    "ema_cross",
    "rsi_reversion",
    "donchian_breakout",
    "ema_trend_hold",
    "bmsb",
    "emalyarovich_smas",
    "k_davey_mom_keltner",
    "basic_keltner_reversion",
]

WINDOW_OUTPUT_COLUMNS = [
    "window_id",
    "train_start_idx",
    "train_end_idx",
    "test_start_idx",
    "test_end_idx",
    "train_start_ts",
    "train_end_ts",
    "test_start_ts",
    "test_end_ts",
    "train_bars",
    "test_bars",
    "total_trades",
    "total_net_profit",
    "profit_factor",
    "max_drawdown_pct",
]

STRATEGY_RUNNERS: dict[str, dict[str, Any]] = {
    "ema_cross": {
        "module": ema_runner,
        "func": ema_runner.run_backtest_ema_cross_v2,
        "params": {"ema_fast": 20, "ema_slow": 50},
    },
    "rsi_reversion": {
        "module": rsi_runner,
        "func": rsi_runner.run_backtest_rsi_reversion_v2,
        "params": {"rsi_period": 14, "rsi_entry": 30, "rsi_exit": 50},
    },
    "donchian_breakout": {
        "module": donchian_runner,
        "func": donchian_runner.run_backtest_donchian_breakout_v2,
        "params": {"donchian_lookback": 20},
    },
    "ema_trend_hold": {
        "module": ema_trend_hold_runner,
        "func": ema_trend_hold_runner.run_backtest_ema_trend_hold_v2,
        "params": {"trend_ema": 200},
    },
    "bmsb": {
        "module": bmsb_runner,
        "func": bmsb_runner.run_backtest_bmsb_v2,
        "params": {"sma_period": 20, "ema_period": 21, "tensignal_window": 3},
    },
    "emalyarovich_smas": {
        "module": emalyarovich_runner,
        "func": emalyarovich_runner.run_backtest_emalyarovich_smas_v2,
        "params": {"sma_fast": 20, "sma_slow": 200, "slope_bars": 3},
    },
    "k_davey_mom_keltner": {
        "module": k_davey_runner,
        "func": k_davey_runner.run_backtest_k_davey_mom_keltner_v2,
        "params": {"position_mode": "fixed", "trade_size": 1.0},
    },
    "basic_keltner_reversion": {
        "module": basic_keltner_runner,
        "func": basic_keltner_runner.run_backtest_basic_keltner_reversion_v2,
        "params": {"kc_ema_length": 20, "kc_atr_length": 20, "kc_atr_mult": 1.5},
    },
}


@dataclass(frozen=True)
class WalkForwardWindow:
    window_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int


def _validate_positive(value: float, name: str) -> None:
    if value is None or float(value) <= 0:
        raise ValueError(f"{name} must be > 0")


def _pct_to_len(total_bars: int, pct: float, name: str) -> int:
    _validate_positive(pct, name)
    length = int(total_bars * (float(pct) / 100.0))
    if length < 1:
        raise ValueError(f"Computed {name} length must be >= 1 bar")
    return length


def resolve_window_lengths(
    total_bars: int,
    *,
    train_pct: float | None = None,
    test_pct: float | None = None,
    step_pct: float | None = None,
    train_bars: int | None = None,
    test_bars: int | None = None,
    step_bars: int | None = None,
) -> tuple[str, int, int, int]:
    if total_bars < 1:
        raise ValueError("total_bars must be >= 1")

    fixed_mode = any(v is not None for v in (train_bars, test_bars, step_bars))
    pct_mode = any(v is not None for v in (train_pct, test_pct, step_pct))

    if fixed_mode and pct_mode:
        raise ValueError("Provide either percentage inputs or fixed-bar inputs, not both")

    if fixed_mode:
        if train_bars is None or test_bars is None or step_bars is None:
            raise ValueError("train_bars, test_bars, and step_bars are all required in fixed-bars mode")
        train_len = int(train_bars)
        test_len = int(test_bars)
        step_len = int(step_bars)

        _validate_positive(train_len, "train_bars")
        _validate_positive(test_len, "test_bars")
        _validate_positive(step_len, "step_bars")
        mode = "fixed"
    else:
        if train_pct is None or test_pct is None or step_pct is None:
            raise ValueError("train_pct, test_pct, and step_pct are required in percentage mode")

        _validate_positive(train_pct, "train_pct")
        _validate_positive(test_pct, "test_pct")
        _validate_positive(step_pct, "step_pct")

        if float(train_pct) + float(test_pct) > 100.0:
            raise ValueError("train_pct + test_pct must be <= 100")

        train_len = _pct_to_len(total_bars, float(train_pct), "train_pct")
        test_len = _pct_to_len(total_bars, float(test_pct), "test_pct")
        step_len = _pct_to_len(total_bars, float(step_pct), "step_pct")
        mode = "percentage"

    if step_len < test_len:
        raise ValueError("No-overlap policy violated: step_len must be >= test_len")

    return mode, train_len, test_len, step_len


def plan_walk_forward_windows(
    total_bars: int,
    train_len: int,
    test_len: int,
    step_len: int,
) -> list[WalkForwardWindow]:
    if train_len < 1 or test_len < 1 or step_len < 1:
        raise ValueError("train_len, test_len, and step_len must be >= 1")
    if step_len < test_len:
        raise ValueError("No-overlap policy violated: step_len must be >= test_len")
    if train_len + test_len > total_bars:
        raise ValueError("train_len + test_len must be <= total_bars")

    windows: list[WalkForwardWindow] = []
    train_start = 0
    window_id = 1

    while True:
        train_end = train_start + train_len
        test_start = train_end
        test_end = test_start + test_len

        if test_end > total_bars:
            break

        windows.append(
            WalkForwardWindow(
                window_id=window_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )

        window_id += 1
        train_start += step_len

    if not windows:
        raise ValueError("No walk-forward windows produced with current parameters")

    return windows


def load_walk_forward_data(
    mode: str,
    *,
    exchange: str = "binance",
    symbol: str = "BTC/USDT",
    timeframe: str = "1d",
    start_date: str | None = None,
    end_date: str | None = None,
    use_clean: bool = True,
    deterministic_rows: int = 420,
) -> pd.DataFrame:
    mode_norm = str(mode).strip().lower()

    if mode_norm == "deterministic":
        df = make_synthetic_ohlcv_v1(rows=int(deterministic_rows), freq="D")
    elif mode_norm == "db":
        df = fetch_ohlcv(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            limit=50000,
            use_clean=use_clean,
        )
    else:
        raise ValueError("mode must be one of: deterministic, db")

    if df is None or df.empty:
        raise ValueError("No OHLCV data available for walk-forward run")

    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    out = df[required].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=False)
    return out


@contextmanager
def _patched_attr(module: object, attr_name: str, replacement: Any):
    original = getattr(module, attr_name)
    setattr(module, attr_name, replacement)
    try:
        yield
    finally:
        setattr(module, attr_name, original)


def _extract_window_metrics(stats: dict[str, Any]) -> dict[str, Any]:
    stats = stats or {}
    profit_factor_raw = stats.get("Profit Factor")
    profit_factor = None

    if profit_factor_raw is not None:
        candidate = float(profit_factor_raw)
        if math.isfinite(candidate):
            profit_factor = candidate

    return {
        "total_trades": int(stats.get("Total trades", 0) or 0),
        "total_net_profit": float(stats.get("Total Net Profit", 0.0) or 0.0),
        "profit_factor": profit_factor,
        "max_drawdown_pct": float(stats.get("Max Drawdown (%)", 0.0) or 0.0),
    }


def _build_summary(
    windows_df: pd.DataFrame,
    *,
    strategy: str,
    data_mode: str,
    total_bars: int,
    window_mode: str,
    train_len: int,
    test_len: int,
    step_len: int,
) -> dict[str, Any]:
    summary = {
        "strategy": strategy,
        "data_mode": data_mode,
        "total_bars": int(total_bars),
        "window_mode": window_mode,
        "train_len": int(train_len),
        "test_len": int(test_len),
        "step_len": int(step_len),
        "window_count": int(len(windows_df)),
        "windows_with_trades": 0,
        "total_trades": 0,
        "aggregate_net_profit": 0.0,
        "mean_window_net_profit": 0.0,
        "mean_window_profit_factor": None,
    }

    if windows_df.empty:
        return summary

    summary["windows_with_trades"] = int((windows_df["total_trades"] > 0).sum())
    summary["total_trades"] = int(windows_df["total_trades"].sum())
    summary["aggregate_net_profit"] = float(windows_df["total_net_profit"].sum())
    summary["mean_window_net_profit"] = float(windows_df["total_net_profit"].mean())

    pf_series = windows_df["profit_factor"].dropna()
    if not pf_series.empty:
        summary["mean_window_profit_factor"] = float(pf_series.mean())

    return summary


def run_walk_forward_strategy(
    data_df: pd.DataFrame,
    windows: list[WalkForwardWindow],
    *,
    strategy: str,
    exchange: str,
    symbol: str,
    timeframe: str,
    ema_fast: int = 20,
    ema_slow: int = 50,
    initial_balance: float = 1000.0,
) -> pd.DataFrame:
    strategy_spec = STRATEGY_RUNNERS.get(strategy)
    if strategy_spec is None:
        raise ValueError(f"Unsupported strategy='{strategy}'")

    runner_module = strategy_spec["module"]
    runner_func = strategy_spec["func"]

    rows: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="walk_forward_") as temp_dir:
        for window in windows:
            window_df = data_df.iloc[window.test_start:window.test_end].copy()
            start_date = pd.Timestamp(window_df["timestamp"].iloc[0]).strftime("%Y-%m-%d")
            end_date = pd.Timestamp(window_df["timestamp"].iloc[-1]).strftime("%Y-%m-%d")

            def fake_fetch(**_kwargs):
                return window_df.copy()

            strategy_params = dict(strategy_spec["params"])
            if strategy == "ema_cross":
                strategy_params["ema_fast"] = int(ema_fast)
                strategy_params["ema_slow"] = int(ema_slow)

            with _patched_attr(runner_module, "fetch_ohlcv", fake_fetch):
                stats, _chart_path, _csv_path = runner_func(
                    exchange=exchange,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    use_clean=True,
                    run_id=f"wf_{window.window_id:03d}",
                    initial_balance=float(initial_balance),
                    generate_report=False,
                    generate_plots=False,
                    generate_equity=False,
                    base_path=temp_dir,
                    **strategy_params,
                )

            metrics = _extract_window_metrics(stats)

            rows.append(
                {
                    "window_id": window.window_id,
                    "train_start_idx": window.train_start,
                    "train_end_idx": window.train_end,
                    "test_start_idx": window.test_start,
                    "test_end_idx": window.test_end,
                    "train_start_ts": pd.Timestamp(data_df["timestamp"].iloc[window.train_start]).isoformat(),
                    "train_end_ts": pd.Timestamp(data_df["timestamp"].iloc[window.train_end - 1]).isoformat(),
                    "test_start_ts": pd.Timestamp(data_df["timestamp"].iloc[window.test_start]).isoformat(),
                    "test_end_ts": pd.Timestamp(data_df["timestamp"].iloc[window.test_end - 1]).isoformat(),
                    "train_bars": window.train_end - window.train_start,
                    "test_bars": window.test_end - window.test_start,
                    **metrics,
                }
            )

    if not rows:
        return pd.DataFrame(columns=WINDOW_OUTPUT_COLUMNS)

    return pd.DataFrame(rows).reindex(columns=WINDOW_OUTPUT_COLUMNS)


def run_walk_forward(
    *,
    strategy: str,
    data_mode: str,
    data_df: pd.DataFrame,
    train_len: int,
    test_len: int,
    step_len: int,
    window_mode: str,
    exchange: str = "binance",
    symbol: str = "BTC/USDT",
    timeframe: str = "1d",
    ema_fast: int = 20,
    ema_slow: int = 50,
    initial_balance: float = 1000.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if strategy not in STRATEGY_RUNNERS:
        raise ValueError(
            f"Unsupported strategy='{strategy}'. Supported: {', '.join(SUPPORTED_WALK_FORWARD_STRATEGIES)}"
        )

    windows = plan_walk_forward_windows(
        total_bars=len(data_df),
        train_len=int(train_len),
        test_len=int(test_len),
        step_len=int(step_len),
    )

    windows_df = run_walk_forward_strategy(
        data_df=data_df,
        windows=windows,
        strategy=strategy,
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        initial_balance=initial_balance,
    )

    summary = _build_summary(
        windows_df,
        strategy=strategy,
        data_mode=data_mode,
        total_bars=len(data_df),
        window_mode=window_mode,
        train_len=train_len,
        test_len=test_len,
        step_len=step_len,
    )

    return windows_df, summary
