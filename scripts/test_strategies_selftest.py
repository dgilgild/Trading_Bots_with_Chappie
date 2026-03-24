"""Self-test suite for strategy logic and backtest smoke checks.

Run:
    PYTHONPATH=. python3 scripts/test_strategies_selftest.py
"""

from __future__ import annotations

import math
import os
import tempfile
import traceback
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import pandas as pd

import src.strategies.basic_keltner_reversion.backtest_basic_keltner_reversion_v2 as bk_runner
import src.strategies.bmsb.backtest_bmsb_v2 as bmsb_runner
import src.strategies.donchian_breakout.backtest_donchian_breakout_v2 as don_runner
import src.strategies.ema_cross.backtest_ema_cross_v2 as ema_runner
import src.strategies.ema_trend_hold.backtest_ema_trend_hold_v2 as trend_runner
import src.strategies.emalyarovich_smas.backtest_emalyarovich_smas_v2 as sma_runner
import src.strategies.k_davey_mom_keltner.backtest_k_davey_mom_keltner_v2 as kd_runner
import src.strategies.rsi_reversion.backtest_rsi_reversion_v2 as rsi_runner
from src.strategies.basic_keltner_reversion.strategy import keltner_reversion
from src.strategies.bmsb.strategy import compute_bmsb, compute_tensignal
from src.strategies.donchian_breakout.strategy import check_signal as don_check
from src.strategies.donchian_breakout.strategy import compute_donchian
from src.strategies.ema_cross.strategy import check_signal as ema_check
from src.strategies.ema_trend_hold.strategy import check_signal as trend_check
from src.strategies.emalyarovich_smas.strategy import check_signal as smas_check
from src.strategies.k_davey_mom_keltner.strategy import (
    compute_keltner_stochastic,
    compute_position_size,
)
from src.strategies.rsi_reversion.strategy import check_signal as rsi_check


@dataclass
class TestResult:
    name: str
    ok: bool
    details: str = ""


def make_synthetic_ohlcv(rows: int = 420, freq: str = "D") -> pd.DataFrame:
    idx = pd.date_range("2021-01-01", periods=rows, freq=freq)
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


@contextmanager
def patched_attr(module, attr_name, replacement):
    original = getattr(module, attr_name)
    setattr(module, attr_name, replacement)
    try:
        yield
    finally:
        setattr(module, attr_name, original)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _build_backtest_jobs(common: dict) -> list[tuple[str, object, object, dict]]:
    return [
        (
            "ema_cross",
            ema_runner,
            ema_runner.run_backtest_ema_cross_v2,
            {**common, "ema_fast": 20, "ema_slow": 50},
        ),
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
            {
                **common,
                "symbol": "MES",
                "position_mode": "fixed",
                "trade_size": 1.0,
            },
        ),
        (
            "basic_keltner_reversion",
            bk_runner,
            bk_runner.run_backtest_basic_keltner_reversion_v2,
            {**common, "kc_ema_length": 20, "kc_atr_length": 20, "kc_atr_mult": 1.5},
        ),
    ]


def test_signal_level_logic() -> list[TestResult]:
    results: list[TestResult] = []

    try:
        # EMA cross long
        df = pd.DataFrame({"close": [10, 9, 8, 7, 6, 7, 8]})
        sig, _ = ema_check(df, fast=2, slow=4, trend_period=3, current_side=None)
        _assert(sig == "LONG", "EMA cross should produce LONG on bullish cross + trend filter")

        # EMA cross short
        df = pd.DataFrame({"close": [8, 9, 10, 11, 12, 11, 10]})
        sig, _ = ema_check(df, fast=2, slow=4, trend_period=3, current_side=None)
        _assert(sig == "SHORT", "EMA cross should produce SHORT on bearish cross + trend filter")

        # EMA cross exit short path
        df = pd.DataFrame({"close": [18, 5, 13, 18, 10, 8, 13, 12]})
        sig, _ = ema_check(df, fast=2, slow=4, trend_period=6, current_side="SHORT")
        _assert(sig == "EXIT", "EMA cross should produce EXIT for short on opposite cross")

        results.append(TestResult("ema_cross.signal_logic", True))
    except Exception as exc:
        results.append(TestResult("ema_cross.signal_logic", False, str(exc)))

    try:
        sig, _ = rsi_check(25, entry_level=30, exit_level=50, current_side=None)
        _assert(sig == "LONG", "RSI strategy should LONG below entry threshold")
        sig, _ = rsi_check(55, entry_level=30, exit_level=50, current_side="LONG")
        _assert(sig == "EXIT", "RSI strategy should EXIT long above exit threshold")
        results.append(TestResult("rsi_reversion.signal_logic", True))
    except Exception as exc:
        results.append(TestResult("rsi_reversion.signal_logic", False, str(exc)))

    try:
        base = pd.DataFrame(
            {
                "high": [10, 11, 12, 13, 14, 15],
                "low": [5, 6, 7, 8, 9, 10],
                "close": [7, 8, 9, 10, 11, 16],
            }
        )
        don = compute_donchian(base, lookback=3)
        sig, _ = don_check(don, lookback=3, current_side=None)
        _assert(sig == "LONG", "Donchian should LONG on breakout above rolling high")
        results.append(TestResult("donchian_breakout.signal_logic", True))
    except Exception as exc:
        results.append(TestResult("donchian_breakout.signal_logic", False, str(exc)))

    try:
        sig, _ = trend_check(price=105, trend_value=100, trend_period=200, current_side=None)
        _assert(sig == "LONG", "EMA trend hold should LONG when price > trend EMA")
        sig, _ = trend_check(price=99, trend_value=100, trend_period=200, current_side="LONG")
        _assert(sig == "EXIT", "EMA trend hold should EXIT long when price < trend EMA")
        results.append(TestResult("ema_trend_hold.signal_logic", True))
    except Exception as exc:
        results.append(TestResult("ema_trend_hold.signal_logic", False, str(exc)))

    try:
        bdf = pd.DataFrame({"close": [10, 11, 12, 13, 14, 15, 16]})
        bdf["high"] = bdf["close"] + 0.5
        bdf["low"] = bdf["close"] - 0.5
        bdf = compute_bmsb(bdf, sma_period=3, ema_period=3)
        ten = compute_tensignal(bdf, window=3)
        _assert("bmsb" in bdf.columns, "BMSB column should exist")
        _assert(len(ten) == len(bdf), "tensignal length should match input")
        results.append(TestResult("bmsb.indicator_logic", True))
    except Exception as exc:
        results.append(TestResult("bmsb.indicator_logic", False, str(exc)))

    try:
        sdf = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104, 105],
                "low": [99, 100, 101, 102, 103, 99],
                "sma_fast": [95, 96, 97, 98, 99, 100],
                "sma_slow": [90, 91, 92, 93, 94, 95],
            }
        )
        sig, _ = smas_check(sdf, sma_fast=3, sma_slow=3, slope_bars=3, current_side=None)
        _assert(sig == "LONG", "E.Malyarovich SMAs should LONG on pullback + positive slope")
        results.append(TestResult("emalyarovich_smas.signal_logic", True))
    except Exception as exc:
        results.append(TestResult("emalyarovich_smas.signal_logic", False, str(exc)))

    try:
        kdf = make_synthetic_ohlcv(rows=160)
        kstoch = compute_keltner_stochastic(kdf, length=5, atr_mult=0.5)
        _assert(len(kstoch) == len(kdf), "Keltner stochastic series length should match input")
        _assert(not kstoch.dropna().empty, "Keltner stochastic should produce non-NaN values after warmup")
        size = compute_position_size(
            net_equity=16000,
            base_equity=15000,
            sizing_factor=0.33,
            max_contracts=15,
            use_position_sizing=True,
        )
        _assert(1 <= size <= 15, "K. Davey dynamic contracts should be capped to valid bounds")
        results.append(TestResult("k_davey_mom_keltner.indicator_and_size_logic", True))
    except Exception as exc:
        results.append(TestResult("k_davey_mom_keltner.indicator_and_size_logic", False, str(exc)))

    try:
        kc_df = make_synthetic_ohlcv(rows=80)
        sig, _ = keltner_reversion(kc_df, ema_length=20, atr_length=20, atr_mult=1.5)
        _assert(sig in {"LONG", "SHORT", "EXIT", None}, "Basic KC should return known signal set")
        results.append(TestResult("basic_keltner_reversion.signal_logic", True))
    except Exception as exc:
        results.append(TestResult("basic_keltner_reversion.signal_logic", False, str(exc)))

    return results


def test_backtest_smoke() -> list[TestResult]:
    results: list[TestResult] = []
    df = make_synthetic_ohlcv(rows=430, freq="D")

    def fake_fetch(**_kwargs):
        return df.copy()

    with tempfile.TemporaryDirectory() as tmp:
        common = {
            "exchange": "binance",
            "symbol": "BTC/USDT",
            "timeframe": "1d",
            "start_date": "2021-01-01",
            "end_date": "2022-12-31",
            "use_clean": True,
            "run_id": "selftest",
            "generate_report": False,
            "generate_plots": False,
            "generate_equity": False,
            "base_path": tmp,
        }

        jobs = _build_backtest_jobs(common)

        for strategy_name, module, fn, kwargs in jobs:
            try:
                with patched_attr(module, "fetch_ohlcv", fake_fetch):
                    stats, chart_path, csv_path = fn(**kwargs)
                _assert(isinstance(stats, dict), "Backtest should return stats dict")
                _assert(chart_path is None, "Chart path expected None when generate_equity=False")
                _assert(
                    csv_path is None or isinstance(csv_path, str),
                    "CSV path should be None or string",
                )
                results.append(TestResult(f"{strategy_name}.backtest_smoke", True))
            except Exception as exc:
                results.append(TestResult(f"{strategy_name}.backtest_smoke", False, str(exc)))

    return results


def test_backtest_regression() -> list[TestResult]:
    """Golden regression checks over deterministic synthetic data.

    These tests intentionally lock a baseline for trade counts, net PnL,
    and representative trigger strings so behavior drift is caught early.
    """

    results: list[TestResult] = []
    df = make_synthetic_ohlcv(rows=430, freq="D")

    def fake_fetch(**_kwargs):
        return df.copy()

    expected = {
        "ema_cross": {
            "total_trades": 2,
            "total_net_profit": -62.531054,
            "first_entry_trigger_contains": "EMA20 crossed ABOVE EMA50",
            "last_exit_trigger_contains": "stop_loss",
        },
        "rsi_reversion": {
            "total_trades": 6,
            "total_net_profit": 57.056364,
            "first_entry_trigger_contains": "RSI",
            "last_exit_trigger_contains": "above 50",
        },
        "donchian_breakout": {
            "total_trades": 0,
            "total_net_profit": None,
        },
        "ema_trend_hold": {
            "total_trades": 2,
            "total_net_profit": -45.390923,
            "first_entry_trigger_contains": "Price above EMA200",
            "last_exit_trigger_contains": "stop_loss",
        },
        "bmsb": {
            "total_trades": 10,
            "total_net_profit": 242.295125,
            "first_entry_trigger_contains": "BMSB long",
            "last_exit_trigger_contains": "BMSB crossunder",
        },
        "emalyarovich_smas": {
            "total_trades": 6,
            "total_net_profit": -42.313773,
            "first_entry_trigger_contains": "SMA20 touch",
            "last_exit_trigger_contains": "Close below SMA20",
        },
        "k_davey_mom_keltner": {
            "total_trades": 6,
            "total_net_profit": -66.654390,
            "first_entry_trigger_contains": "Momentum+Keltner long",
            "last_exit_trigger_contains": "Keltner stoch exit",
        },
        "basic_keltner_reversion": {
            "total_trades": 24,
            "total_net_profit": -308.453923,
            "first_entry_trigger_contains": "KC_LONG",
            "last_exit_trigger_contains": "KC_EXIT_LONG",
        },
    }

    with tempfile.TemporaryDirectory() as tmp:
        common = {
            "exchange": "binance",
            "symbol": "BTC/USDT",
            "timeframe": "1d",
            "start_date": "2021-01-01",
            "end_date": "2022-12-31",
            "use_clean": True,
            "run_id": "selftest",
            "generate_report": False,
            "generate_plots": False,
            "generate_equity": False,
            "base_path": tmp,
        }

        jobs = _build_backtest_jobs(common)

        for strategy_name, module, fn, kwargs in jobs:
            try:
                with patched_attr(module, "fetch_ohlcv", fake_fetch):
                    stats, _chart_path, csv_rel_path = fn(**kwargs)

                spec = expected[strategy_name]
                expected_trades = spec["total_trades"]

                if expected_trades == 0:
                    _assert(stats == {}, f"{strategy_name}: expected empty stats when no trades")
                    _assert(csv_rel_path is None, f"{strategy_name}: expected no CSV when no trades")
                    results.append(TestResult(f"{strategy_name}.backtest_regression", True))
                    continue

                _assert("Total trades" in stats, f"{strategy_name}: missing Total trades stat")
                _assert("Total Net Profit" in stats, f"{strategy_name}: missing Total Net Profit stat")
                _assert(
                    int(stats["Total trades"]) == int(expected_trades),
                    f"{strategy_name}: expected {expected_trades} trades, got {stats['Total trades']}",
                )

                expected_net = float(spec["total_net_profit"])
                got_net = float(stats["Total Net Profit"])
                _assert(
                    math.isclose(got_net, expected_net, rel_tol=0.0, abs_tol=1e-3),
                    f"{strategy_name}: expected net {expected_net}, got {got_net}",
                )

                _assert(csv_rel_path is not None, f"{strategy_name}: expected CSV path")
                csv_abs_path = os.path.join(tmp, "static", csv_rel_path)
                _assert(os.path.exists(csv_abs_path), f"{strategy_name}: CSV file missing on disk")

                trades_df = pd.read_csv(csv_abs_path)
                _assert(
                    len(trades_df) == int(expected_trades),
                    f"{strategy_name}: CSV rows mismatch expected trades",
                )

                first_trigger = str(trades_df.iloc[0]["entry_trigger"])
                last_exit = str(trades_df.iloc[-1]["exit_trigger"])
                _assert(
                    spec["first_entry_trigger_contains"] in first_trigger,
                    (
                        f"{strategy_name}: first entry_trigger mismatch; "
                        f"expected contains '{spec['first_entry_trigger_contains']}', got '{first_trigger}'"
                    ),
                )
                _assert(
                    spec["last_exit_trigger_contains"] in last_exit,
                    (
                        f"{strategy_name}: last exit_trigger mismatch; "
                        f"expected contains '{spec['last_exit_trigger_contains']}', got '{last_exit}'"
                    ),
                )

                results.append(TestResult(f"{strategy_name}.backtest_regression", True))
            except Exception as exc:
                results.append(TestResult(f"{strategy_name}.backtest_regression", False, str(exc)))

    return results


def main() -> int:
    all_results: list[TestResult] = []

    try:
        all_results.extend(test_signal_level_logic())
        all_results.extend(test_backtest_smoke())
        all_results.extend(test_backtest_regression())
    except Exception:
        print("FATAL: unexpected test harness failure")
        print(traceback.format_exc())
        return 2

    passed = sum(1 for r in all_results if r.ok)
    failed = len(all_results) - passed

    print("Strategy self-test results")
    print("-" * 60)
    for r in all_results:
        status = "PASS" if r.ok else "FAIL"
        line = f"[{status}] {r.name}"
        if r.details:
            line += f" -> {r.details}"
        print(line)

    print("-" * 60)
    print(f"Total: {len(all_results)} | Passed: {passed} | Failed: {failed}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
