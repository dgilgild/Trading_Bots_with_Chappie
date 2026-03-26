from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import pytest

import src.strategies.basic_keltner_reversion.backtest_basic_keltner_reversion_v2 as bk_runner
import src.strategies.bmsb.backtest_bmsb_v2 as bmsb_runner
import src.strategies.donchian_breakout.backtest_donchian_breakout_v2 as don_runner
import src.strategies.ema_cross.backtest_ema_cross_v2 as ema_runner
import src.strategies.ema_trend_hold.backtest_ema_trend_hold_v2 as trend_runner
import src.strategies.emalyarovich_smas.backtest_emalyarovich_smas_v2 as sma_runner
import src.strategies.k_davey_mom_keltner.backtest_k_davey_mom_keltner_v2 as kd_runner
import src.strategies.rsi_reversion.backtest_rsi_reversion_v2 as rsi_runner


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


@pytest.fixture(scope="session")
def ohlcv_factory() -> Callable[[int, str], pd.DataFrame]:
    return make_synthetic_ohlcv


@pytest.fixture(scope="session")
def synthetic_ohlcv(ohlcv_factory: Callable[[int, str], pd.DataFrame]) -> pd.DataFrame:
    return ohlcv_factory(rows=430, freq="D")


@pytest.fixture
def common_backtest_kwargs(tmp_path: pytest.TempPathFactory) -> dict:
    return {
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
        "base_path": str(tmp_path),
    }


def build_backtest_jobs(common: dict) -> list[tuple[str, object, object, dict]]:
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


@pytest.fixture
def backtest_jobs(common_backtest_kwargs: dict) -> list[tuple[str, object, object, dict]]:
    return build_backtest_jobs(common_backtest_kwargs)


@pytest.fixture(scope="session")
def regression_expected() -> dict:
    return {
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
