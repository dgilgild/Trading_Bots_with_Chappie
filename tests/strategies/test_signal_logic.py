from __future__ import annotations

import pandas as pd
import pytest

from src.core.backtester_v2 import BacktesterV2
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

pytestmark = pytest.mark.unit


def _run_bmsb_signal_flow(
    df: pd.DataFrame,
    *,
    trail_percent: float = 0.05,
    use_tp_sl: bool = True,
) -> tuple[BacktesterV2, list[float]]:
    bt = BacktesterV2(
        initial_capital=1000.0,
        position_mode="fixed",
        trade_size=100.0,
        commission_pct=0.0,
        slippage_pct=0.0,
        allow_short=False,
        stop_loss_pct=None,
        take_profit_pct=None,
    )
    trailing_stops: list[float] = []

    for i in range(1, len(df)):
        current_bar = df.iloc[i]
        price = float(current_bar["close"])
        high = float(current_bar["high"])
        low = float(current_bar["low"])
        timestamp = current_bar["timestamp"]

        if use_tp_sl and bt.position is not None and bt.position["side"] == "LONG":
            trail_price = price * (1 - trail_percent)
            prev_stop = bt.position.get("stop_price")
            if prev_stop is None:
                bt.position["stop_price"] = trail_price
            else:
                bt.position["stop_price"] = max(trail_price, prev_stop)
            trailing_stops.append(float(bt.position["stop_price"]))

        bt.on_bar(high=high, low=low, timestamp=timestamp, bar_index=i)

        buysignal = price > float(current_bar["bmsb"])
        tensignal = float(current_bar["tensignal"])
        sellsignal = (
            float(df["close"].iloc[i - 1]) >= float(df["bmsb"].iloc[i - 1])
            and price < float(current_bar["bmsb"])
        )

        if buysignal and tensignal >= 1:
            bt.on_signal("LONG", price, timestamp, "BMSB long", i)

        if bt.position is not None and bt.position["side"] == "LONG" and sellsignal:
            bt.on_signal("EXIT", price, timestamp, "BMSB crossunder", i)

    return bt, trailing_stops


def test_ema_cross_signal_logic() -> None:
    bullish = pd.DataFrame({"close": [10, 9, 8, 7, 6, 7, 8]})
    sig, _ = ema_check(bullish, fast=2, slow=4, trend_period=3, current_side=None)
    assert sig == "LONG", "EMA cross should produce LONG on bullish cross + trend filter"

    bearish = pd.DataFrame({"close": [8, 9, 10, 11, 12, 11, 10]})
    sig, _ = ema_check(bearish, fast=2, slow=4, trend_period=3, current_side=None)
    assert sig == "SHORT", "EMA cross should produce SHORT on bearish cross + trend filter"

    short_exit = pd.DataFrame({"close": [18, 5, 13, 18, 10, 8, 13, 12]})
    sig, _ = ema_check(short_exit, fast=2, slow=4, trend_period=6, current_side="SHORT")
    assert sig == "EXIT", "EMA cross should produce EXIT for short on opposite cross"


def test_rsi_reversion_signal_logic() -> None:
    sig, _ = rsi_check(25, entry_level=30, exit_level=50, current_side=None)
    assert sig == "LONG", "RSI strategy should LONG below entry threshold"

    sig, _ = rsi_check(55, entry_level=30, exit_level=50, current_side="LONG")
    assert sig == "EXIT", "RSI strategy should EXIT long above exit threshold"


def test_donchian_breakout_signal_logic() -> None:
    base = pd.DataFrame(
        {
            "high": [10, 11, 12, 13, 14, 15],
            "low": [5, 6, 7, 8, 9, 10],
            "close": [7, 8, 9, 10, 11, 16],
        }
    )
    don = compute_donchian(base, lookback=3)
    sig, _ = don_check(don, lookback=3, current_side=None)
    assert sig == "LONG", "Donchian should LONG on breakout above rolling high"


def test_ema_trend_hold_signal_logic() -> None:
    sig, _ = trend_check(price=105, trend_value=100, trend_period=200, current_side=None)
    assert sig == "LONG", "EMA trend hold should LONG when price > trend EMA"

    sig, _ = trend_check(price=99, trend_value=100, trend_period=200, current_side="LONG")
    assert sig == "EXIT", "EMA trend hold should EXIT long when price < trend EMA"


def test_bmsb_indicator_logic() -> None:
    bdf = pd.DataFrame({"close": [10, 11, 12, 13, 14, 15, 16]})
    bdf["high"] = bdf["close"] + 0.5
    bdf["low"] = bdf["close"] - 0.5

    bdf = compute_bmsb(bdf, sma_period=3, ema_period=3)
    ten = compute_tensignal(bdf, window=3)

    assert "bmsb" in bdf.columns, "BMSB column should exist"
    assert len(ten) == len(bdf), "tensignal length should match input"


def test_bmsb_entry_requires_buysignal_and_tensignal_gate() -> None:
    bmsb_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-01-01", periods=4, freq="D", tz="UTC"),
            "close": [100.0, 102.0, 103.0, 101.0],
            "high": [100.5, 102.5, 103.5, 101.5],
            "low": [99.5, 101.5, 102.5, 100.5],
            "bmsb": [101.0, 100.0, 101.0, 100.5],
            "tensignal": [1.0, 0.0, 1.0, 1.0],
        }
    )

    bt, _ = _run_bmsb_signal_flow(bmsb_df, use_tp_sl=False)

    assert bt.position is not None, "BMSB should open a position once the gate is satisfied"
    assert bt.position["side"] == "LONG", "BMSB entry should open a LONG position"
    assert len(bt.lots) == 1, "BMSB should not open on buysignal-only bars with tensignal < 1"
    assert bt.lots[0]["entry_time"] == bmsb_df.iloc[2]["timestamp"], (
        "BMSB should enter on the first bar with buysignal and tensignal >= 1"
    )


def test_bmsb_exit_uses_crossunder_logic() -> None:
    bmsb_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-02-01", periods=4, freq="D", tz="UTC"),
            "close": [99.0, 101.0, 98.0, 98.0],
            "high": [99.5, 101.5, 98.5, 98.5],
            "low": [98.5, 100.5, 97.5, 97.5],
            "bmsb": [100.0, 100.0, 100.0, 100.0],
            "tensignal": [1.0, 1.0, 1.0, 1.0],
        }
    )

    bt, _ = _run_bmsb_signal_flow(bmsb_df, use_tp_sl=False)

    assert bt.position is None, "BMSB crossunder should close the open long"
    assert len(bt.trades) == 1, "BMSB crossunder path should produce one closed trade"
    assert bt.trades[0]["entry_trigger"] == "BMSB long"
    assert bt.trades[0]["exit_trigger"] == "BMSB crossunder"


def test_bmsb_trailing_stop_updates_and_exits_through_backtester_flow() -> None:
    bmsb_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-03-01", periods=5, freq="D", tz="UTC"),
            "close": [99.0, 101.0, 110.0, 120.0, 113.0],
            "high": [99.5, 101.5, 110.5, 120.5, 114.0],
            "low": [98.5, 100.5, 109.5, 119.5, 113.0],
            "bmsb": [100.0, 100.0, 105.0, 110.0, 112.0],
            "tensignal": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    bt, trailing_stops = _run_bmsb_signal_flow(bmsb_df, trail_percent=0.05, use_tp_sl=True)

    assert trailing_stops, "Trailing stop path should update at least once for open long"
    assert trailing_stops == sorted(trailing_stops), "Trailing stop should not move downward"
    assert trailing_stops[-1] == pytest.approx(114.0), (
        "Trailing stop should retain the highest computed level before exit"
    )
    assert len(bt.trades) == 1, "Trailing-stop scenario should close exactly one trade"
    assert bt.trades[0]["exit_trigger"] == "stop_loss", (
        "BacktesterV2 on_bar flow should close at trailing stop as stop_loss"
    )


def test_emalyarovich_smas_signal_logic() -> None:
    sdf = pd.DataFrame(
        {
            "close": [100, 101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103, 99],
            "sma_fast": [95, 96, 97, 98, 99, 100],
            "sma_slow": [90, 91, 92, 93, 94, 95],
        }
    )
    sig, _ = smas_check(sdf, sma_fast=3, sma_slow=3, slope_bars=3, current_side=None)
    assert sig == "LONG", "E.Malyarovich SMAs should LONG on pullback + positive slope"


def test_k_davey_mom_keltner_indicator_and_size_logic(ohlcv_factory) -> None:
    kdf = ohlcv_factory(rows=160)
    kstoch = compute_keltner_stochastic(kdf, length=5, atr_mult=0.5)

    assert len(kstoch) == len(kdf), "Keltner stochastic series length should match input"
    assert not kstoch.dropna().empty, "Keltner stochastic should produce values after warmup"

    size = compute_position_size(
        net_equity=16000,
        base_equity=15000,
        sizing_factor=0.33,
        max_contracts=15,
        use_position_sizing=True,
    )
    assert 1 <= size <= 15, "K. Davey dynamic contracts should be capped to valid bounds"


def test_basic_keltner_reversion_signal_set(ohlcv_factory) -> None:
    kc_df = ohlcv_factory(rows=80)
    sig, _ = keltner_reversion(kc_df, ema_length=20, atr_length=20, atr_mult=1.5)
    assert sig in {"LONG", "SHORT", "EXIT", None}, "Basic KC should return known signal set"
