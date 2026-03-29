from __future__ import annotations

import json
import math
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import pandas as pd

from src.core.analysis.walk_forward import SUPPORTED_WALK_FORWARD_STRATEGIES
from src.core.testing.synthetic_data import make_synthetic_ohlcv_v1
from src.core.ta import compute_rsi
from src.strategies.bmsb.strategy import compute_bmsb, compute_tensignal
import src.strategies.basic_keltner_reversion.backtest_basic_keltner_reversion_v2 as bk_runner
import src.strategies.bmsb.backtest_bmsb_v2 as bmsb_runner
import src.strategies.donchian_breakout.backtest_donchian_breakout_v2 as don_runner
from src.strategies.donchian_breakout.strategy import compute_donchian
import src.strategies.ema_cross.backtest_ema_cross_v2 as ema_runner
import src.strategies.ema_trend_hold.backtest_ema_trend_hold_v2 as trend_runner
import src.strategies.emalyarovich_smas.backtest_emalyarovich_smas_v2 as sma_runner
import src.strategies.k_davey_mom_keltner.backtest_k_davey_mom_keltner_v2 as kd_runner
import src.strategies.rsi_reversion.backtest_rsi_reversion_v2 as rsi_runner


INDEX_COLUMNS = [
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


STRATEGY_JOBS: dict[str, dict[str, Any]] = {
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
        "module": don_runner,
        "func": don_runner.run_backtest_donchian_breakout_v2,
        "params": {"donchian_lookback": 20},
    },
    "ema_trend_hold": {
        "module": trend_runner,
        "func": trend_runner.run_backtest_ema_trend_hold_v2,
        "params": {"trend_ema": 200},
    },
    "bmsb": {
        "module": bmsb_runner,
        "func": bmsb_runner.run_backtest_bmsb_v2,
        "params": {"sma_period": 20, "ema_period": 21, "tensignal_window": 3},
    },
    "emalyarovich_smas": {
        "module": sma_runner,
        "func": sma_runner.run_backtest_emalyarovich_smas_v2,
        "params": {"sma_fast": 20, "sma_slow": 200, "slope_bars": 3},
    },
    "k_davey_mom_keltner": {
        "module": kd_runner,
        "func": kd_runner.run_backtest_k_davey_mom_keltner_v2,
        "params": {"symbol": "MES", "position_mode": "fixed", "trade_size": 1.0},
    },
    "basic_keltner_reversion": {
        "module": bk_runner,
        "func": bk_runner.run_backtest_basic_keltner_reversion_v2,
        "params": {"kc_ema_length": 20, "kc_atr_length": 20, "kc_atr_mult": 1.5},
    },
}


# Okabe-Ito style colorblind-friendly palette
CB_BLUE = "#0072B2"
CB_ORANGE = "#E69F00"
CB_SKY = "#56B4E9"
CB_TEAL = "#009E73"
CB_VERMILLION = "#D55E00"
CB_PURPLE = "#CC79A7"
CB_GREY = "#7F7F7F"


@contextmanager
def _patched_attr(module: object, attr_name: str, replacement: Any):
    original = getattr(module, attr_name)
    setattr(module, attr_name, replacement)
    try:
        yield
    finally:
        setattr(module, attr_name, original)


def _supported_strategies() -> list[str]:
    return [s for s in SUPPORTED_WALK_FORWARD_STRATEGIES if s in STRATEGY_JOBS]


def _trade_path_from_rel(base_path: Path, csv_rel_path: str) -> Path:
    return base_path / "static" / csv_rel_path


def _run_deterministic_backtest(strategy: str, rows: int) -> tuple[pd.DataFrame, str]:
    if strategy not in STRATEGY_JOBS:
        raise ValueError(f"Unsupported strategy='{strategy}'")

    synthetic_df = make_synthetic_ohlcv_v1(rows=int(rows), freq="D")

    def fake_fetch(**_kwargs):
        return synthetic_df.copy()

    spec = STRATEGY_JOBS[strategy]
    module = spec["module"]
    run_fn = spec["func"]

    with tempfile.TemporaryDirectory(prefix=f"trade_explain_{strategy}_") as tmp_dir:
        base_path = Path(tmp_dir)
        kwargs = {
            "exchange": "binance",
            "symbol": "BTC/USDT",
            "timeframe": "1d",
            "start_date": "2021-01-01",
            "end_date": "2022-12-31",
            "use_clean": True,
            "run_id": "trade_explainability",
            "generate_report": False,
            "generate_plots": False,
            "generate_equity": False,
            "base_path": str(base_path),
            **spec["params"],
        }

        with _patched_attr(module, "fetch_ohlcv", fake_fetch):
            _stats, _chart, csv_rel_path = run_fn(**kwargs)

        if not csv_rel_path:
            return pd.DataFrame(), "deterministic_backtest"

        trade_csv_path = _trade_path_from_rel(base_path, csv_rel_path)
        if not trade_csv_path.exists():
            return pd.DataFrame(), "deterministic_backtest"

        trades_df = pd.read_csv(trade_csv_path)
        return trades_df, "deterministic_backtest"


def _normalize_trade_timestamps(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return trades_df

    out = trades_df.copy()
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True, errors="coerce")
    out["exit_time"] = pd.to_datetime(out["exit_time"], utc=True, errors="coerce")
    return out


def _timestamp_lookup(df: pd.DataFrame) -> tuple[pd.Series, dict[int, int]]:
    ts_utc = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    lookup: dict[int, int] = {}
    for idx, ts in enumerate(ts_utc):
        if pd.isna(ts):
            continue
        lookup[int(ts.value)] = idx
    return ts_utc, lookup


def _locate_index(ts_utc: pd.Series, lookup: dict[int, int], ts: Any) -> int | None:
    parsed = pd.to_datetime(ts, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None

    key = int(parsed.value)
    exact = lookup.get(key)
    if exact is not None:
        return exact

    arr = ts_utc.view("int64")
    pos = int(arr.searchsorted(key))
    if pos <= 0:
        return 0
    if pos >= len(arr):
        return len(arr) - 1

    left = pos - 1
    right = pos
    if abs(arr[left] - key) <= abs(arr[right] - key):
        return left
    return right


def _to_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        out = float(value)
        if math.isnan(out):
            return default
        return out
    except (TypeError, ValueError):
        return default


def _fmt_price(value: Any) -> str:
    val = _to_float(value)
    if val is None:
        return "n/a"
    return f"{val:.6f}"


def _fmt_bool(value: bool) -> str:
    return "MET" if bool(value) else "NOT_MET"


def _pick_int_from_trades(trades_df: pd.DataFrame, column: str, default: int) -> int:
    if column not in trades_df.columns:
        return int(default)
    numeric = pd.to_numeric(trades_df[column], errors="coerce").dropna()
    if numeric.empty:
        return int(default)
    return int(numeric.iloc[0])


def _prepare_indicator_frame(strategy: str, price_df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    df = price_df.copy()
    if strategy == "ema_cross":
        ema_fast = _pick_int_from_trades(trades_df, "ema_fast", 20)
        ema_slow = _pick_int_from_trades(trades_df, "ema_slow", 50)
        df["ema_fast"] = df["close"].ewm(span=ema_fast, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=ema_slow, adjust=False).mean()
        df["ema_trend"] = df["close"].ewm(span=200, adjust=False).mean()
    elif strategy == "rsi_reversion":
        rsi_period = _pick_int_from_trades(trades_df, "rsi_period", 14)
        df["rsi"] = compute_rsi(df["close"], rsi_period)
    elif strategy == "bmsb":
        sma_period = _pick_int_from_trades(trades_df, "bmsb_sma", 20)
        ema_period = _pick_int_from_trades(trades_df, "bmsb_ema", 21)
        tensignal_window = _pick_int_from_trades(trades_df, "bmsb_tensignal_window", 3)
        df = compute_bmsb(df, sma_period=sma_period, ema_period=ema_period)
        df["tensignal"] = compute_tensignal(df, tensignal_window)
    elif strategy == "donchian_breakout":
        donchian_lookback = _pick_int_from_trades(trades_df, "donchian_lookback", 20)
        df = compute_donchian(df, donchian_lookback)
    return df


def _plot_trade_window(
    strategy: str,
    df: pd.DataFrame,
    trade_row: pd.Series,
    entry_idx: int,
    exit_idx: int,
    context_bars: int,
    timeframe_label: str,
    image_path: Path,
    high_contrast: bool = False,
) -> None:
    left = max(0, min(entry_idx, exit_idx) - int(context_bars))
    right = min(len(df) - 1, max(entry_idx, exit_idx) + int(context_bars))
    window = df.iloc[left : right + 1].copy()

    fig, ax = plt.subplots(figsize=(12, 5))
    rsi_ax = None
    ts = pd.to_datetime(window["timestamp"], errors="coerce")
    x = mdates.date2num(ts)

    candle_wick_width = 0.8
    candle_body_alpha = 0.45
    candle_body_edge_color = None
    candle_body_edge_width = 0.7
    close_line_width = 1.2
    overlay_line_width = 1.0
    overlay_styles: dict[str, str] = {}
    marker_size = 80
    marker_edge_width = 0.4
    grid_alpha = 0.25

    if high_contrast:
        candle_wick_width = 1.2
        candle_body_alpha = 0.7
        candle_body_edge_color = "#111111"
        candle_body_edge_width = 1.1
        close_line_width = 2.2
        overlay_line_width = 1.8
        overlay_styles = {
            "ema_fast": "-",
            "ema_slow": "--",
            "ema_trend": ":",
            "bmsb": "-",
            "bmsb_sma": "--",
            "bmsb_ema": ":",
            "donchian_high": "--",
            "donchian_low": "-",
            "rsi": "-",
            "rsi_entry": "--",
            "rsi_exit": ":",
        }
        marker_size = 130
        marker_edge_width = 1.0
        grid_alpha = 0.4

    if len(x) >= 2:
        candle_width = float((x[1] - x[0]) * 0.7)
    else:
        candle_width = 0.6

    # Full OHLC candle rendering: wick + body (colorblind-friendly colors)
    for i, row in window.iterrows():
        xi = x[window.index.get_loc(i)]
        open_p = _to_float(row.get("open"), default=0.0)
        high_p = _to_float(row.get("high"), default=0.0)
        low_p = _to_float(row.get("low"), default=0.0)
        close_p = _to_float(row.get("close"), default=0.0)

        up = close_p >= open_p
        body_color = CB_TEAL if up else CB_VERMILLION

        ax.vlines(xi, low_p, high_p, color=body_color, linewidth=candle_wick_width, alpha=0.9)
        lower = min(open_p, close_p)
        height = max(abs(close_p - open_p), 1e-8)
        rect = Rectangle(
            (xi - candle_width / 2.0, lower),
            candle_width,
            height,
            facecolor=body_color,
            edgecolor=candle_body_edge_color or body_color,
            alpha=candle_body_alpha,
            linewidth=candle_body_edge_width,
        )
        ax.add_patch(rect)

    ax.plot(ts, window["close"], label="close", color=CB_BLUE, linewidth=close_line_width)

    if strategy == "ema_cross":
        for col, color in (("ema_fast", CB_ORANGE), ("ema_slow", CB_PURPLE), ("ema_trend", CB_SKY)):
            if col in window.columns:
                ax.plot(
                    ts,
                    window[col],
                    label=col,
                    color=color,
                    linewidth=overlay_line_width,
                    linestyle=overlay_styles.get(col, "-"),
                )
    elif strategy == "bmsb":
        for col, color in (("bmsb", CB_ORANGE), ("bmsb_sma", CB_TEAL), ("bmsb_ema", CB_PURPLE)):
            if col in window.columns:
                ax.plot(
                    ts,
                    window[col],
                    label=col,
                    color=color,
                    linewidth=overlay_line_width,
                    linestyle=overlay_styles.get(col, "-"),
                )
    elif strategy == "donchian_breakout":
        for col, color in (("donchian_high", CB_ORANGE), ("donchian_low", CB_PURPLE)):
            if col in window.columns:
                ax.plot(
                    ts,
                    window[col],
                    label=col,
                    color=color,
                    linewidth=overlay_line_width,
                    linestyle=overlay_styles.get(col, "-"),
                )
    elif strategy == "rsi_reversion" and "rsi" in window.columns:
        rsi_ax = ax.twinx()
        rsi_ax.plot(
            ts,
            window["rsi"],
            label="rsi",
            color=CB_ORANGE,
            linewidth=overlay_line_width,
            linestyle=overlay_styles.get("rsi", "-"),
        )
        rsi_entry = _to_float(trade_row.get("rsi_entry"), default=30.0)
        rsi_exit = _to_float(trade_row.get("rsi_exit"), default=50.0)
        rsi_ax.axhline(
            y=rsi_entry,
            color=CB_TEAL,
            linewidth=overlay_line_width,
            linestyle=overlay_styles.get("rsi_entry", "--"),
            label="rsi_entry",
            alpha=0.9,
        )
        rsi_ax.axhline(
            y=rsi_exit,
            color=CB_PURPLE,
            linewidth=overlay_line_width,
            linestyle=overlay_styles.get("rsi_exit", ":"),
            label="rsi_exit",
            alpha=0.9,
        )
        rsi_ax.set_ylim(0, 100)
        rsi_ax.set_ylabel("rsi")

    entry_ts = pd.to_datetime(trade_row.get("entry_time"), errors="coerce")
    exit_ts = pd.to_datetime(trade_row.get("exit_time"), errors="coerce")
    entry_price = _to_float(trade_row.get("entry_price"), default=0.0)
    exit_price = _to_float(trade_row.get("exit_price"), default=0.0)

    if pd.notna(entry_ts) and entry_price is not None:
        ax.scatter(
            [entry_ts],
            [entry_price],
            marker="^",
            s=marker_size,
            color=CB_BLUE,
            edgecolor="black",
            linewidth=marker_edge_width,
            label="entry",
        )
    if pd.notna(exit_ts) and exit_price is not None:
        ax.scatter(
            [exit_ts],
            [exit_price],
            marker="v",
            s=marker_size,
            color=CB_PURPLE,
            edgecolor="black",
            linewidth=marker_edge_width,
            label="exit",
        )

    trade_no = int(trade_row.get("trade_number", 0))
    side = str(trade_row.get("side", "n/a"))
    net_pnl = _to_float(trade_row.get("net_pnl"), default=0.0)
    ax.set_title(f"{strategy} trade #{trade_no:03d} ({side}) net_pnl={net_pnl:.4f}")
    ax.set_xlabel(f"timestamp ({timeframe_label})")
    ax.set_ylabel("price")
    ax.grid(alpha=grid_alpha, color=CB_GREY)
    handles, labels = ax.get_legend_handles_labels()
    if rsi_ax is not None:
        rsi_handles, rsi_labels = rsi_ax.get_legend_handles_labels()
        handles.extend(rsi_handles)
        labels.extend(rsi_labels)
    ax.legend(handles, labels, loc="best", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()

    fig.tight_layout()
    image_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(image_path, dpi=140)
    plt.close(fig)


def _explain_ema_cross(df: pd.DataFrame, trade_row: pd.Series, entry_idx: int, exit_idx: int) -> str:
    side = str(trade_row.get("side", "")).upper()
    entry_signal_idx = max(1, entry_idx - 1)
    exit_signal_idx = max(1, exit_idx - 1)

    def _cross_values(signal_idx: int) -> tuple[pd.Series, pd.Series]:
        prev_bar = df.iloc[signal_idx - 1]
        last_bar = df.iloc[signal_idx]
        return prev_bar, last_bar

    prev_entry, last_entry = _cross_values(entry_signal_idx)
    bullish_cross = bool(prev_entry["ema_fast"] < prev_entry["ema_slow"] and last_entry["ema_fast"] > last_entry["ema_slow"])
    bearish_cross = bool(prev_entry["ema_fast"] > prev_entry["ema_slow"] and last_entry["ema_fast"] < last_entry["ema_slow"])
    trend_long = bool(last_entry["close"] > last_entry["ema_trend"])
    trend_short = bool(last_entry["close"] < last_entry["ema_trend"])

    entry_checks = [
        f"- bullish_cross(prev<slow && last>slow): {_fmt_bool(bullish_cross)}",
        f"- bearish_cross(prev>slow && last<slow): {_fmt_bool(bearish_cross)}",
        f"- trend_filter_long(close>ema_trend): {_fmt_bool(trend_long)}",
        f"- trend_filter_short(close<ema_trend): {_fmt_bool(trend_short)}",
    ]

    exit_trigger = str(trade_row.get("exit_trigger", ""))
    exit_checks: list[str] = []
    exit_low = _to_float(df.iloc[exit_idx].get("low"))
    exit_high = _to_float(df.iloc[exit_idx].get("high"))
    stop_price = _to_float(trade_row.get("stop_price"))
    tp_price = _to_float(trade_row.get("take_profit_price"))

    if "stop_loss" in exit_trigger.lower() and stop_price is not None:
        if side == "LONG":
            stop_hit = exit_low is not None and exit_low <= stop_price
        else:
            stop_hit = exit_high is not None and exit_high >= stop_price
        exit_checks.append(f"- stop_loss_hit_vs_bar_extreme: {_fmt_bool(stop_hit)}")
    elif "take_profit" in exit_trigger.lower() and tp_price is not None:
        if side == "LONG":
            tp_hit = exit_high is not None and exit_high >= tp_price
        else:
            tp_hit = exit_low is not None and exit_low <= tp_price
        exit_checks.append(f"- take_profit_hit_vs_bar_extreme: {_fmt_bool(tp_hit)}")
    else:
        prev_exit, last_exit = _cross_values(exit_signal_idx)
        bull_exit_cross = bool(prev_exit["ema_fast"] < prev_exit["ema_slow"] and last_exit["ema_fast"] > last_exit["ema_slow"])
        bear_exit_cross = bool(prev_exit["ema_fast"] > prev_exit["ema_slow"] and last_exit["ema_fast"] < last_exit["ema_slow"])
        exit_checks.extend(
            [
                f"- bullish_cross_for_exit(prev<slow && last>slow): {_fmt_bool(bull_exit_cross)}",
                f"- bearish_cross_for_exit(prev>slow && last<slow): {_fmt_bool(bear_exit_cross)}",
            ]
        )

    entry_exec = df.iloc[entry_idx]
    exit_exec = df.iloc[exit_idx]

    return "\n".join(
        [
            "mode: deterministic_full",
            "strategy_explainability: ema_cross",
            "entry_indicator_values:",
            f"- entry_exec_ema_fast: {_fmt_price(entry_exec.get('ema_fast'))}",
            f"- entry_exec_ema_slow: {_fmt_price(entry_exec.get('ema_slow'))}",
            f"- entry_exec_ema_trend: {_fmt_price(entry_exec.get('ema_trend'))}",
            f"- entry_signal_ema_fast: {_fmt_price(last_entry.get('ema_fast'))}",
            f"- entry_signal_ema_slow: {_fmt_price(last_entry.get('ema_slow'))}",
            f"- entry_signal_ema_trend: {_fmt_price(last_entry.get('ema_trend'))}",
            "entry_conditions:",
            *entry_checks,
            "exit_indicator_values:",
            f"- exit_exec_ema_fast: {_fmt_price(exit_exec.get('ema_fast'))}",
            f"- exit_exec_ema_slow: {_fmt_price(exit_exec.get('ema_slow'))}",
            f"- exit_exec_ema_trend: {_fmt_price(exit_exec.get('ema_trend'))}",
            "exit_conditions:",
            *exit_checks,
        ]
    )


def _explain_bmsb(df: pd.DataFrame, trade_row: pd.Series, entry_idx: int, exit_idx: int) -> str:
    side = str(trade_row.get("side", "")).upper()
    entry_bar = df.iloc[entry_idx]
    exit_bar = df.iloc[exit_idx]
    prev_exit_idx = max(0, exit_idx - 1)
    prev_exit_bar = df.iloc[prev_exit_idx]

    entry_buysignal = bool(entry_bar["close"] > entry_bar["bmsb"])
    entry_tensignal = _to_float(entry_bar.get("tensignal"), default=0.0)
    entry_tensignal_ok = entry_tensignal >= 1.0

    entry_checks = [
        f"- buysignal(close>bmsb): {_fmt_bool(entry_buysignal)}",
        f"- tensignal_ge_1: {_fmt_bool(entry_tensignal_ok)}",
    ]

    exit_trigger = str(trade_row.get("exit_trigger", ""))
    exit_checks: list[str] = []
    if "crossunder" in exit_trigger.lower():
        sell_prev = bool(prev_exit_bar["close"] >= prev_exit_bar["bmsb"])
        sell_now = bool(exit_bar["close"] < exit_bar["bmsb"])
        exit_checks.extend(
            [
                f"- prev_close_ge_prev_bmsb: {_fmt_bool(sell_prev)}",
                f"- close_lt_bmsb: {_fmt_bool(sell_now)}",
            ]
        )
    elif "stop_loss" in exit_trigger.lower() or "take_profit" in exit_trigger.lower():
        stop_price = _to_float(trade_row.get("stop_price"))
        tp_price = _to_float(trade_row.get("take_profit_price"))
        if "stop_loss" in exit_trigger.lower() and stop_price is not None:
            stop_hit = bool(exit_bar["low"] <= stop_price) if side == "LONG" else bool(exit_bar["high"] >= stop_price)
            exit_checks.append(f"- stop_loss_hit_vs_bar_extreme: {_fmt_bool(stop_hit)}")
        if "take_profit" in exit_trigger.lower() and tp_price is not None:
            tp_hit = bool(exit_bar["high"] >= tp_price) if side == "LONG" else bool(exit_bar["low"] <= tp_price)
            exit_checks.append(f"- take_profit_hit_vs_bar_extreme: {_fmt_bool(tp_hit)}")

    if not exit_checks:
        exit_checks.append("- explicit_exit_rule_match: NOT_MET")

    return "\n".join(
        [
            "mode: deterministic_full",
            "strategy_explainability: bmsb",
            "entry_indicator_values:",
            f"- close: {_fmt_price(entry_bar.get('close'))}",
            f"- bmsb: {_fmt_price(entry_bar.get('bmsb'))}",
            f"- bmsb_sma: {_fmt_price(entry_bar.get('bmsb_sma'))}",
            f"- bmsb_ema: {_fmt_price(entry_bar.get('bmsb_ema'))}",
            f"- tensignal: {_fmt_price(entry_bar.get('tensignal'))}",
            "entry_conditions:",
            *entry_checks,
            "exit_indicator_values:",
            f"- close: {_fmt_price(exit_bar.get('close'))}",
            f"- bmsb: {_fmt_price(exit_bar.get('bmsb'))}",
            f"- bmsb_sma: {_fmt_price(exit_bar.get('bmsb_sma'))}",
            f"- bmsb_ema: {_fmt_price(exit_bar.get('bmsb_ema'))}",
            f"- tensignal: {_fmt_price(exit_bar.get('tensignal'))}",
            "exit_conditions:",
            *exit_checks,
        ]
    )


def _explain_rsi_reversion(df: pd.DataFrame, trade_row: pd.Series, entry_idx: int, exit_idx: int) -> str:
    side = str(trade_row.get("side", "")).upper()
    entry_signal_idx = max(0, entry_idx - 1)
    exit_signal_idx = max(0, exit_idx - 1)

    entry_exec = df.iloc[entry_idx]
    entry_signal = df.iloc[entry_signal_idx]
    exit_exec = df.iloc[exit_idx]
    exit_signal = df.iloc[exit_signal_idx]

    rsi_entry = _to_float(trade_row.get("rsi_entry"), default=30.0)
    rsi_exit = _to_float(trade_row.get("rsi_exit"), default=50.0)

    entry_signal_rsi = _to_float(entry_signal.get("rsi"))
    exit_signal_rsi = _to_float(exit_signal.get("rsi"))

    entry_checks = [
        f"- signal_rsi_lt_entry_threshold: {_fmt_bool(entry_signal_rsi is not None and entry_signal_rsi < rsi_entry)}",
        f"- threshold_entry(rsi_entry={rsi_entry:.2f})_present: {_fmt_bool(rsi_entry is not None)}",
    ]

    exit_trigger = str(trade_row.get("exit_trigger", ""))
    exit_checks: list[str] = []
    exit_low = _to_float(exit_exec.get("low"))
    exit_high = _to_float(exit_exec.get("high"))
    stop_price = _to_float(trade_row.get("stop_price"))
    tp_price = _to_float(trade_row.get("take_profit_price"))

    if "stop_loss" in exit_trigger.lower() and stop_price is not None:
        if side == "LONG":
            stop_hit = exit_low is not None and exit_low <= stop_price
        else:
            stop_hit = exit_high is not None and exit_high >= stop_price
        exit_checks.append(f"- stop_loss_hit_vs_bar_extreme: {_fmt_bool(stop_hit)}")
    elif "take_profit" in exit_trigger.lower() and tp_price is not None:
        if side == "LONG":
            tp_hit = exit_high is not None and exit_high >= tp_price
        else:
            tp_hit = exit_low is not None and exit_low <= tp_price
        exit_checks.append(f"- take_profit_hit_vs_bar_extreme: {_fmt_bool(tp_hit)}")
    else:
        exit_checks.extend(
            [
                f"- signal_rsi_gt_exit_threshold: {_fmt_bool(exit_signal_rsi is not None and exit_signal_rsi > rsi_exit)}",
                f"- position_side_long_for_exit: {_fmt_bool(side == 'LONG')}",
            ]
        )

    return "\n".join(
        [
            "mode: deterministic_full",
            "strategy_explainability: rsi_reversion",
            "entry_indicator_values:",
            f"- entry_exec_rsi: {_fmt_price(entry_exec.get('rsi'))}",
            f"- entry_signal_rsi: {_fmt_price(entry_signal.get('rsi'))}",
            f"- rsi_entry_threshold: {_fmt_price(rsi_entry)}",
            f"- rsi_exit_threshold: {_fmt_price(rsi_exit)}",
            "entry_conditions:",
            *entry_checks,
            "exit_indicator_values:",
            f"- exit_exec_rsi: {_fmt_price(exit_exec.get('rsi'))}",
            f"- exit_signal_rsi: {_fmt_price(exit_signal.get('rsi'))}",
            f"- rsi_entry_threshold: {_fmt_price(rsi_entry)}",
            f"- rsi_exit_threshold: {_fmt_price(rsi_exit)}",
            "exit_conditions:",
            *exit_checks,
        ]
    )


def _explain_donchian_breakout(df: pd.DataFrame, trade_row: pd.Series, entry_idx: int, exit_idx: int) -> str:
    side = str(trade_row.get("side", "")).upper()
    entry_signal_idx = max(0, entry_idx - 1)
    exit_signal_idx = max(0, exit_idx - 1)

    entry_exec = df.iloc[entry_idx]
    entry_signal = df.iloc[entry_signal_idx]
    exit_exec = df.iloc[exit_idx]
    exit_signal = df.iloc[exit_signal_idx]

    lookback = _to_float(trade_row.get("donchian_lookback"), default=20.0)

    entry_signal_close = _to_float(entry_signal.get("close"))
    entry_signal_high = _to_float(entry_signal.get("donchian_high"))
    entry_checks = [
        f"- signal_close_gt_donchian_high: {_fmt_bool(entry_signal_close is not None and entry_signal_high is not None and entry_signal_close > entry_signal_high)}",
        f"- donchian_high_available: {_fmt_bool(entry_signal_high is not None)}",
    ]

    exit_trigger = str(trade_row.get("exit_trigger", ""))
    exit_checks: list[str] = []
    exit_low = _to_float(exit_exec.get("low"))
    exit_high = _to_float(exit_exec.get("high"))
    stop_price = _to_float(trade_row.get("stop_price"))
    tp_price = _to_float(trade_row.get("take_profit_price"))

    if "stop_loss" in exit_trigger.lower() and stop_price is not None:
        if side == "LONG":
            stop_hit = exit_low is not None and exit_low <= stop_price
        else:
            stop_hit = exit_high is not None and exit_high >= stop_price
        exit_checks.append(f"- stop_loss_hit_vs_bar_extreme: {_fmt_bool(stop_hit)}")
    elif "take_profit" in exit_trigger.lower() and tp_price is not None:
        if side == "LONG":
            tp_hit = exit_high is not None and exit_high >= tp_price
        else:
            tp_hit = exit_low is not None and exit_low <= tp_price
        exit_checks.append(f"- take_profit_hit_vs_bar_extreme: {_fmt_bool(tp_hit)}")
    else:
        exit_signal_close = _to_float(exit_signal.get("close"))
        exit_signal_low = _to_float(exit_signal.get("donchian_low"))
        exit_checks.extend(
            [
                f"- signal_close_lt_donchian_low: {_fmt_bool(exit_signal_close is not None and exit_signal_low is not None and exit_signal_close < exit_signal_low)}",
                f"- donchian_low_available: {_fmt_bool(exit_signal_low is not None)}",
                f"- position_side_long_for_exit: {_fmt_bool(side == 'LONG')}",
            ]
        )

    return "\n".join(
        [
            "mode: deterministic_full",
            "strategy_explainability: donchian_breakout",
            "entry_indicator_values:",
            f"- entry_exec_close: {_fmt_price(entry_exec.get('close'))}",
            f"- entry_exec_donchian_high: {_fmt_price(entry_exec.get('donchian_high'))}",
            f"- entry_exec_donchian_low: {_fmt_price(entry_exec.get('donchian_low'))}",
            f"- entry_signal_close: {_fmt_price(entry_signal.get('close'))}",
            f"- entry_signal_donchian_high: {_fmt_price(entry_signal.get('donchian_high'))}",
            f"- entry_signal_donchian_low: {_fmt_price(entry_signal.get('donchian_low'))}",
            f"- donchian_lookback: {_fmt_price(lookback)}",
            "entry_conditions:",
            *entry_checks,
            "exit_indicator_values:",
            f"- exit_exec_close: {_fmt_price(exit_exec.get('close'))}",
            f"- exit_exec_donchian_high: {_fmt_price(exit_exec.get('donchian_high'))}",
            f"- exit_exec_donchian_low: {_fmt_price(exit_exec.get('donchian_low'))}",
            f"- exit_signal_close: {_fmt_price(exit_signal.get('close'))}",
            f"- exit_signal_donchian_high: {_fmt_price(exit_signal.get('donchian_high'))}",
            f"- exit_signal_donchian_low: {_fmt_price(exit_signal.get('donchian_low'))}",
            f"- donchian_lookback: {_fmt_price(lookback)}",
            "exit_conditions:",
            *exit_checks,
        ]
    )


def _collect_best_effort_indicators(df: pd.DataFrame, idx: int) -> list[str]:
    if idx < 0 or idx >= len(df):
        return []

    excluded = {"timestamp", "open", "high", "low", "close", "volume"}
    numeric_cols = [
        col
        for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]
    out: list[str] = []
    for col in numeric_cols[:8]:
        out.append(f"- {col}: {_fmt_price(df.iloc[idx].get(col))}")
    return out


def _explain_fallback(strategy: str, df: pd.DataFrame, entry_idx: int, exit_idx: int, trade_row: pd.Series) -> str:
    entry_indicators = _collect_best_effort_indicators(df, entry_idx)
    exit_indicators = _collect_best_effort_indicators(df, exit_idx)

    if not entry_indicators:
        entry_indicators = ["- none_available"]
    if not exit_indicators:
        exit_indicators = ["- none_available"]

    entry_trigger = str(trade_row.get("entry_trigger", "")).strip()
    exit_trigger = str(trade_row.get("exit_trigger", "")).strip()

    return "\n".join(
        [
            "mode: fallback",
            f"strategy_explainability: {strategy}",
            "note: strategy-specific deterministic condition reconstruction is not implemented",
            "entry_conditions:",
            f"- entry_trigger_present: {_fmt_bool(bool(entry_trigger))}",
            "exit_conditions:",
            f"- exit_trigger_present: {_fmt_bool(bool(exit_trigger))}",
            "entry_indicator_values_best_effort:",
            *entry_indicators,
            "exit_indicator_values_best_effort:",
            *exit_indicators,
        ]
    )


def _build_trade_explanation(
    strategy: str,
    df: pd.DataFrame,
    trade_row: pd.Series,
    entry_idx: int,
    exit_idx: int,
) -> tuple[str, bool]:
    header = "\n".join(
        [
            f"trade_number: {int(trade_row.get('trade_number', 0))}",
            f"side: {trade_row.get('side', 'n/a')}",
            f"entry_time: {trade_row.get('entry_time', 'n/a')}",
            f"exit_time: {trade_row.get('exit_time', 'n/a')}",
            f"entry_price: {_fmt_price(trade_row.get('entry_price'))}",
            f"exit_price: {_fmt_price(trade_row.get('exit_price'))}",
            f"net_pnl: {_fmt_price(trade_row.get('net_pnl'))}",
            f"entry_trigger: {trade_row.get('entry_trigger', 'n/a')}",
            f"exit_trigger: {trade_row.get('exit_trigger', 'n/a')}",
        ]
    )

    if strategy == "ema_cross":
        body = _explain_ema_cross(df, trade_row, entry_idx, exit_idx)
        return f"{header}\n{body}\n", False
    if strategy == "rsi_reversion":
        body = _explain_rsi_reversion(df, trade_row, entry_idx, exit_idx)
        return f"{header}\n{body}\n", False
    if strategy == "donchian_breakout":
        body = _explain_donchian_breakout(df, trade_row, entry_idx, exit_idx)
        return f"{header}\n{body}\n", False
    if strategy == "bmsb":
        body = _explain_bmsb(df, trade_row, entry_idx, exit_idx)
        return f"{header}\n{body}\n", False

    body = _explain_fallback(strategy, df, entry_idx, exit_idx, trade_row)
    return f"{header}\n{body}\n", True


def _resolve_trades(
    strategy: str,
    rows: int,
    csv_baseline_dir: Path | None,
) -> tuple[pd.DataFrame, str]:
    if csv_baseline_dir is not None:
        candidate = csv_baseline_dir / f"{strategy}.csv"
        if candidate.exists():
            return pd.read_csv(candidate), "deterministic_csv_baseline"
    return _run_deterministic_backtest(strategy=strategy, rows=rows)


def generate_trade_explainability_for_strategy(
    *,
    strategy: str,
    rows: int = 430,
    context_bars: int = 30,
    max_trades: int = 50,
    timeframe_label: str = "1d",
    high_contrast: bool = False,
    output_dir: Path,
    csv_baseline_dir: Path | None = None,
) -> dict[str, Any]:
    if strategy not in _supported_strategies():
        supported = ", ".join(_supported_strategies())
        raise ValueError(f"Unsupported strategy='{strategy}'. Supported: {supported}")

    output_dir.mkdir(parents=True, exist_ok=True)

    price_df = make_synthetic_ohlcv_v1(rows=int(rows), freq="D")
    trades_raw, source_mode = _resolve_trades(strategy, rows=int(rows), csv_baseline_dir=csv_baseline_dir)
    trades_df = _normalize_trade_timestamps(trades_raw)
    indicator_df = _prepare_indicator_frame(strategy, price_df, trades_df)

    ts_utc, lookup = _timestamp_lookup(indicator_df)

    records: list[dict[str, Any]] = []
    if not trades_df.empty:
        trade_count = min(len(trades_df), int(max_trades))
        for n in range(trade_count):
            trade_row = trades_df.iloc[n].copy()
            trade_row["trade_number"] = n + 1

            entry_idx = _locate_index(ts_utc, lookup, trade_row.get("entry_time"))
            exit_idx = _locate_index(ts_utc, lookup, trade_row.get("exit_time"))
            if entry_idx is None or exit_idx is None:
                continue

            image_file = f"trade_{n + 1:03d}.png"
            text_file = f"trade_{n + 1:03d}.txt"
            image_path = output_dir / image_file
            text_path = output_dir / text_file

            _plot_trade_window(
                strategy=strategy,
                df=indicator_df,
                trade_row=trade_row,
                entry_idx=entry_idx,
                exit_idx=exit_idx,
                context_bars=int(context_bars),
                timeframe_label=timeframe_label,
                image_path=image_path,
                high_contrast=high_contrast,
            )

            explanation_text, fallback_mode = _build_trade_explanation(
                strategy=strategy,
                df=indicator_df,
                trade_row=trade_row,
                entry_idx=entry_idx,
                exit_idx=exit_idx,
            )
            text_path.write_text(explanation_text, encoding="utf-8")

            records.append(
                {
                    "trade_number": int(n + 1),
                    "side": str(trade_row.get("side", "n/a")),
                    "entry_time": str(trade_row.get("entry_time", "")),
                    "exit_time": str(trade_row.get("exit_time", "")),
                    "entry_price": _to_float(trade_row.get("entry_price")),
                    "exit_price": _to_float(trade_row.get("exit_price")),
                    "net_pnl": _to_float(trade_row.get("net_pnl")),
                    "entry_trigger": str(trade_row.get("entry_trigger", "")),
                    "exit_trigger": str(trade_row.get("exit_trigger", "")),
                    "fallback_mode": bool(fallback_mode),
                    "image_file": image_file,
                    "explanation_file": text_file,
                }
            )

    index_df = pd.DataFrame(records, columns=INDEX_COLUMNS)
    csv_index_path = output_dir / "trade_artifacts_index.csv"
    json_index_path = output_dir / "trade_artifacts_index.json"
    index_df.to_csv(csv_index_path, index=False)

    payload = {
        "strategy": strategy,
        "source_mode": source_mode,
        "deterministic_source_of_truth": "src.core.testing.synthetic_data.make_synthetic_ohlcv_v1",
        "rows": int(rows),
        "context_bars": int(context_bars),
        "timeframe_label": timeframe_label,
        "high_contrast": bool(high_contrast),
        "max_trades": int(max_trades),
        "requested_trades": int(len(trades_df)),
        "generated_trades": int(len(index_df)),
        "fallback_trades": int(index_df["fallback_mode"].sum()) if not index_df.empty else 0,
        "artifacts": records,
    }
    json_index_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "strategy": strategy,
        "source_mode": source_mode,
        "output_dir": output_dir,
        "csv_index": csv_index_path,
        "json_index": json_index_path,
        "requested_trades": int(len(trades_df)),
        "generated_trades": int(len(index_df)),
        "fallback_trades": int(payload["fallback_trades"]),
    }


def _build_summary_warnings_for_strategy(strategy_result: dict[str, Any], max_trades: int) -> list[str]:
    warnings: list[str] = []
    strategy_name = str(strategy_result.get("strategy", "unknown"))

    requested = int(strategy_result.get("requested_trades", 0))
    generated = int(strategy_result.get("generated_trades", 0))
    fallback = int(strategy_result.get("fallback_trades", 0))

    expected_generated_max = min(requested, int(max_trades))
    if requested == 0:
        warnings.append(f"{strategy_name}: requested_trades is zero")
    if generated == 0:
        warnings.append(f"{strategy_name}: generated_trades is zero (no per-trade artifacts created)")
    if generated < expected_generated_max:
        warnings.append(
            f"{strategy_name}: generated_trades ({generated}) below expected upper bound ({expected_generated_max})"
        )
    if generated > 0 and fallback == generated:
        warnings.append(f"{strategy_name}: all generated trades used fallback explainability mode")

    csv_index = Path(strategy_result["csv_index"])
    json_index = Path(strategy_result["json_index"])
    if not csv_index.exists():
        warnings.append(f"{strategy_name}: missing csv index artifact ({csv_index.name})")
    if not json_index.exists():
        warnings.append(f"{strategy_name}: missing json index artifact ({json_index.name})")

    return warnings


def _build_run_summary_payload(
    *,
    run_output_dir: Path,
    rows: int,
    context_bars: int,
    timeframe_label: str,
    high_contrast: bool,
    max_trades: int,
    strategy_results: list[dict[str, Any]],
) -> dict[str, Any]:
    strategies_processed = [str(result["strategy"]) for result in strategy_results]
    per_strategy_counts: list[dict[str, Any]] = []
    run_warnings: list[str] = []

    for strategy_result in strategy_results:
        strategy_warnings = _build_summary_warnings_for_strategy(strategy_result, max_trades=max_trades)
        run_warnings.extend(strategy_warnings)
        per_strategy_counts.append(
            {
                "strategy": str(strategy_result["strategy"]),
                "source_mode": str(strategy_result["source_mode"]),
                "output_dir": str(strategy_result["output_dir"]),
                "csv_index": str(strategy_result["csv_index"]),
                "json_index": str(strategy_result["json_index"]),
                "requested_trades": int(strategy_result["requested_trades"]),
                "generated_trades": int(strategy_result["generated_trades"]),
                "fallback_trades": int(strategy_result["fallback_trades"]),
                "warnings": strategy_warnings,
            }
        )

    total_generated_trades = sum(int(item["generated_trades"]) for item in per_strategy_counts)
    return {
        "run_output_dir": str(run_output_dir),
        "run_metadata": {
            "rows": int(rows),
            "timeframe_label": timeframe_label,
            "context_bars": int(context_bars),
            "max_trades": int(max_trades),
            "high_contrast": bool(high_contrast),
        },
        "strategies_processed": strategies_processed,
        "per_strategy_counts": per_strategy_counts,
        "total_generated_trades": int(total_generated_trades),
        "warnings": run_warnings,
    }


def _write_run_summary_files(run_output_dir: Path, summary_payload: dict[str, Any]) -> tuple[Path, Path]:
    run_output_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = run_output_dir / "summary.json"
    summary_report_path = run_output_dir / "summary_report.txt"

    summary_json_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report_lines = [
        "trade_explainability_run_summary",
        f"run_output_dir: {summary_payload['run_output_dir']}",
        "run_metadata:",
        f"- rows: {summary_payload['run_metadata']['rows']}",
        f"- timeframe_label: {summary_payload['run_metadata']['timeframe_label']}",
        f"- context_bars: {summary_payload['run_metadata']['context_bars']}",
        f"- max_trades: {summary_payload['run_metadata']['max_trades']}",
        f"- high_contrast: {summary_payload['run_metadata']['high_contrast']}",
        "strategies_processed:",
    ]

    for strategy_name in summary_payload["strategies_processed"]:
        report_lines.append(f"- {strategy_name}")

    report_lines.append("per_strategy_counts:")
    for item in summary_payload["per_strategy_counts"]:
        report_lines.append(
            "- {strategy}: requested_trades={requested_trades}, generated_trades={generated_trades}, "
            "fallback_trades={fallback_trades}".format(**item)
        )

    report_lines.append(f"total_generated_trades: {summary_payload['total_generated_trades']}")
    report_lines.append("warnings:")
    if summary_payload["warnings"]:
        for warning in summary_payload["warnings"]:
            report_lines.append(f"- {warning}")
    else:
        report_lines.append("- none")

    summary_report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return summary_report_path, summary_json_path


def generate_trade_explainability(
    *,
    strategy: str,
    rows: int = 430,
    context_bars: int = 30,
    max_trades: int = 50,
    timeframe_label: str = "1d",
    high_contrast: bool = False,
    run_output_dir: Path,
    csv_baseline_dir: Path | None = None,
) -> dict[str, Any]:
    if strategy == "all":
        selected = _supported_strategies()
    else:
        selected = [strategy]

    results: list[dict[str, Any]] = []
    for strategy_name in selected:
        strategy_output_dir = run_output_dir / strategy_name
        result = generate_trade_explainability_for_strategy(
            strategy=strategy_name,
            rows=rows,
            context_bars=context_bars,
            max_trades=max_trades,
            timeframe_label=timeframe_label,
            high_contrast=high_contrast,
            output_dir=strategy_output_dir,
            csv_baseline_dir=csv_baseline_dir,
        )
        results.append(result)

    summary_payload = _build_run_summary_payload(
        run_output_dir=run_output_dir,
        rows=rows,
        context_bars=context_bars,
        timeframe_label=timeframe_label,
        high_contrast=high_contrast,
        max_trades=max_trades,
        strategy_results=results,
    )
    summary_report_path, summary_json_path = _write_run_summary_files(
        run_output_dir=run_output_dir,
        summary_payload=summary_payload,
    )

    return {
        "run_output_dir": run_output_dir,
        "rows": int(rows),
        "context_bars": int(context_bars),
        "timeframe_label": timeframe_label,
        "high_contrast": bool(high_contrast),
        "max_trades": int(max_trades),
        "strategies": results,
        "summary_report": summary_report_path,
        "summary_json": summary_json_path,
    }
