import json
import os
import uuid
import pandas as pd

from datetime import datetime, timezone
from flask import Flask, render_template, request, redirect, url_for, jsonify, current_app
from src.core.database import get_connection, init_db
from src.core.data import fetch_ohlcv, date_to_ms
from src.core.plotting.plot_trades import plot_trades_candlestick, plot_trades_candlestick_windows
from src.strategies.ema_cross.backtest_ema_cross_v2 import run_backtest_ema_cross_v2
from src.strategies.rsi_reversion.backtest_rsi_reversion_v2 import (
    run_backtest_rsi_reversion_v2,
)
from src.strategies.donchian_breakout.backtest_donchian_breakout_v2 import (
    run_backtest_donchian_breakout_v2,
)
from src.strategies.ema_trend_hold.backtest_ema_trend_hold_v2 import (
    run_backtest_ema_trend_hold_v2,
)
from src.strategies.bmsb.backtest_bmsb_v2 import run_backtest_bmsb_v2
from src.strategies.emalyarovich_smas.backtest_emalyarovich_smas_v2 import (
    run_backtest_emalyarovich_smas_v2,
)
from src.strategies.k_davey_mom_keltner.backtest_k_davey_mom_keltner_v2 import (
    run_backtest_k_davey_mom_keltner_v2,
)
from src.strategies.basic_keltner_reversion.backtest_basic_keltner_reversion_v2 import (
    run_backtest_basic_keltner_reversion_v2,
)

app = Flask(__name__,template_folder="templates",static_folder="static")
init_db()

TRADE_VIEWER_CONTEXT_BEFORE = 5
TRADE_VIEWER_CONTEXT_AFTER = 5
TRADE_VIEWER_MAX_CANDLES_NO_CONFIRM = 50


def _date_to_query_ms(value, is_end=False):
    ts = date_to_ms(value)
    if ts is None:
        return None

    value_str = str(value).strip() if value is not None else ""
    if is_end and value_str and len(value_str) == 10:
        ts += 24 * 60 * 60 * 1000 - 1

    return int(ts)


def _run_date_window_text(run, params):
    start_from_params = str(params.get("start_date") or "").strip()
    end_from_params = str(params.get("end_date") or "").strip()

    start_text = start_from_params
    end_text = end_from_params

    if not start_text and run.get("start_ts") is not None:
        start_ts = pd.to_datetime(run["start_ts"], unit="ms", errors="coerce", utc=True)
        if pd.notna(start_ts):
            start_text = start_ts.strftime("%Y-%m-%d")

    if not end_text and run.get("end_ts") is not None:
        end_ts = pd.to_datetime(run["end_ts"], unit="ms", errors="coerce", utc=True)
        if pd.notna(end_ts):
            end_text = end_ts.strftime("%Y-%m-%d")

    if not start_text:
        start_text = "earliest available"
    if not end_text:
        end_text = "latest available"

    return f"{start_text} -> {end_text}"


def _build_no_trades_message(date_window_text):
    return (
        "No closed trades were recorded for this run in the selected "
        f"date range: {date_window_text}."
    )


def _extract_data_issue_message(stats):
    if not isinstance(stats, dict):
        return None

    no_data_reason = str(stats.get("No Data Reason") or "").strip()
    if no_data_reason:
        return f"No backtest results available: {no_data_reason}"

    return None


def _load_trades_for_run(run):
    csv_rel_path = run.get("csv_path")
    if not csv_rel_path:
        return None, "Trades CSV not available for this run."

    csv_abs_path = os.path.join(current_app.static_folder, csv_rel_path)
    if not os.path.exists(csv_abs_path):
        return None, "Trades CSV file was not found on disk."

    trades_df = pd.read_csv(csv_abs_path)
    if trades_df.empty:
        return None, "No trades available for this run."

    for col in ["entry_time", "exit_time"]:
        if col in trades_df.columns:
            ts = pd.to_datetime(trades_df[col], utc=True, errors="coerce")
            trades_df[col] = ts.dt.tz_convert(None)

    if "entry_time" not in trades_df.columns or "exit_time" not in trades_df.columns:
        return None, "Trades CSV is missing entry/exit timestamps."

    trades_df = trades_df.dropna(subset=["entry_time", "exit_time"]).copy()
    if trades_df.empty:
        return None, "Trades CSV has no valid timestamps."

    if "trade_number" not in trades_df.columns:
        trades_df["trade_number"] = range(1, len(trades_df) + 1)
    else:
        trades_df["trade_number"] = pd.to_numeric(trades_df["trade_number"], errors="coerce").fillna(0).astype(int)

    return trades_df, None


def _build_trade_selector_options(trades_df):
    options = []
    for row in trades_df.to_dict("records"):
        trade_number = int(row.get("trade_number", 0))
        side = str(row.get("side", "n/a")).upper()
        entry_time = row.get("entry_time")
        exit_time = row.get("exit_time")
        entry_txt = pd.to_datetime(entry_time).strftime("%Y-%m-%d %H:%M") if pd.notna(entry_time) else "n/a"
        exit_txt = pd.to_datetime(exit_time).strftime("%Y-%m-%d %H:%M") if pd.notna(exit_time) else "n/a"
        pnl = pd.to_numeric(row.get("net_pnl", 0.0), errors="coerce")
        pnl_value = 0.0 if pd.isna(pnl) else float(pnl)
        pnl_txt = f"{pnl_value:.2f}"
        label = f"#{trade_number:03d} | {side} | {entry_txt} -> {exit_txt} | net_pnl={pnl_txt}"
        options.append(
            {
                "trade_number": trade_number,
                "label": label,
                "side": side,
                "net_pnl": pnl_value,
            }
        )
    return options


def _build_indicator_overlays(ohlc_df, strategy, params):
    if ohlc_df is None or ohlc_df.empty:
        return None

    series = pd.to_numeric(ohlc_df["close"], errors="coerce")
    indicators = {}

    if strategy == "ema_cross":
        ema_fast = int(params.get("ema_fast", 20))
        ema_slow = int(params.get("ema_slow", 50))
        indicators[f"ema_fast_{ema_fast}"] = series.ewm(span=ema_fast, adjust=False).mean()
        indicators[f"ema_slow_{ema_slow}"] = series.ewm(span=ema_slow, adjust=False).mean()
    elif strategy == "ema_trend_hold":
        trend_ema = int(params.get("trend_ema", 200))
        indicators[f"trend_ema_{trend_ema}"] = series.ewm(span=trend_ema, adjust=False).mean()
    elif strategy == "donchian_breakout":
        lookback = int(params.get("donchian_lookback", 20))
        rolling_high = pd.to_numeric(ohlc_df["high"], errors="coerce").rolling(lookback).max().shift(1)
        rolling_low = pd.to_numeric(ohlc_df["low"], errors="coerce").rolling(lookback).min().shift(1)
        indicators[f"donchian_high_{lookback}"] = rolling_high
        indicators[f"donchian_low_{lookback}"] = rolling_low
    elif strategy == "bmsb":
        sma_period = int(params.get("bmsb_sma", 20))
        ema_period = int(params.get("bmsb_ema", 21))
        indicators[f"sma_{sma_period}"] = series.rolling(sma_period).mean()
        indicators[f"ema_{ema_period}"] = series.ewm(span=ema_period, adjust=False).mean()
    elif strategy == "emalyarovich_smas":
        sma_fast = int(params.get("sma_fast", 20))
        sma_slow = int(params.get("sma_slow", 200))
        indicators[f"sma_fast_{sma_fast}"] = series.rolling(sma_fast).mean()
        indicators[f"sma_slow_{sma_slow}"] = series.rolling(sma_slow).mean()
    elif strategy == "basic_keltner_reversion":
        ema_len = int(params.get("kc_rev_ema_length", 20))
        indicators[f"kc_ema_{ema_len}"] = series.ewm(span=ema_len, adjust=False).mean()
    elif strategy == "k_davey_mom_keltner":
        trend_ema = int(params.get("keltner_trend_ema", 200))
        indicators[f"trend_ema_{trend_ema}"] = series.ewm(span=trend_ema, adjust=False).mean()

    return indicators or None


def _build_indicator_toggle_options(strategy, params):
    options = []

    if strategy == "ema_cross":
        ema_fast = int(params.get("ema_fast", 20))
        ema_slow = int(params.get("ema_slow", 50))
        options = [
            {"value": f"ema_fast_{ema_fast}", "label": f"EMA Fast ({ema_fast})", "checked": True},
            {"value": f"ema_slow_{ema_slow}", "label": f"EMA Slow ({ema_slow})", "checked": True},
        ]
    elif strategy == "ema_trend_hold":
        trend_ema = int(params.get("trend_ema", 200))
        options = [
            {"value": f"trend_ema_{trend_ema}", "label": f"Trend EMA ({trend_ema})", "checked": True},
        ]
    elif strategy == "donchian_breakout":
        lookback = int(params.get("donchian_lookback", 20))
        options = [
            {"value": f"donchian_high_{lookback}", "label": f"Donchian High ({lookback})", "checked": True},
            {"value": f"donchian_low_{lookback}", "label": f"Donchian Low ({lookback})", "checked": True},
        ]
    elif strategy == "bmsb":
        sma_period = int(params.get("bmsb_sma", 20))
        ema_period = int(params.get("bmsb_ema", 21))
        options = [
            {"value": f"sma_{sma_period}", "label": f"SMA ({sma_period})", "checked": True},
            {"value": f"ema_{ema_period}", "label": f"EMA ({ema_period})", "checked": True},
        ]
    elif strategy == "emalyarovich_smas":
        sma_fast = int(params.get("sma_fast", 20))
        sma_slow = int(params.get("sma_slow", 200))
        options = [
            {"value": f"sma_fast_{sma_fast}", "label": f"SMA Fast ({sma_fast})", "checked": True},
            {"value": f"sma_slow_{sma_slow}", "label": f"SMA Slow ({sma_slow})", "checked": True},
        ]
    elif strategy == "basic_keltner_reversion":
        ema_len = int(params.get("kc_rev_ema_length", 20))
        options = [
            {"value": f"kc_ema_{ema_len}", "label": f"Keltner EMA ({ema_len})", "checked": True},
        ]
    elif strategy == "k_davey_mom_keltner":
        trend_ema = int(params.get("keltner_trend_ema", 200))
        options = [
            {"value": f"trend_ema_{trend_ema}", "label": f"Trend EMA ({trend_ema})", "checked": True},
        ]

    return options


def _closest_index_position(index, ts):
    if len(index) == 0:
        return None
    pos = index.searchsorted(ts)
    if pos <= 0:
        return 0
    if pos >= len(index):
        return len(index) - 1
    before = index[pos - 1]
    after = index[pos]
    return pos - 1 if abs(ts - before) <= abs(after - ts) else pos


def _safe_float(value, default=0.0):
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return float(default)
    return float(numeric)


def _format_trade_viewer_time(value):
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return "n/a"
    return str(ts)


def _write_trade_packet_txt(file_abs_path, payload):
    selected_indicators = payload.get("selected_indicators") or []
    indicators_line = ", ".join(selected_indicators) if selected_indicators else "none"

    lines = [
        "Single Trade Packet",
        "===================",
        "",
        f"run_id: {payload.get('run_id', 'n/a')}",
        f"strategy: {payload.get('strategy', 'n/a')}",
        f"symbol: {payload.get('symbol', 'n/a')}",
        f"timeframe: {payload.get('timeframe', 'n/a')}",
        "",
        f"trade_number: {payload.get('trade_number', 'n/a')}",
        f"side: {payload.get('side', 'n/a')}",
        f"entry_time: {payload.get('entry_time', 'n/a')}",
        f"entry_price: {payload.get('entry_price', 0.0):.8f}",
        f"exit_time: {payload.get('exit_time', 'n/a')}",
        f"exit_price: {payload.get('exit_price', 0.0):.8f}",
        f"net_pnl: {payload.get('net_pnl', 0.0):.8f}",
        "",
        f"entry_trigger: {payload.get('entry_trigger', 'n/a')}",
        f"exit_trigger: {payload.get('exit_trigger', 'n/a')}",
        "",
        f"candles_count_window: {payload.get('candles_count', 0)}",
        f"annotation_mode: {payload.get('annotation_mode', 'n/a')}",
        f"annotation_note: {payload.get('conditions_note', 'n/a')}",
        f"selected_indicators: {indicators_line}",
    ]

    with open(file_abs_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _build_trades_chart_for_results(run, params):
    trades_df, trades_error = _load_trades_for_run(run)
    if trades_error:
        return None, None, trades_error

    start_date = trades_df["entry_time"].min().floor("D")
    end_date = trades_df["exit_time"].max().ceil("D")
    use_clean = bool(params.get("use_clean", True))

    ohlc_df = fetch_ohlcv(
        exchange=run["exchange"],
        symbol=run["symbol"],
        timeframe=run["timeframe"],
        start_date=start_date,
        end_date=end_date,
        limit=50000,
        use_clean=use_clean,
    )

    charts_rel_dir = f"backtests/{run['strategy']}/{run['run_id']}/trades_windows"
    charts_abs_dir = os.path.join(current_app.static_folder, charts_rel_dir)

    annotation_mode, chart_items, was_truncated, total_trade_windows = plot_trades_candlestick_windows(
        df=ohlc_df,
        trades=trades_df,
        title=(
            f"{run['strategy']} {run['symbol']} {run['timeframe']}\n"
            "Candlesticks + Entry/Exit Markers + Conditions"
        ),
        output_dir=charts_abs_dir,
        filename_prefix=f"trades_graph_{run['run_id']}",
        candles_per_chart=50,
        max_charts=10,
    )

    if not chart_items:
        if annotation_mode == "missing_ohlc":
            return None, None, "OHLC data is unavailable for the trades range."
        if annotation_mode == "missing_trades":
            return None, None, "No trade rows available to render entry/exit markers."
        return None, None, "Trades chart could not be generated."

    chart_entries = [
        {
            "path": f"{charts_rel_dir}/{item['filename']}",
            "label": item["window_label"],
        }
        for item in chart_items
    ]

    notes = []
    if annotation_mode == "best_effort":
        notes.append(
            "Condition labels are best-effort: rendered from entry_trigger/exit_trigger when "
            "available, with fallback labels otherwise."
        )
    if was_truncated:
        notes.append(
            f"Showing first 10 trade windows of 50 candles each (total windows with trades: {total_trade_windows})."
        )
    annotation_note = " ".join(notes) if notes else None

    return chart_entries, annotation_note, None


def _load_ohlc_for_trade_viewer(run, params, trades_df):
    run_start = params.get("start_date")
    run_end = params.get("end_date")

    start_date = run_start or trades_df["entry_time"].min().floor("D")
    end_date = run_end or trades_df["exit_time"].max().ceil("D")

    use_clean = bool(params.get("use_clean", True))
    return fetch_ohlcv(
        exchange=run["exchange"],
        symbol=run["symbol"],
        timeframe=run["timeframe"],
        start_date=start_date,
        end_date=end_date,
        limit=50000,
        use_clean=use_clean,
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run_backtest", methods=["POST"])
def run_backtest():
    strategy = request.form.get("strategy")
    exchange = request.form.get("exchange")
    symbol = request.form.get("symbol")
    timeframe = request.form.get("timeframe")
    use_clean = request.form.get("use_clean") == "1"

    ema_fast = int(request.form.get("ema_fast"))
    ema_slow = int(request.form.get("ema_slow"))
    rsi_period = int(request.form.get("rsi_period") or 14)
    rsi_entry = float(request.form.get("rsi_entry") or 30)
    rsi_exit = float(request.form.get("rsi_exit") or 50)
    donchian_lookback = int(request.form.get("donchian_lookback") or 20)
    trend_ema = int(request.form.get("trend_ema") or 200)
    bmsb_sma = int(request.form.get("bmsb_sma") or 20)
    bmsb_ema = int(request.form.get("bmsb_ema") or 21)
    bmsb_tensignal = int(request.form.get("bmsb_tensignal") or 3)
    bmsb_trail = float(request.form.get("bmsb_trail") or 0.05)
    slope_bars = int(request.form.get("slope_bars") or 3)
    sma_fast = int(request.form.get("sma_fast") or 20)
    sma_slow = int(request.form.get("sma_slow") or 200)
    mom_length_long = int(request.form.get("mom_length_long") or 40)
    mom_length_short = int(request.form.get("mom_length_short") or 40)
    keltner_length = int(request.form.get("keltner_length") or 5)
    keltner_atr_mult = float(request.form.get("keltner_atr_mult") or 0.5)
    keltner_entry_threshold = float(request.form.get("keltner_entry_threshold") or 30)
    keltner_exit_threshold = float(request.form.get("keltner_exit_threshold") or 70)
    keltner_trend_ema = int(request.form.get("keltner_trend_ema") or 200)
    keltner_vol_mult = float(request.form.get("keltner_vol_mult") or 0.8)
    keltner_vol_sma = int(request.form.get("keltner_vol_sma") or 100)
    use_position_sizing = request.form.get("use_position_sizing") != "0"
    base_equity = float(request.form.get("base_equity") or 15000)
    sizing_factor = float(request.form.get("sizing_factor") or 0.33)
    max_contracts = int(request.form.get("max_contracts") or 15)
    atr_period = int(request.form.get("atr_period") or 14)
    atr_sl_long = float(request.form.get("atr_sl_long") or 4.0)
    atr_sl_short = float(request.form.get("atr_sl_short") or 1.0)
    kc_rev_ema_length = int(request.form.get("kc_rev_ema_length") or 20)
    kc_rev_atr_length = int(request.form.get("kc_rev_atr_length") or 20)
    kc_rev_atr_mult = float(request.form.get("kc_rev_atr_mult") or 1.5)

    initial_balance = float(request.form.get("initial_balance") or 1000)
    position_mode = "all_in"
    position_pct_input = float(request.form.get("position_pct") or 3.0)
    position_pct = max(position_pct_input, 0.0) / 100.0
    trade_size = initial_balance * position_pct
    commission_pct = float(request.form.get("commission_pct") or 0.1) / 100.0
    slippage_pct = float(request.form.get("slippage_pct") or 0.1) / 100.0
    allow_short = True
    stop_loss_pct = float(request.form.get("stop_loss_pct") or 2.0) / 100.0
    take_profit_pct_raw = request.form.get("take_profit_pct")
    take_profit_pct = float(take_profit_pct_raw) / 100.0 if take_profit_pct_raw else None
    
    use_tp_sl = request.form.get("use_tp_sl") != "0"

    if not use_tp_sl:
        stop_loss_pct = None
        take_profit_pct = None
    elif stop_loss_pct <= 0:
        stop_loss_pct = None
    if take_profit_pct is not None and take_profit_pct <= 0:
        take_profit_pct = None
    pyramiding = int(request.form.get("pyramiding") or 1)

    start_date = request.form.get("start_date")
    end_date = request.form.get("end_date")
    start_ts = _date_to_query_ms(start_date)
    end_ts = _date_to_query_ms(end_date, is_end=True)

    params = {
        "start_date": start_date,
        "end_date": end_date,
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "use_clean": use_clean,
        "initial_balance": initial_balance,
        "position_mode": position_mode,
        "trade_size": trade_size,
        "position_pct": position_pct,
        "commission_pct": commission_pct,
        "slippage_pct": slippage_pct,
        "allow_short": allow_short,
        "stop_loss_pct": stop_loss_pct,
        "take_profit_pct": take_profit_pct,
        "pyramiding": pyramiding,
        "use_tp_sl": use_tp_sl,
        "rsi_period": rsi_period,
        "rsi_entry": rsi_entry,
        "rsi_exit": rsi_exit,
        "donchian_lookback": donchian_lookback,
        "trend_ema": trend_ema,
        "bmsb_sma": bmsb_sma,
        "bmsb_ema": bmsb_ema,
        "bmsb_tensignal": bmsb_tensignal,
        "bmsb_trail": bmsb_trail,
        "slope_bars": slope_bars,
        "sma_fast": sma_fast,
        "sma_slow": sma_slow,
        "mom_length_long": mom_length_long,
        "mom_length_short": mom_length_short,
        "keltner_length": keltner_length,
        "keltner_atr_mult": keltner_atr_mult,
        "keltner_entry_threshold": keltner_entry_threshold,
        "keltner_exit_threshold": keltner_exit_threshold,
        "keltner_trend_ema": keltner_trend_ema,
        "keltner_vol_mult": keltner_vol_mult,
        "keltner_vol_sma": keltner_vol_sma,
        "use_position_sizing": use_position_sizing,
        "base_equity": base_equity,
        "sizing_factor": sizing_factor,
        "max_contracts": max_contracts,
        "atr_period": atr_period,
        "atr_sl_long": atr_sl_long,
        "atr_sl_short": atr_sl_short,
        "kc_rev_ema_length": kc_rev_ema_length,
        "kc_rev_atr_length": kc_rev_atr_length,
        "kc_rev_atr_mult": kc_rev_atr_mult,
    }

    # 1) Generar run_id
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    run_id = f"{timestamp}_{unique_id}"

    #2) Ejecutar backtest
    if strategy == "rsi_reversion":
        stats, chart_path, csv_path = run_backtest_rsi_reversion_v2(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            rsi_period=rsi_period,
            rsi_entry=rsi_entry,
            rsi_exit=rsi_exit,
            run_id=run_id,
            initial_balance=initial_balance,
            position_mode=position_mode,
            trade_size=trade_size,
            position_pct=position_pct,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
            allow_short=False,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            pyramiding=pyramiding,
            base_path=current_app.root_path,
        )
    elif strategy == "donchian_breakout":
        stats, chart_path, csv_path = run_backtest_donchian_breakout_v2(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            donchian_lookback=donchian_lookback,
            run_id=run_id,
            initial_balance=initial_balance,
            position_mode=position_mode,
            trade_size=trade_size,
            position_pct=position_pct,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
            allow_short=False,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            pyramiding=pyramiding,
            base_path=current_app.root_path,
        )
    elif strategy == "ema_trend_hold":
        stats, chart_path, csv_path = run_backtest_ema_trend_hold_v2(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            trend_ema=trend_ema,
            run_id=run_id,
            initial_balance=initial_balance,
            position_mode=position_mode,
            trade_size=trade_size,
            position_pct=position_pct,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
            allow_short=False,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            pyramiding=pyramiding,
            base_path=current_app.root_path,
        )
    elif strategy == "bmsb":
        stats, chart_path, csv_path = run_backtest_bmsb_v2(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            sma_period=bmsb_sma,
            ema_period=bmsb_ema,
            tensignal_window=bmsb_tensignal,
            trail_percent=bmsb_trail,
            run_id=run_id,
            initial_balance=initial_balance,
            position_mode=position_mode,
            trade_size=trade_size,
            position_pct=position_pct,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
            allow_short=False,
            stop_loss_pct=None,
            take_profit_pct=None,
            pyramiding=pyramiding,
            use_tp_sl=use_tp_sl,
            base_path=current_app.root_path,
        )
    elif strategy == "emalyarovich_smas":
        stats, chart_path, csv_path = run_backtest_emalyarovich_smas_v2(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            sma_fast=sma_fast,
            sma_slow=sma_slow,
            slope_bars=slope_bars,
            run_id=run_id,
            initial_balance=initial_balance,
            position_mode=position_mode,
            trade_size=trade_size,
            position_pct=position_pct,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
            allow_short=False,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct or 0.03,
            pyramiding=pyramiding,
            base_path=current_app.root_path,
        )
    elif strategy == "k_davey_mom_keltner":
        stats, chart_path, csv_path = run_backtest_k_davey_mom_keltner_v2(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            mom_length_long=mom_length_long,
            mom_length_short=mom_length_short,
            keltner_length=keltner_length,
            keltner_atr_mult=keltner_atr_mult,
            entry_threshold=keltner_entry_threshold,
            exit_threshold=keltner_exit_threshold,
            trend_ema=keltner_trend_ema,
            volatility_atr_period=14,
            volatility_sma_period=keltner_vol_sma,
            volatility_mult=keltner_vol_mult,
            use_position_sizing=use_position_sizing,
            base_equity=base_equity,
            sizing_factor=sizing_factor,
            max_contracts=max_contracts,
            run_id=run_id,
            initial_balance=initial_balance,
            position_mode="contracts",
            trade_size=trade_size,
            position_pct=position_pct,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
            allow_short=True,
            atr_period=atr_period,
            atr_sl_mult_long=atr_sl_long,
            atr_sl_mult_short=atr_sl_short,
            pyramiding=pyramiding,
            base_path=current_app.root_path,
        )
    elif strategy == "basic_keltner_reversion":
        stats, chart_path, csv_path = run_backtest_basic_keltner_reversion_v2(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            kc_ema_length=kc_rev_ema_length,
            kc_atr_length=kc_rev_atr_length,
            kc_atr_mult=kc_rev_atr_mult,
            run_id=run_id,
            initial_balance=initial_balance,
            position_mode=position_mode,
            trade_size=trade_size,
            position_pct=position_pct,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
            allow_short=True,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            pyramiding=pyramiding,
            base_path=current_app.root_path,
        )
    else:
        stats, chart_path, csv_path = run_backtest_ema_cross_v2(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            run_id=run_id,
            initial_balance=initial_balance,
            position_mode=position_mode,
            trade_size=trade_size,
            position_pct=position_pct,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
            allow_short=allow_short,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            pyramiding=pyramiding,
            base_path=current_app.root_path,
        )

    #3) guardad en DB
    created_at = now.isoformat()

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO backtest_runs (
            run_id, strategy, exchange, symbol, timeframe,
            start_ts, end_ts,
            params_json, stats_json,
            chart_path, csv_path, created_at,
            ema_fast, ema_slow, use_clean,
            initial_balance, position_mode, trade_size,
            commission_pct, slippage_pct, stop_loss_pct, take_profit_pct, allow_short
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        run_id, strategy, exchange, symbol, timeframe,
        start_ts, end_ts,
        json.dumps(params),
        json.dumps(stats),
        chart_path,
        csv_path,
        created_at,
        ema_fast,
        ema_slow,
        int(use_clean),
        initial_balance,
        position_mode,
        trade_size,
        commission_pct,
        slippage_pct,
        stop_loss_pct,
        take_profit_pct,
        int(allow_short),
    ))

    conn.commit()
    conn.close()

    return redirect(url_for("results", run_id=run_id))

@app.route("/results/<run_id>")
def results(run_id):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT * FROM backtest_runs WHERE run_id = ?", (run_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return "Run not found", 404

    stats = json.loads(row["stats_json"])
    params = json.loads(row["params_json"])

    run = dict(row)
    run_date_window = _run_date_window_text(run, params)
    no_trades_message = _build_no_trades_message(run_date_window) if not run.get("csv_path") else None
    data_issue_message = _extract_data_issue_message(stats)
    if data_issue_message:
        no_trades_message = data_issue_message
    report_rel_path = f"backtests/{run['strategy']}/{run['run_id']}/quantstats_report.html"
    report_abs_path = os.path.join(current_app.static_folder, report_rel_path)
    report_path = report_rel_path if os.path.exists(report_abs_path) else None

    exit_notes = {
        "ema_cross": "Exit: EMA cross in opposite direction (if TP/SL = 0.00).",
        "rsi_reversion": "Exit: RSI above exit level (if TP/SL = 0.00).",
        "donchian_breakout": "Exit: Close below Donchian low (if TP/SL = 0.00).",
        "ema_trend_hold": "Exit: Close below Trend EMA (if TP/SL = 0.00).",
        "bmsb": "Exit: Close below BMSB (if TP/SL = 0.00).",
        "emalyarovich_smas": "Exit: Close below SMA Fast (if TP/SL = 0.00).",
        "k_davey_mom_keltner": "Exit: Keltner Stochastic crosses threshold (if TP/SL = 0.00).",
    }

    trades_charts, trades_chart_note, trades_chart_error = _build_trades_chart_for_results(run, params)

    trade_selector_options = []
    trade_selector_error = None
    trades_df, trades_error = _load_trades_for_run(run)
    if trades_error:
        trade_selector_error = trades_error
    else:
        trade_selector_options = _build_trade_selector_options(trades_df)

    if no_trades_message:
        if trades_chart_error == "Trades CSV not available for this run.":
            trades_chart_error = no_trades_message
        if trade_selector_error == "Trades CSV not available for this run.":
            trade_selector_error = no_trades_message

    return render_template(
        "results.html",
        run=run,
        stats=stats,
        params=params,
        report_path=report_path,
        exit_note=exit_notes.get(run["strategy"]),
        trades_charts=trades_charts,
        trades_chart_note=trades_chart_note,
        trades_chart_error=trades_chart_error,
        trade_selector_options=trade_selector_options,
        trade_selector_error=trade_selector_error,
        trade_viewer_indicator_toggles=_build_indicator_toggle_options(run["strategy"], params),
        trade_viewer_max_candles=TRADE_VIEWER_MAX_CANDLES_NO_CONFIRM,
        run_date_window=run_date_window,
        no_trades_message=no_trades_message,
    )


@app.route("/api/runs/<run_id>/params")
def run_params(run_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT run_id, strategy, symbol, timeframe, params_json FROM backtest_runs WHERE run_id = ?",
        (run_id,),
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        return jsonify({"ok": False, "error": "Run not found."}), 404

    try:
        params = json.loads(row["params_json"] or "{}")
    except Exception:
        params = {}

    return jsonify(
        {
            "ok": True,
            "run_id": row["run_id"],
            "strategy": row["strategy"],
            "symbol": row["symbol"],
            "timeframe": row["timeframe"],
            "params": params,
        }
    )


@app.route("/results/<run_id>/trade-viewer-chart", methods=["POST"])
def trade_viewer_chart(run_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM backtest_runs WHERE run_id = ?", (run_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return jsonify({"ok": False, "error": "Run not found."}), 404

    run = dict(row)
    params = json.loads(run["params_json"])

    payload = request.get_json(silent=True) or {}
    trade_number = int(payload.get("trade_number") or 0)
    force_generate = bool(payload.get("force_generate", False))
    export_packet = bool(payload.get("export_packet", False))
    selected_indicators_raw = payload.get("selected_indicators")
    selected_indicators = None
    if isinstance(selected_indicators_raw, list):
        selected_indicators = {
            str(item).strip() for item in selected_indicators_raw if str(item).strip()
        }

    if trade_number <= 0:
        return jsonify({"ok": False, "error": "Invalid trade number."}), 400

    trades_df, trades_error = _load_trades_for_run(run)
    if trades_error:
        return jsonify({"ok": False, "error": trades_error}), 400

    selected = trades_df[trades_df["trade_number"] == trade_number]
    if selected.empty:
        return jsonify({"ok": False, "error": f"Trade #{trade_number} not found."}), 404
    trade_row = selected.iloc[0]

    ohlc_df = _load_ohlc_for_trade_viewer(run, params, trades_df)
    if ohlc_df is None or ohlc_df.empty:
        return jsonify({"ok": False, "error": "OHLC data is unavailable for this run."}), 400

    ohlc_df = ohlc_df.copy()
    ohlc_df["timestamp"] = pd.to_datetime(ohlc_df["timestamp"], errors="coerce")
    ohlc_df = ohlc_df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if ohlc_df.empty:
        return jsonify({"ok": False, "error": "OHLC data has no valid timestamps."}), 400

    time_index = pd.DatetimeIndex(ohlc_df["timestamp"])
    entry_time = pd.to_datetime(trade_row["entry_time"], errors="coerce")
    exit_time = pd.to_datetime(trade_row["exit_time"], errors="coerce")
    if pd.isna(entry_time) or pd.isna(exit_time):
        return jsonify({"ok": False, "error": "Selected trade has invalid timestamps."}), 400

    entry_idx = _closest_index_position(time_index, entry_time)
    exit_idx = _closest_index_position(time_index, exit_time)
    if entry_idx is None or exit_idx is None:
        return jsonify({"ok": False, "error": "Unable to map trade timestamps to OHLC bars."}), 400

    if entry_idx > exit_idx:
        entry_idx, exit_idx = exit_idx, entry_idx

    start_idx = max(0, entry_idx - TRADE_VIEWER_CONTEXT_BEFORE)
    end_idx = min(len(ohlc_df) - 1, exit_idx + TRADE_VIEWER_CONTEXT_AFTER)
    candles_count = int(end_idx - start_idx + 1)

    if candles_count > TRADE_VIEWER_MAX_CANDLES_NO_CONFIRM and not force_generate:
        return jsonify(
            {
                "ok": True,
                "warning_required": True,
                "candles_count": candles_count,
                "max_candles_without_confirm": TRADE_VIEWER_MAX_CANDLES_NO_CONFIRM,
                "message": (
                    f"Trade #{trade_number} requires {candles_count} candles, which exceeds "
                    f"{TRADE_VIEWER_MAX_CANDLES_NO_CONFIRM}. Confirm to generate chart."
                ),
            }
        )

    start_ts = ohlc_df.iloc[start_idx]["timestamp"]
    end_ts = ohlc_df.iloc[end_idx]["timestamp"]
    single_trade_df = pd.DataFrame([trade_row])

    all_indicators = _build_indicator_overlays(ohlc_df, run["strategy"], params)
    if selected_indicators is None:
        indicators = all_indicators
    elif not selected_indicators:
        indicators = None
    else:
        indicators = {
            key: value
            for key, value in (all_indicators or {}).items()
            if key in selected_indicators
        } or None

    charts_rel_dir = f"backtests/{run['strategy']}/{run['run_id']}/trade_viewer"
    charts_abs_dir = os.path.join(current_app.static_folder, charts_rel_dir)
    os.makedirs(charts_abs_dir, exist_ok=True)

    file_name = f"trade_{trade_number:03d}.png"
    file_abs = os.path.join(charts_abs_dir, file_name)

    annotation_mode, chart_created = plot_trades_candlestick(
        df=ohlc_df,
        trades=single_trade_df,
        indicators=indicators,
        start_date=start_ts,
        end_date=end_ts,
        title=(
            f"{run['strategy']} {run['symbol']} {run['timeframe']}\n"
            f"Trade #{trade_number:03d} | 5 candles before entry and 5 after exit"
        ),
        output_path=file_abs,
        figsize=(18, 7),
    )

    if not chart_created:
        if annotation_mode == "missing_ohlc":
            return jsonify({"ok": False, "error": "Missing OHLC data for selected trade window."}), 400
        return jsonify({"ok": False, "error": "Could not generate trade chart."}), 400

    entry_trigger = str(trade_row.get("entry_trigger") or "n/a")
    exit_trigger = str(trade_row.get("exit_trigger") or "n/a")
    side = str(trade_row.get("side") or "n/a")
    net_pnl = _safe_float(trade_row.get("net_pnl", 0.0), default=0.0)
    chart_rel_path = f"{charts_rel_dir}/{file_name}"
    conditions_note = (
        "Condition labels are best-effort from trigger fields."
        if annotation_mode == "best_effort"
        else "Condition labels rendered from trigger fields."
    )

    used_indicators = list(indicators.keys()) if indicators else []
    packet_rel_path = None
    if export_packet:
        packet_name = f"trade_{trade_number:03d}.txt"
        packet_rel_path = f"{charts_rel_dir}/{packet_name}"
        packet_abs_path = os.path.join(charts_abs_dir, packet_name)
        _write_trade_packet_txt(
            packet_abs_path,
            {
                "run_id": run["run_id"],
                "strategy": run["strategy"],
                "symbol": run["symbol"],
                "timeframe": run["timeframe"],
                "trade_number": trade_number,
                "side": side,
                "entry_time": _format_trade_viewer_time(trade_row.get("entry_time")),
                "entry_price": _safe_float(trade_row.get("entry_price", 0.0), default=0.0),
                "exit_time": _format_trade_viewer_time(trade_row.get("exit_time")),
                "exit_price": _safe_float(trade_row.get("exit_price", 0.0), default=0.0),
                "net_pnl": net_pnl,
                "entry_trigger": entry_trigger,
                "exit_trigger": exit_trigger,
                "candles_count": candles_count,
                "annotation_mode": annotation_mode,
                "conditions_note": conditions_note,
                "selected_indicators": used_indicators,
            },
        )

    return jsonify(
        {
            "ok": True,
            "warning_required": False,
            "candles_count": candles_count,
            "chart_path": chart_rel_path,
            "annotation_mode": annotation_mode,
            "trade": {
                "trade_number": trade_number,
                "side": side,
                "entry_time": _format_trade_viewer_time(trade_row.get("entry_time")),
                "exit_time": _format_trade_viewer_time(trade_row.get("exit_time")),
                "entry_price": _safe_float(trade_row.get("entry_price", 0.0), default=0.0),
                "exit_price": _safe_float(trade_row.get("exit_price", 0.0), default=0.0),
                "entry_trigger": entry_trigger,
                "exit_trigger": exit_trigger,
                "net_pnl": net_pnl,
            },
            "conditions_note": conditions_note,
            "selected_indicators": used_indicators,
            "packet": (
                {
                    "png_path": chart_rel_path,
                    "txt_path": packet_rel_path,
                }
                if export_packet
                else None
            ),
        }
    )


@app.route("/history")
def history():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT run_id, strategy, exchange, symbol, timeframe, created_at
        FROM backtest_runs
        ORDER BY id DESC
        LIMIT 50
    """)
    runs = cur.fetchall()

    conn.close()
    return render_template("history.html", runs=runs)

@app.route("/charts/<strategy>")
def view_charts(strategy):
    charts_dir = os.path.join(current_app.static_folder,"charts",strategy)
    print(f"Looking for charts in: {charts_dir}")  # Debugging line
    if not os.path.exists(charts_dir):
        return f"No charts found for strategy {strategy} - Looking for charts in: {charts_dir}"

    images = [
        f"charts/{strategy}/{file}"
        for file in os.listdir(charts_dir)
        if file.endswith(".png")
    ]

    images.sort(reverse=True)


    return render_template("charts.html", strategy=strategy, images=images)

if __name__ == "__main__":
    app.run(debug=True)
