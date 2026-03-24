import json
import os
import uuid
import pandas as pd

from datetime import datetime, timezone
from flask import Flask, render_template, request, redirect, url_for, jsonify, current_app
from src.core.database import get_connection, init_db
from src.core.data import fetch_ohlcv
from src.core.plotting.plot_trades import plot_trades_candlestick_windows
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


def _build_trades_chart_for_results(run, params):
    csv_rel_path = run.get("csv_path")
    if not csv_rel_path:
        return None, None, "Trades CSV not available for this run."

    csv_abs_path = os.path.join(current_app.static_folder, csv_rel_path)
    if not os.path.exists(csv_abs_path):
        return None, None, "Trades CSV file was not found on disk."

    trades_df = pd.read_csv(csv_abs_path)
    if trades_df.empty:
        return None, None, "No trades available to render the trades chart."

    for col in ["entry_time", "exit_time"]:
        if col in trades_df.columns:
            trades_df[col] = pd.to_datetime(trades_df[col], utc=True, errors="coerce").dt.tz_convert(None)

    if "entry_time" not in trades_df.columns or "exit_time" not in trades_df.columns:
        return None, None, "Trades CSV is missing entry/exit timestamps."

    trades_df = trades_df.dropna(subset=["entry_time", "exit_time"]).copy()
    if trades_df.empty:
        return None, None, "Trades CSV has no valid timestamps to draw chart markers."

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

    params = {
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
        None, None,
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
