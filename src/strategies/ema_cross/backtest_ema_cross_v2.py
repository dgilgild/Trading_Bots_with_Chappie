import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from flask import current_app
from src.core.data import fetch_ohlcv
from src.core.backtester_v2 import BacktesterV2
from src.core.reporting import generate_quantstats_report
from src.core.ta import compute_adx, compute_atr
from src.strategies.ema_cross.strategy import check_signal
from src.visualization.plot_trades import plot_trades_by_date
from src.core.plotting.plot_trades import plot_trades


REQUIRED_OHLC_COLUMNS = ("timestamp", "open", "high", "low", "close")


def _validate_and_prepare_ohlc(df):
    if df is None or df.empty:
        return None, "No OHLCV data found for the selected filters."

    missing = [col for col in REQUIRED_OHLC_COLUMNS if col not in df.columns]
    if missing:
        missing_txt = ", ".join(sorted(missing))
        return None, f"OHLCV data is missing required column(s): {missing_txt}."

    prepared = df.copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="coerce")

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in prepared.columns:
            prepared[col] = pd.to_numeric(prepared[col], errors="coerce")

    prepared = prepared.dropna(subset=["timestamp", "high", "low", "close"])
    prepared = prepared.sort_values("timestamp").reset_index(drop=True)
    if prepared.empty:
        return None, "OHLCV rows are present but invalid after timestamp/price validation."

    return prepared, None


def export_trades_csv(trades, output_dir, run_id, metadata, prefix="ema_cross_v2"):
    if not trades:
        print("[WARN] No trades to export")
        return None, None

    df = pd.DataFrame(trades)

    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)

    df["pnl_pct"] = (
        (df["exit_price"] - df["entry_price"]) / df["entry_price"] * 100
    ).round(3)

    df["net_return_pct"] = (
        (df["net_pnl"] / df["position_size"]) * 100
    ).round(3)

    df["duration_minutes"] = (
        (df["exit_time"] - df["entry_time"]).dt.total_seconds() / 60
    ).round(2)

    df["result"] = df["net_pnl"].apply(lambda x: "WIN" if x > 0 else "LOSS")

    df["balance"] = df["cash_after_trade"].round(6)

    for key, value in metadata.items():
        df[key] = value

    filename = f"{run_id}_trades.csv"
    path = os.path.join(output_dir, filename)

    df = df[
        [
            "run_id",
            "exchange",
            "symbol",
            "timeframe",
            "ema_fast",
            "ema_slow",
            "side",
            "result",
            "position_size",
            "qty",
            "pyramid_level",
            "balance",
            "cash_after_trade",
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "stop_price",
            "take_profit_price",
            "commission_pct",
            "slippage_pct",
            "stop_loss_pct",
            "take_profit_pct",
            "gross_pnl",
            "commission_paid",
            "pnl_pct",
            "net_return_pct",
            "net_pnl",
            "entry_trigger",
            "exit_trigger",
            "bars_in_trade",
            "use_clean",
            "position_mode",
        ]
    ]

    df.to_csv(path, index=False)

    DB_csv_path = f"backtests/ema_cross/{run_id}/{filename}"

    return path, DB_csv_path


def run_backtest_ema_cross_v2(
    exchange,
    symbol,
    timeframe,
    start_date,
    end_date,
    ema_fast,
    ema_slow,
    use_clean=True,
    run_id=None,
    initial_balance=1000.0,
    position_mode="all_in",
    trade_size=100.0,
    commission_pct=0.001,
    slippage_pct=0.01,
    allow_short=True,
    stop_loss_pct=0.02,
    take_profit_pct=None,
    atr_period=None,
    atr_sl_mult=None,
    atr_tp_mult=None,
    adx_period=None,
    adx_threshold=None,
    generate_report=True,
    pyramiding=1,
    position_pct=None,
    generate_plots=True,
    generate_equity=True,
    base_path=None,
):

    # --------------------------------------------------
    # 0) Crear carpeta
    # --------------------------------------------------
    output_dir = os.path.join(
        base_path,
        "static",
        "backtests",
        "ema_cross",
        run_id
    )

    os.makedirs(output_dir, exist_ok=True)

    print("BACKTEST V2 RUN ID:", run_id)
    print("OUTPUT DIR:", output_dir)

    # --------------------------------------------------
    # 1) Data
    # --------------------------------------------------
    df = fetch_ohlcv(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        limit=50000,
        use_clean=use_clean,
    )

    df, data_error = _validate_and_prepare_ohlc(df)
    if data_error:
        print(f"[WARN] EMA Cross V2 skipped: {data_error}")
        return {
            "Status": "No data",
            "Total trades": 0,
            "No Data Reason": data_error,
        }, None, None

    df["ema_fast"] = df["close"].ewm(span=ema_fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=ema_slow, adjust=False).mean()

    atr_series = compute_atr(df, atr_period) if atr_period else None
    adx_series = compute_adx(df, adx_period) if adx_period else None

    # --------------------------------------------------
    # 2) Backtester V2
    # --------------------------------------------------
    bt = BacktesterV2(
        initial_capital=initial_balance,
        position_mode=position_mode,
        trade_size=trade_size,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
        allow_short=allow_short,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        atr_sl_mult=atr_sl_mult,
        atr_tp_mult=atr_tp_mult,
        pyramiding=pyramiding,
        position_pct=position_pct,
    )

    # --------------------------------------------------
    # 3) Loop vela a vela
    # --------------------------------------------------
    for i in range(ema_slow + 1, len(df)):

        slice_df = df.iloc[:i].copy()

        current_bar = df.iloc[i]

        high = current_bar["high"]
        low = current_bar["low"]
        price = current_bar["close"]
        timestamp = current_bar["timestamp"]

        # 1️⃣ Primero chequeamos stop intrabar
        bt.on_bar(
            high=high,
            low=low,
            timestamp=timestamp,
            bar_index=i
        )

        # 2️⃣ Después calculamos señal EMA
        current_side = bt.position["side"] if bt.position else None
        signal, trigger = check_signal(slice_df, ema_fast, ema_slow, current_side=current_side)

        # 3️⃣ Ejecutamos señal
        atr_value = None
        if atr_series is not None:
            atr_raw = atr_series.iloc[i]
            if pd.notna(atr_raw):
                atr_value = float(atr_raw)

        if signal in {"LONG", "SHORT"}:
            if atr_series is not None and atr_value is None:
                continue

            if adx_series is not None and adx_threshold is not None:
                adx_raw = adx_series.iloc[i]
                if pd.isna(adx_raw) or float(adx_raw) < float(adx_threshold):
                    continue

        bt.on_signal(signal, price, timestamp, trigger, i, atr_value=atr_value)

   
    # --------------------------------------------------
    # 4) Stats
    # --------------------------------------------------
    stats = bt.stats()
    clean_stats = {
        k: v.item() if hasattr(v, "item") else v
        for k, v in stats.items()
    }

    # --------------------------------------------------
    # 5) Equity Curve (compatible)
    # --------------------------------------------------
    equity = []
    equity_dates = []
    current_equity = bt.initial_capital

    for t in bt.trades:
        current_equity += t["net_pnl"]
        equity.append(current_equity)
        equity_dates.append(t["exit_time"])

    DB_equity_path = None
    if generate_equity and bt.trades:
        equity_filename = f"equity_curve_{run_id}.png"
        equity_path = os.path.join(output_dir, equity_filename)

        plt.figure(figsize=(10, 5))
        plt.plot(equity_dates, equity)
        plt.title("Equity Curve - EMA Cross V2")
        plt.xlabel("Date")
        plt.ylabel("Equity ($)")
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(equity_path)
        plt.close()

        DB_equity_path = f"backtests/ema_cross/{run_id}/{equity_filename}"
    elif generate_equity:
        print("[INFO] No closed trades; skipping equity curve chart generation.")

    if generate_report:
        generate_quantstats_report(
            equity_dates,
            equity,
            output_dir,
            title=f"EMA Cross {symbol} {timeframe}",
        )

    # --------------------------------------------------
    # 6) Plot Trades
    # --------------------------------------------------
    trades_chart_path = None
    if generate_plots:
        plot_trades_by_date(
            df=df,
            trades=bt.trades,
            start_date=start_date,
            end_date=end_date,
            title="EMA Cross V2",
            show_plot=False,
        )

        indicators = {
            f"EMA {ema_fast}": df["ema_fast"],
            f"EMA {ema_slow}": df["ema_slow"],
        }

        trades_chart_path = plot_trades(
            df=df,
            trades=bt.trades,
            indicators=indicators,
            start_date=start_date,
            end_date=end_date,
            title="EMA Cross – Trades V2"
        )

    # --------------------------------------------------
    # 7) CSV
    # --------------------------------------------------
    csv_path, DB_csv_path = export_trades_csv(
        bt.trades,
        output_dir,
        run_id,
        metadata={
            "run_id": run_id,
            "exchange": exchange,
            "symbol": symbol,
            "timeframe": timeframe,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "use_clean": use_clean,
            "initial_balance": initial_balance,
            "position_mode": position_mode,
            "trade_size": trade_size,
            "commission_pct": commission_pct,
            "slippage_pct": slippage_pct,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "allow_short": allow_short,
        },
    )

    return clean_stats, DB_equity_path, DB_csv_path
