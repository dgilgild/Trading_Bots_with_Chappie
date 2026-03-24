import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from src.core.data import fetch_ohlcv
from src.core.backtester_v2 import BacktesterV2
from src.core.reporting import generate_quantstats_report
from src.strategies.bmsb.strategy import compute_bmsb, compute_tensignal


def export_trades_csv(trades, output_dir, run_id, metadata):
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

    DB_csv_path = f"backtests/bmsb/{run_id}/{filename}"

    return path, DB_csv_path


def run_backtest_bmsb_v2(
    exchange,
    symbol,
    timeframe,
    start_date,
    end_date,
    sma_period=20,
    ema_period=21,
    tensignal_window=3,
    trail_percent=0.05,
    use_clean=True,
    run_id=None,
    initial_balance=1000.0,
    position_mode="all_in",
    trade_size=100.0,
    commission_pct=0.001,
    slippage_pct=0.001,
    allow_short=False,
    stop_loss_pct=None,
    take_profit_pct=None,
    trading_start_date=None,
    generate_report=True,
    pyramiding=1,
    position_pct=None,
    use_tp_sl=True,
    generate_plots=True,
    generate_equity=True,
    base_path=None,
):

    output_dir = os.path.join(
        base_path,
        "static",
        "backtests",
        "bmsb",
        run_id,
    )

    os.makedirs(output_dir, exist_ok=True)

    print("BMSB BACKTEST RUN ID:", run_id)
    print("OUTPUT DIR:", output_dir)

    df = fetch_ohlcv(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        limit=50000,
        use_clean=use_clean,
    )

    if df is None or df.empty:
        print("[WARN] No data returned for BMSB backtest")
        return {}, None, None

    df = compute_bmsb(df, sma_period, ema_period)
    df["tensignal"] = compute_tensignal(df, tensignal_window)

    if trading_start_date:
        trading_start_ts = pd.Timestamp(trading_start_date, tz="UTC")
    else:
        trading_start_ts = None

    bt = BacktesterV2(
        initial_capital=initial_balance,
        position_mode=position_mode,
        trade_size=trade_size,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
        allow_short=allow_short,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        pyramiding=pyramiding,
        position_pct=position_pct,
    )

    for i in range(max(sma_period, ema_period, tensignal_window) + 1, len(df)):
        current_bar = df.iloc[i]
        price = current_bar["close"]
        high = current_bar["high"]
        low = current_bar["low"]
        timestamp = current_bar["timestamp"]

        if trading_start_ts is not None and pd.Timestamp(timestamp, tz="UTC") < trading_start_ts:
            continue

        if use_tp_sl and bt.position is not None and bt.position["side"] == "LONG":
            trail_price = price * (1 - trail_percent)
            prev_stop = bt.position.get("stop_price")
            if prev_stop is None:
                bt.position["stop_price"] = trail_price
            else:
                bt.position["stop_price"] = max(trail_price, prev_stop)

        bt.on_bar(high=high, low=low, timestamp=timestamp, bar_index=i)

        current_side = bt.position["side"] if bt.position else None
        buysignal = price > current_bar["bmsb"]
        tensignal = current_bar["tensignal"]
        sellsignal = (
            df["close"].iloc[i - 1] >= df["bmsb"].iloc[i - 1]
            and price < current_bar["bmsb"]
        )

        if buysignal and tensignal >= 1:
            bt.on_signal("LONG", price, timestamp, "BMSB long", i)

        if bt.position is not None and bt.position["side"] == "LONG" and sellsignal:
            bt.on_signal("EXIT", price, timestamp, "BMSB crossunder", i)

    stats = bt.stats()
    clean_stats = {k: v.item() if hasattr(v, "item") else v for k, v in stats.items()}

    equity = []
    equity_dates = []
    current_equity = bt.initial_capital

    for t in bt.trades:
        current_equity += t["net_pnl"]
        equity.append(current_equity)
        equity_dates.append(t["exit_time"])

    DB_equity_path = None
    if generate_equity:
        equity_filename = f"equity_curve_{run_id}.png"
        equity_path = os.path.join(output_dir, equity_filename)

        plt.figure(figsize=(10, 5))
        plt.plot(equity_dates, equity)
        plt.title("Equity Curve - BMSB")
        plt.xlabel("Date")
        plt.ylabel("Equity ($)")
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(equity_path)
        plt.close()

        DB_equity_path = f"backtests/bmsb/{run_id}/{equity_filename}"

    if generate_report:
        generate_quantstats_report(
            equity_dates,
            equity,
            output_dir,
            title=f"BMSB {symbol} {timeframe}",
        )

    csv_path, DB_csv_path = export_trades_csv(
        bt.trades,
        output_dir,
        run_id,
        metadata={
            "run_id": run_id,
            "exchange": exchange,
            "symbol": symbol,
            "timeframe": timeframe,
            "ema_fast": int(sma_period),
            "ema_slow": int(ema_period),
            "use_clean": use_clean,
            "initial_balance": initial_balance,
            "position_mode": position_mode,
            "trade_size": trade_size,
            "commission_pct": commission_pct,
            "slippage_pct": slippage_pct,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "allow_short": allow_short,
            "bmsb_sma": sma_period,
            "bmsb_ema": ema_period,
            "bmsb_trail_percent": trail_percent,
            "bmsb_tensignal_window": tensignal_window,
        },
    )

    return clean_stats, DB_equity_path, DB_csv_path
