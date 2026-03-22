import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from src.core.data import fetch_ohlcv
from src.core.backtester_v2 import BacktesterV2
from src.core.reporting import generate_quantstats_report
from src.strategies.emalyarovich_smas.strategy import compute_smas, check_signal


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

    DB_csv_path = f"backtests/emalyarovich_smas/{run_id}/{filename}"

    return path, DB_csv_path


def run_backtest_emalyarovich_smas_v2(
    exchange,
    symbol,
    timeframe,
    start_date,
    end_date,
    sma_fast=20,
    sma_slow=200,
    slope_bars=3,
    use_clean=True,
    run_id=None,
    initial_balance=1000.0,
    position_mode="all_in",
    trade_size=100.0,
    commission_pct=0.001,
    slippage_pct=0.001,
    allow_short=False,
    stop_loss_pct=0.02,
    take_profit_pct=0.03,
    generate_report=True,
    pyramiding=1,
    position_pct=None,
    generate_plots=True,
    generate_equity=True,
    base_path=None,
):

    output_dir = os.path.join(
        base_path,
        "static",
        "backtests",
        "emalyarovich_smas",
        run_id,
    )

    os.makedirs(output_dir, exist_ok=True)

    print("E.MALYAROVICH SMAS RUN ID:", run_id)
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
        print("[WARN] No data returned for SMAs backtest")
        return {}, None, None

    df = compute_smas(df, sma_fast, sma_slow)

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

    for i in range(max(sma_fast, sma_slow, slope_bars) + 1, len(df)):
        current_bar = df.iloc[i]
        price = current_bar["close"]
        high = current_bar["high"]
        low = current_bar["low"]
        timestamp = current_bar["timestamp"]

        bt.on_bar(high=high, low=low, timestamp=timestamp, bar_index=i)

        current_side = bt.position["side"] if bt.position else None
        signal, trigger = check_signal(
            df.iloc[:i + 1],
            sma_fast,
            sma_slow,
            slope_bars,
            current_side=current_side,
        )

        bt.on_signal(signal, price, timestamp, trigger, i)

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
        plt.title("Equity Curve - E.Malyarovich SMAs")
        plt.xlabel("Date")
        plt.ylabel("Equity ($)")
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(equity_path)
        plt.close()

        DB_equity_path = f"backtests/emalyarovich_smas/{run_id}/{equity_filename}"

    if generate_report:
        generate_quantstats_report(
            equity_dates,
            equity,
            output_dir,
            title=f"E.Malyarovich SMAs {symbol} {timeframe}",
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
            "ema_fast": int(sma_fast),
            "ema_slow": int(sma_slow),
            "use_clean": use_clean,
            "initial_balance": initial_balance,
            "position_mode": position_mode,
            "trade_size": trade_size,
            "commission_pct": commission_pct,
            "slippage_pct": slippage_pct,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "allow_short": allow_short,
            "slope_bars": slope_bars,
        },
    )

    return clean_stats, DB_equity_path, DB_csv_path
