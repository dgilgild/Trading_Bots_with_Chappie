import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from datetime import datetime, timezone
from flask import current_app
from src.core.data import fetch_ohlcv
from src.core.backtester import Backtester
from src.strategies.ema_cross.strategy import check_signal
from src.visualization.plot_trades import plot_trades_by_date
from src.core.plotting.plot_trades import plot_trades



def export_trades_csv(trades, output_dir, run_id, prefix="ema_cross"):
    if not trades:
        print("[WARN] No trades to export")
        return None

    df = pd.DataFrame(trades)

    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)

    df["pnl_pct"] = (
        (df["exit_price"] - df["entry_price"]) / df["entry_price"] * 100
    ).round(3)

    df["result"] = df["net_pnl"].apply(lambda x: "WIN" if x > 0 else "LOSS")

    filename = f"{run_id}_trades.csv"
    path = os.path.join(output_dir, filename)

    df = df[
        [
            "entry_time",
            "exit_time",
            "side",
            "entry_price",
            "exit_price",
            "pnl_pct",
            "net_pnl",
            "result",
            "entry_trigger",
            "exit_trigger",
            "bars_in_trade",
        ]
    ]

    df.to_csv(path, index=False)

    DB_csv_path = f"backtests/ema_cross/{run_id}/{filename}"

    return path, DB_csv_path


def run_backtest_ema_cross(
    exchange,
    symbol,
    timeframe,
    start_date,
    end_date,
    ema_fast,
    ema_slow,
    use_clean=True,
    run_id=None
):

    # 0)  Crear carpeta del run
    output_dir = os.path.join(
        current_app.root_path,
        "static",
        "backtests",
        "ema_cross",
        run_id
    )

    os.makedirs(output_dir, exist_ok=True)
    print("0.- BACKTEST RUN ID:", run_id)
    print("0.- OUTPUT DIR:", output_dir)


    # 1) Data
    df = fetch_ohlcv(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        limit=50000,
        use_clean=use_clean,
    )

    # 2) Backtester
    bt = Backtester(initial_capital=1000)

    df["ema_fast"] = df["close"].ewm(span=ema_fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=ema_slow, adjust=False).mean()

    # 3) Loop vela a vela
    for i in range(ema_slow + 1, len(df)):
        slice_df = df.iloc[:i].copy()

        signal, trigger = check_signal(slice_df, ema_fast, ema_slow)

        price = slice_df.iloc[-1]["close"]
        timestamp = slice_df.iloc[-1]["timestamp"]

        bt.on_signal(signal, price, timestamp, trigger,i)

    # 4) Stats
    stats = bt.stats()
    clean_stats = {
        k: v.item() if hasattr(v, "item") else v
        for k, v in stats.items()
    }

    # 5) Equity curve (date based)

    equity = []
    equity_dates = []
    current_equity = bt.initial_capital

    for t in bt.trades:
        current_equity += t["net_pnl"]
        equity.append(current_equity)
        equity_dates.append(t["exit_time"])

    equity_filename = f"equity_curve_{run_id}.png"
    equity_path = os.path.join(output_dir, equity_filename)

    print("5.- equity_filename:", equity_filename)
    print("5.- equity_path:", equity_path)

    plt.figure(figsize=(10, 5))
    plt.plot(equity_dates, equity)
    plt.title("Equity Curve - EMA Cross")
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

    #web_path = f"/static/charts/ema_cross/{filename}"

    # 6) Plot trades x fecha
    #from src.visualization.plot_trades import plot_trades_by_date

    plot_trades_by_date(
        df=df,
        trades=bt.trades,
        start_date="2018-01-01",
        end_date="2018-02-01",
        title="EMA Cross – BTC/USDT"
    )
 
    # 7) 
    indicators = {
        f"EMA {ema_fast}": df["close"].ewm(span=ema_fast, adjust=False).mean(),
        f"EMA {ema_slow}": df["close"].ewm(span=ema_slow, adjust=False).mean(),
    }

    trades_chart_path = plot_trades(
        df=df,
        trades=bt.trades,
        indicators=indicators,
        start_date=start_date,
        end_date=end_date,
        title="EMA Cross – Trades"
    )


    # 8) CSV trades

    print("8.- TOTAL TRADES:", len(bt.trades))
    if bt.trades:
        print("8.- TRADE KEYS:", bt.trades[0].keys())

    df = pd.DataFrame(bt.trades)

    csv_path, DB_csv_path = export_trades_csv(bt.trades, output_dir, run_id)

    print("8.- CSV trades equity filename:", equity_filename)
    print("8.- CSV Path:", csv_path)

    return clean_stats, DB_equity_path, DB_csv_path



    if return_raw:
        return clean_stats, equity_chart_path, csv_path, df, bt.trades

    return clean_stats, equity_chart_path, csv_path


if __name__ == "__main__":
    stats, chart_path, csv_path = run_backtest_ema_cross(
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="15m",
        start_date="2018-01-01",
        end_date=None,
        ema_fast=20,
        ema_slow=50,
    )

    print(stats)
    print("Chart:", chart_path)
    print("CSV:", csv_path)
