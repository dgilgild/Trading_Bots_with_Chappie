import os
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import pandas as pd

from src.core.data import fetch_ohlcv
from src.core.backtester import Backtester
from src.strategies.ema_cross.strategy import check_signal
from src.visualization.plot_trades import plot_trades_by_date
from src.core.plotting.plot_trades import plot_trades



def export_trades_csv(trades, prefix="ema_cross"):
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

    export_dir = os.path.join("data", "reports", "trades")
    os.makedirs(export_dir, exist_ok=True)

    filename = f"{prefix}_trades_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    path = os.path.join(export_dir, filename)

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
    return path



def run_backtest_ema_cross(
    exchange,
    symbol,
    timeframe,
    start_date,
    end_date,
    ema_fast,
    ema_slow,
    use_clean=True,
):

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

    # 5) Equity curve
    equity = [bt.initial_capital]
    for t in bt.trades:
        equity.append(equity[-1] + t["net_pnl"])

    output_dir = "web/static/charts/ema_cross"
    os.makedirs(output_dir, exist_ok=True)

    filename = f"ema_cross_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.png"
    file_path = os.path.join(output_dir, filename)

    plt.figure(figsize=(10, 5))
    plt.plot(equity)
    plt.title("Equity Curve - EMA Cross")
    plt.xlabel("Trade #")
    plt.ylabel("Equity ($)")
    plt.grid(True)
    plt.savefig(file_path)
    plt.close()

    web_path = f"/static/charts/ema_cross/{filename}"

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

    print("TOTAL TRADES:", len(bt.trades))
    if bt.trades:
        print("TRADE KEYS:", bt.trades[0].keys())

    df = pd.DataFrame(bt.trades)
    print("DF COLUMNS:", df.columns.tolist())

    csv_path = export_trades_csv(bt.trades)

    return clean_stats, web_path, csv_path


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
