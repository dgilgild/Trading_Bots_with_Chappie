import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from flask import current_app
from src.core.data import fetch_ohlcv
from src.core.backtester_v2 import BacktesterV2
from src.strategies.ema_cross.strategy import check_signal
from src.visualization.plot_trades import plot_trades_by_date
from src.core.plotting.plot_trades import plot_trades


def export_trades_csv(trades, output_dir, run_id, prefix="ema_cross_v2"):
    if not trades:
        print("[WARN] No trades to export")
        return None, None

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
            "position_size",
            "gross_pnl",
            "commission_paid",
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
    initial_balance=1000,
    position_mode="all_in",
    trade_size=100,
    commission_pct=0.001,
    slippage_pct=0.01,
    allow_short=False,
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

    df["ema_fast"] = df["close"].ewm(span=ema_fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=ema_slow, adjust=False).mean()

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
        signal, trigger = check_signal(slice_df, ema_fast, ema_slow)

        # 3️⃣ Ejecutamos señal
        bt.on_signal(signal, price, timestamp, trigger, i)

   
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

    # --------------------------------------------------
    # 6) Plot Trades
    # --------------------------------------------------
    plot_trades_by_date(
        df=df,
        trades=bt.trades,
        start_date=start_date,
        end_date=end_date,
        title="EMA Cross V2"
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
        run_id
    )

    return clean_stats, DB_equity_path, DB_csv_path