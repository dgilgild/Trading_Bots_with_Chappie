import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

from src.core.data import fetch_ohlcv
from src.core.backtester import Backtester
from src.strategies.ema_cross.strategy import check_signal


def run_backtest_ema_cross(exchange, symbol, timeframe, start_date, end_date, ema_fast, ema_slow):
    # 1) Cargar histórico desde SQLite (fetch_ohlcv debe usar DB local)
    df = fetch_ohlcv(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        limit=50000
    )


    # 2) Inicializar backtester
    bt = Backtester(initial_capital=1000)

    df["ema_fast"] = df["close"].ewm(span=ema_fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=ema_slow, adjust=False).mean()

    cross_up = (df["ema_fast"].shift(1) < df["ema_slow"].shift(1)) & (df["ema_fast"] > df["ema_slow"])
    cross_down = (df["ema_fast"].shift(1) > df["ema_slow"].shift(1)) & (df["ema_fast"] < df["ema_slow"])

    print("Cross UP:", cross_up.sum())
    print("Cross DOWN:", cross_down.sum())
    print(df.head())
    print(df.tail())
    print("Rows:", len(df))
    print(df.dtypes)

    # 3) Recorrer velas
    for i in range(ema_slow + 1, len(df)):
        #slice_df = df.iloc[:i]
        slice_df = df.iloc[:i].copy()

        signal = check_signal(slice_df, ema_fast, ema_slow)

        price = slice_df.iloc[-1]["close"]
        timestamp = slice_df.iloc[-1]["timestamp"]

        bt.on_signal(signal, price, timestamp)

    # 4) Stats finales
    stats = bt.stats()

    clean_stats = {}
    for k, v in stats.items():
        if hasattr(v, "item"):   # numpy types
            clean_stats[k] = v.item()
        else:
            clean_stats[k] = v

    print("TRADES:", bt.trades)
    print("TOTAL TRADES:", len(bt.trades))

    # 5) Guardar gráfico en web/static/charts/ema_cross/
    # ===== GENERAR EQUITY CURVE PNG =====

    #equity = [1000]
    #equity = bt.equity_curve

    equity = [bt.initial_capital]
    for t in bt.trades:
        equity.append(equity[-1] + t["net_pnl"])


    output_dir = "web/static/charts/ema_cross"
    os.makedirs(output_dir, exist_ok=True)

    filename = f"ema_cross_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
    file_path = os.path.join(output_dir, filename)

    print("TRADES:", len(bt.trades))
    print("EQUITY POINTS:", len(bt.equity_curve))

    plt.figure(figsize=(10, 5))
    plt.plot(equity)
    plt.title("Equity Curve - EMA Cross")
    plt.xlabel("Trade #")
    plt.ylabel("Equity ($)")
    plt.grid(True)
    plt.savefig(file_path)
    plt.close()

    web_path = f"/static/charts/ema_cross/{filename}"

    # Guardamos trades en JSON por ahora (después hacemos PNG con matplotlib)
    #with open(file_path, "w") as f:
    #    json.dump(bt.trades, f, indent=4, default=str)

    return clean_stats, web_path


if __name__ == "__main__":
    # test manual local
    stats, chart_path = run_backtest_ema_cross(
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="15m",
        start_date="2018-01-01",
        end_date=None,
        ema_fast=20,
        ema_slow=50
    )

    print(stats)
    print("Chart path:", chart_path)
