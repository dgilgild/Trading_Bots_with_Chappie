import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from core.data import fetch_ohlcv
from core.backtester import Backtester
from strategies.ema_cross.strategy import check_signal
from strategies.ema_cross import config
from scripts.plot_trade_results import plot_trade_net_profits


# Cargar histórico
df = fetch_ohlcv(
    config.SYMBOL,
    config.TIMEFRAME,
    limit=2000
)

bt = Backtester(initial_capital=1000)

for i in range(config.EMA_SLOW + 2, len(df)):
    slice_df = df.iloc[:i]

    signal = check_signal(
        slice_df,
        config.EMA_FAST,
        config.EMA_SLOW
    )

    price = slice_df.iloc[-1]["close"]
    timestamp = slice_df.iloc[-1]["timestamp"]

    if signal is not None:
        bt.on_signal(signal, price, timestamp)

print("\nBacktest finished")
print(f"Total trades: {len(bt.trades)}")

print("Backtester instance:", bt)

stats = bt.stats()

print("\n====== BACKTEST RESULTS ======")
for k, v in stats.items():
    print(f"{k}: {v}")

# ---- plot trades ----
strategy_name = "ema_cross"

plot_trade_net_profits(
    trades=bt.trades,
    strategy_name=strategy_name
)
