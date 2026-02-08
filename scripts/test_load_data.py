from src.core.data_loader import load_ohlcv

df = load_ohlcv(
    exchange="binance",
    symbol="BTC/USDT",
    timeframe="15m"
)

print(df.head())
print(df.tail())
print("Rows:", len(df))
