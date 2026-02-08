from src.data.downloader import BinanceDownloader

downloader = BinanceDownloader()

downloader.download(
    symbol="BTC/USDT",
    timeframe="15m",
    start_date="2018-01-01"
)
