import argparse

from src.data.yfinance_downloader import YFinanceDownloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    args = parser.parse_args()

    downloader = YFinanceDownloader()

    mappings = [
        ("ES=F", "ES=F"),
        ("MES=F", "MES=F"),
    ]

    for ticker, symbol_label in mappings:
        count = downloader.download_daily(
            ticker=ticker,
            symbol_label=symbol_label,
            start_date=args.start,
            end_date=args.end,
        )
        print(f"{ticker}: inserted {count} rows")


if __name__ == "__main__":
    main()
