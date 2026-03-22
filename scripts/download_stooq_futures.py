import argparse

from src.data.stooq_downloader import StooqDownloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    args = parser.parse_args()

    downloader = StooqDownloader()

    mappings = [
        ("ES.F", "ES.F"),
        ("MES.F", "MES.F"),
    ]

    for symbol_code, symbol_label in mappings:
        count = downloader.download_daily(
            symbol_code=symbol_code,
            symbol_label=symbol_label,
            start_date=args.start,
            end_date=args.end,
        )
        print(f"{symbol_code}: inserted {count} rows")


if __name__ == "__main__":
    main()
