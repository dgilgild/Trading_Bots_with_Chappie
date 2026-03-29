from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.core.analysis.walk_forward import (
    SUPPORTED_WALK_FORWARD_STRATEGIES,
    load_walk_forward_data,
    resolve_window_lengths,
    run_walk_forward,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run walk-forward evaluation (ST-402)")

    parser.add_argument("--mode", choices=["deterministic", "db"], required=True)
    parser.add_argument("--strategy", default="ema_cross", choices=SUPPORTED_WALK_FORWARD_STRATEGIES)

    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--rows", type=int, default=420, help="deterministic mode only")

    parser.add_argument("--train-pct", type=float, default=None)
    parser.add_argument("--test-pct", type=float, default=None)
    parser.add_argument("--step-pct", type=float, default=None)

    parser.add_argument("--train-bars", type=int, default=None)
    parser.add_argument("--test-bars", type=int, default=None)
    parser.add_argument("--step-bars", type=int, default=None)

    parser.add_argument("--ema-fast", type=int, default=20)
    parser.add_argument("--ema-slow", type=int, default=50)
    parser.add_argument("--initial-balance", type=float, default=1000.0)

    return parser


def _output_dir(run_stamp: str) -> Path:
    return Path("docs") / "test_reports" / "walk_forward" / run_stamp


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    data_df = load_walk_forward_data(
        mode=args.mode,
        exchange=args.exchange,
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
        use_clean=True,
        deterministic_rows=args.rows,
    )

    window_mode, train_len, test_len, step_len = resolve_window_lengths(
        total_bars=len(data_df),
        train_pct=args.train_pct,
        test_pct=args.test_pct,
        step_pct=args.step_pct,
        train_bars=args.train_bars,
        test_bars=args.test_bars,
        step_bars=args.step_bars,
    )

    windows_df, summary = run_walk_forward(
        strategy=args.strategy,
        data_mode=args.mode,
        data_df=data_df,
        train_len=train_len,
        test_len=test_len,
        step_len=step_len,
        window_mode=window_mode,
        exchange=args.exchange,
        symbol=args.symbol,
        timeframe=args.timeframe,
        ema_fast=args.ema_fast,
        ema_slow=args.ema_slow,
        initial_balance=args.initial_balance,
    )

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out_dir = _output_dir(run_stamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    windows_csv = out_dir / "walk_forward_windows.csv"
    summary_json = out_dir / "walk_forward_summary.json"

    windows_df.to_csv(windows_csv, index=False)

    payload = {
        "run_stamp": run_stamp,
        "mode": args.mode,
        "strategy": args.strategy,
        "exchange": args.exchange,
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "window_mode": window_mode,
        "train_len": train_len,
        "test_len": test_len,
        "step_len": step_len,
        "summary": summary,
    }

    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"run_stamp={run_stamp}")
    print(f"windows_csv={windows_csv}")
    print(f"summary_json={summary_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
