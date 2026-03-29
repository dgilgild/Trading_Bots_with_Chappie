from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from src.core.analysis.trade_explainability import (
    generate_trade_explainability,
)
from src.core.analysis.walk_forward import SUPPORTED_WALK_FORWARD_STRATEGIES


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic per-trade explainability artifacts",
    )
    parser.add_argument(
        "--strategy",
        default="ema_cross",
        help=f"Single strategy or 'all'. Supported: {', '.join(SUPPORTED_WALK_FORWARD_STRATEGIES)}",
    )
    parser.add_argument("--rows", type=int, default=430)
    parser.add_argument("--timeframe", default="1d", help="Display timeframe label in chart x-axis")
    parser.add_argument("--context-bars", type=int, default=30)
    parser.add_argument("--max-trades", type=int, default=50)
    parser.add_argument(
        "--high-contrast",
        action="store_true",
        help="Enable high-contrast plotting mode for explainability charts",
    )
    parser.add_argument(
        "--csv-baseline-dir",
        default=None,
        help="Optional directory containing deterministic baseline CSVs named <strategy>.csv",
    )
    return parser


def _validate_strategy(strategy: str) -> None:
    if strategy == "all":
        return
    if strategy not in SUPPORTED_WALK_FORWARD_STRATEGIES:
        supported = ", ".join(SUPPORTED_WALK_FORWARD_STRATEGIES)
        raise ValueError(f"Unsupported strategy='{strategy}'. Supported: {supported} and 'all'")


def main() -> int:
    args = _parser().parse_args()
    _validate_strategy(args.strategy)

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_output_dir = Path("docs") / "test_reports" / "trade_explainability" / run_stamp

    csv_baseline_dir = Path(args.csv_baseline_dir) if args.csv_baseline_dir else None
    result = generate_trade_explainability(
        strategy=args.strategy,
        rows=int(args.rows),
        timeframe_label=str(args.timeframe),
        context_bars=int(args.context_bars),
        max_trades=int(args.max_trades),
        high_contrast=bool(args.high_contrast),
        run_output_dir=run_output_dir,
        csv_baseline_dir=csv_baseline_dir,
    )

    print(f"run_stamp={run_stamp}")
    print(f"run_output_dir={result['run_output_dir']}")
    for strategy_result in result["strategies"]:
        print(
            "strategy={strategy} source={source_mode} requested={requested_trades} generated={generated_trades} "
            "fallback={fallback_trades} output={output_dir}".format(**strategy_result)
        )
        print(f"  csv_index={strategy_result['csv_index']}")
        print(f"  json_index={strategy_result['json_index']}")
    print(f"summary_report={result['summary_report']}")
    print(f"summary_json={result['summary_json']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
