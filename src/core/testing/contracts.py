from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_TRADE_CSV_COLUMNS = (
    "entry_time",
    "exit_time",
    "entry_trigger",
    "exit_trigger",
    "net_pnl",
    "position_size",
    "bars_in_trade",
    "pyramid_level",
)

REQUIRED_TIMESTAMP_COLUMNS = ("entry_time", "exit_time")
REQUIRED_NUMERIC_COLUMNS = ("net_pnl", "position_size", "bars_in_trade", "pyramid_level")


def validate_trade_csv_contract(
    trades_df: pd.DataFrame,
    context: str = "trade CSV",
    validate_exit_time_monotonic: bool = True,
) -> None:
    missing = [col for col in REQUIRED_TRADE_CSV_COLUMNS if col not in trades_df.columns]
    if missing:
        raise ValueError(f"{context}: missing required columns: {', '.join(missing)}")

    parsed_timestamps: dict[str, pd.Series] = {}
    for col in REQUIRED_TIMESTAMP_COLUMNS:
        parsed = pd.to_datetime(trades_df[col], utc=True, errors="coerce")
        parsed_timestamps[col] = parsed
        invalid_count = int(parsed.isna().sum())
        if invalid_count:
            raise ValueError(
                f"{context}: column '{col}' must contain valid non-null timestamps "
                f"(invalid rows: {invalid_count})"
            )

    entry_time = parsed_timestamps["entry_time"]
    exit_time = parsed_timestamps["exit_time"]
    invalid_timeline = entry_time > exit_time
    if bool(invalid_timeline.any()):
        first_invalid_index = int(invalid_timeline[invalid_timeline].index[0])
        raise ValueError(
            f"{context}: entry_time must be <= exit_time for every row "
            f"(first invalid row index: {first_invalid_index})"
        )

    if validate_exit_time_monotonic and not exit_time.is_monotonic_increasing:
        previous_exit = exit_time.shift(1)
        non_monotonic = exit_time < previous_exit
        first_non_monotonic_index = int(non_monotonic[non_monotonic].index[0])
        raise ValueError(
            f"{context}: exit_time must be monotonically non-decreasing "
            f"(first non-monotonic row index: {first_non_monotonic_index})"
        )

    for col in REQUIRED_NUMERIC_COLUMNS:
        numeric = pd.to_numeric(trades_df[col], errors="coerce")
        invalid_count = int(numeric.isna().sum())
        if invalid_count:
            raise ValueError(
                f"{context}: column '{col}' must be numeric "
                f"(invalid rows: {invalid_count})"
            )


def validate_trade_csv_contract_path(
    csv_path: str | Path,
    context: str | None = None,
    validate_exit_time_monotonic: bool = True,
) -> pd.DataFrame:
    path = Path(csv_path)
    trades_df = pd.read_csv(path)
    resolved_context = context or f"trade CSV '{path.name}'"
    validate_trade_csv_contract(
        trades_df,
        context=resolved_context,
        validate_exit_time_monotonic=validate_exit_time_monotonic,
    )
    return trades_df
