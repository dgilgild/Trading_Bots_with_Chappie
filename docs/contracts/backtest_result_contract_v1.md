# Backtest Result Contract v1

Owner: Backtesting Platform
Contract Version: 1

## Run metadata (backtest_runs)
Minimum fields expected by consumers:
- `run_id`
- `strategy`
- `exchange`
- `symbol`
- `timeframe`
- `start_ts`
- `end_ts`
- `params_json`
- `stats_json`
- `chart_path` (nullable)
- `csv_path` (nullable)

## Trades CSV contract (when trades exist)
Required columns:
- `entry_time`
- `exit_time`
- `entry_price`
- `exit_price`
- `side`
- `net_pnl`
- `entry_trigger`
- `exit_trigger`
- `position_size`
- `bars_in_trade`
- `pyramid_level`

## No-trade behavior
- `stats_json` may be empty or indicate no-data/no-trade status.
- `csv_path` may be null.
- UI/consumers must show explicit no-trade/no-data reason when available.

## Explainability/aux artifacts
- Optional per-run outputs can include:
  - explainability images/text
  - summary json/txt
- These are additive and should not break base run consumption.
