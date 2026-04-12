# Config Domain Split v1

Owner: Platform
Contract Version: 1

## Backtesting repo config scope
Allowed examples:
- Data source settings
- Backtest params and defaults
- Reporting toggles

## Future live repo config scope
Allowed examples:
- Exchange API keys
- Execution/risk runtime toggles
- Worker/process controls

## Security rule
- Do not introduce live execution credentials into backtesting workflows.
- Keep `.env` secrets out of git and logs.

## Naming convention
- Prefix config keys by domain where possible:
  - backtest: `BACKTEST_*`
  - live: `LIVE_*`
