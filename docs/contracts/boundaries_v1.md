# Module Boundaries v1

Owner: Backtesting Team
Contract Version: 1

## Stays in `chappie-backtesting-lab`
- Strategy research and parameter exploration
- Backtesting engine and accounting
- Historical data ingestion/sanitization
- Walk-forward analysis
- Explainability/reporting for backtests
- Test harness and regression baselines

## Moves to future live repo (`chappie-live-trading`)
- Exchange execution adapters (Binance/etc.)
- Real-time signal execution loop
- Order lifecycle and reconciliation
- Live risk guardrails and kill-switch runtime
- 24x7 process supervision specifics for execution workers

## Forbidden in backtesting repo
- No live order placement code
- No broker credentials handling for live execution workflows
- No runtime loop that can submit real orders

## Inter-repo dependency rule
- Live repo may consume strategy/backtest contracts from this repo.
- Backtesting repo must not depend on live repo runtime code.
