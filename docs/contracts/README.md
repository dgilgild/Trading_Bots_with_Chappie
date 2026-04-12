# Contracts and Boundaries (v1)

Purpose
- Keep `chappie-backtesting-lab` focused on research/backtesting only.
- Define stable artifacts for a future live-trading project to consume safely.

Contract version
- `contract_version: 1`

Files
- `boundaries_v1.md` - ownership split between backtesting and future live repo.
- `strategy_signal_contract_v1.md` - signal I/O schema expected by strategy logic.
- `backtest_result_contract_v1.md` - run/trade artifacts and metadata schema.
- `config_domain_split_v1.md` - environment/config separation rules.

Change control
- Any breaking change must:
  1. bump contract version,
  2. document migration notes,
  3. update downstream consumer assumptions.
