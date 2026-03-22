---
name: "ST-301: Comparison JSON"
about: "Generate unified machine-readable strategy metrics"
title: "ST-301 Generate unified comparison JSON"
labels: ["sprint", "analytics"]
assignees: []
---

## Story
Produce a single strategy comparison JSON artifact.

## Deliverables
- `docs/strategy_walkthroughs/metrics/strategy_comparison.json`

## Acceptance Criteria
- Includes per strategy: trades, net_pnl, win_rate, avg_pnl, avg_bars_in_trade, drawdown proxy.
- Deterministic values on deterministic baseline data.
