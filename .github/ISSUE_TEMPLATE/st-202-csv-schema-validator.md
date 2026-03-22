---
name: "ST-202: CSV schema validator"
about: "Validate trade CSV schema contracts"
title: "ST-202 Add trade CSV schema validator"
labels: ["sprint", "testing", "high-priority"]
assignees: []
---

## Story
Implement schema validation for strategy trade CSV exports.

## Deliverables
- `src/core/testing/contracts.py`

## Acceptance Criteria
- Required columns validated for all strategies.
- Type checks for key fields (`net_pnl`, `position_size`, `bars_in_trade`).
- Failure messages are explicit and actionable.
