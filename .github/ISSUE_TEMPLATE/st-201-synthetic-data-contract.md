---
name: "ST-201: Synthetic data contract"
about: "Freeze deterministic synthetic dataset contract"
title: "ST-201 Freeze deterministic synthetic dataset contract"
labels: ["sprint", "testing", "high-priority"]
assignees: []
---

## Story
Create a centralized deterministic OHLCV generator for all strategy tests.

## Deliverables
- `src/core/testing/synthetic_data.py`

## Acceptance Criteria
- Same input parameters always produce identical output.
- Test suites consume this module instead of local duplicated generators.
