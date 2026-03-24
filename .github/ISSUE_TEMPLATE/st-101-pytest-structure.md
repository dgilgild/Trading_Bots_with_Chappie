---
name: "ST-101: Pytest structure"
about: "Create pytest structure for strategy tests"
title: "ST-101 Create pytest structure for strategy tests"
labels: ["sprint", "testing", "high-priority"]
assignees: []
---

## Story
Create a pytest-based test structure for strategy validation.

## Deliverables
- `tests/strategies/test_signal_logic.py`
- `tests/strategies/test_backtest_smoke.py`
- `tests/strategies/test_backtest_regression.py`
- `tests/strategies/conftest.py`

## Acceptance Criteria
- Existing script checks are mirrored in pytest.
- Tests run locally with `PYTHONPATH=. pytest`.

## Notes
- Reuse current logic from `scripts/test_strategies_selftest.py`.
