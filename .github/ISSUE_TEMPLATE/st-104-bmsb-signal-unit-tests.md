---
name: "ST-104: BMSB signal unit tests"
about: "Add explicit BMSB entry/exit/trailing unit-level coverage"
title: "ST-104 Add explicit BMSB signal unit tests (entry, exit, trailing)"
labels: ["sprint", "testing", "high-priority"]
assignees: []
---

## Story
Current BMSB unit coverage is partial (indicator shape/columns only). Add explicit tests for strategy behavior.

## Deliverables
- Update `tests/strategies/test_signal_logic.py` with BMSB behavior tests.

## Acceptance Criteria
- Entry condition validated (`buysignal` + `tensignal >= 1`).
- Exit condition validated (crossunder logic).
- Trailing-stop update path validated.
- Coverage matrix updated from PARTIAL to YES for BMSB unit coverage.
