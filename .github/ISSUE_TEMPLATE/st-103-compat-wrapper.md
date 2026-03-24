---
name: "ST-103: Compatibility wrapper"
about: "Keep script-based selftest workflow compatible"
title: "ST-103 Keep compatibility wrapper script"
labels: ["sprint", "testing"]
assignees: []
---

## Story
Keep `scripts/test_strategies_selftest.py` usable after pytest refactor.

## Deliverables
- Wrapper or direct compatibility behavior retained.

## Acceptance Criteria
- Existing workflow does not break.
- Legacy command still runs and prints summary.
