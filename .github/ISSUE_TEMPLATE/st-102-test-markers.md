---
name: "ST-102: Test markers"
about: "Add pytest markers and selective runs"
title: "ST-102 Add test markers and selective runs"
labels: ["sprint", "testing", "high-priority"]
assignees: []
---

## Story
Add marker-based test selection for unit, smoke, and regression suites.

## Deliverables
- Marker setup for `unit`, `smoke`, `regression`
- Docs update in `docs/selftest_manual.txt`

## Acceptance Criteria
- `pytest -m unit` works.
- `pytest -m smoke` works.
- `pytest -m regression` works.
