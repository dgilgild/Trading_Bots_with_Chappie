---
name: "ST-401: Walk-forward core"
about: "Implement rolling walk-forward evaluator"
title: "ST-401 Implement rolling walk-forward evaluator"
labels: ["sprint", "analytics", "enhancement"]
assignees: []
---

## Story
Implement a configurable rolling walk-forward evaluator.

## Deliverables
- `src/core/analysis/walk_forward.py`

## Acceptance Criteria
- Configurable windows (example: 18m train / 6m test).
- Outputs per-window CSV and/or JSON metrics.
