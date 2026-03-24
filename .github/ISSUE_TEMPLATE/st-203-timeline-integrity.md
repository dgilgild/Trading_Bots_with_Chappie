---
name: "ST-203: Timeline integrity checks"
about: "Validate entry/exit timestamp consistency"
title: "ST-203 Add timeline integrity checks"
labels: ["sprint", "testing", "high-priority"]
assignees: []
---

## Story
Add timeline integrity checks for trade exports.

## Acceptance Criteria
- `entry_time <= exit_time` for closed trades.
- No null entry/exit timestamps in closed trades.
- Optional monotonic exit ordering check.
