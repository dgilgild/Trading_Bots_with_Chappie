# Sprint Backlog Checklist

Use this file as the sprint control board. Create one GitHub issue per story ID.

## Epic A - Test Harness Formalization
- [x] ST-101 Create pytest structure for strategy tests
- [x] ST-102 Add test markers and selective runs
- [x] ST-103 Keep compatibility wrapper script
- [x] ST-104 Add explicit BMSB signal unit tests (entry, exit, trailing)

## Epic B - Baseline Data and Contracts
- [x] ST-201 Freeze deterministic synthetic dataset contract
- [x] ST-202 Add trade CSV schema validator
- [x] ST-203 Add timeline integrity checks

## Epic C - Strategy Comparison Artifacts
- [x] ST-301 Generate unified comparison JSON
- [x] ST-302 Generate human-readable comparison report from JSON
- [x] ST-303 Connect PDF generation to baseline metrics

## Epic D - Walk-Forward Validation (First Slice)
- [x] ST-401 Implement rolling walk-forward evaluator
- [x] ST-402 Extend walk-forward run to all strategies

## Epic E - Trade Explainability (Phase 2)
- [x] ST-403 Expand full explainability to additional strategies
- [x] ST-404 Add explainability regression contract tests
- [x] ST-405 Build explainability summary index report

## Suggested Issue Creation Order
1. ST-201
2. ST-101
3. ST-102
4. ST-202
5. ST-104
6. ST-203
7. ST-301
8. ST-302
9. ST-303
10. ST-401
11. ST-402
