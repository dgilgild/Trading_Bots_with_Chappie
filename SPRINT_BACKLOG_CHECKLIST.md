# Sprint Backlog Checklist

Use this file as the sprint control board. Create one GitHub issue per story ID.

## Epic A - Test Harness Formalization
- [x] ST-101 Create pytest structure for strategy tests
- [ ] ST-102 Add test markers and selective runs
- [ ] ST-103 Keep compatibility wrapper script
- [ ] ST-104 Add explicit BMSB signal unit tests (entry, exit, trailing)

## Epic B - Baseline Data and Contracts
- [ ] ST-201 Freeze deterministic synthetic dataset contract
- [ ] ST-202 Add trade CSV schema validator
- [ ] ST-203 Add timeline integrity checks

## Epic C - Strategy Comparison Artifacts
- [ ] ST-301 Generate unified comparison JSON
- [ ] ST-302 Generate human-readable comparison report from JSON
- [ ] ST-303 Connect PDF generation to baseline metrics

## Epic D - Walk-Forward Validation (First Slice)
- [ ] ST-401 Implement rolling walk-forward evaluator
- [ ] ST-402 Extend walk-forward run to all strategies

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
