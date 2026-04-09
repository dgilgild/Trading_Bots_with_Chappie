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

## Epic F - Trade Viewer UX Enhancements
- [x] ST-406 Persist last selected trade per run in UI
- [x] ST-407 Add trade filter controls (LONG/SHORT/winning/losing)
- [x] ST-408 Add indicator toggle panel for overlays
- [x] ST-409 Export single-trade packet from UI (png + txt)

## Epic G - Live Trading Platform Foundation
- [ ] ST-500 Define control plane vs execution plane interfaces
- [ ] ST-501 Add live worker service skeleton and bootstrap contract (runtime engine)
- [ ] ST-502 Introduce live state persistence schema (live DB)
- [ ] ST-503 Implement systemd deployment assets
- [ ] ST-504 Add heartbeat and health visibility path
- [ ] ST-505 Enforce live risk guardrails and kill-switch
- [ ] ST-506 Harden secrets and host security baseline
- [ ] ST-507 Create deployment and rollback runbook
- [ ] ST-508 Execute paper-trading soak validation
- [ ] ST-509 Implement Binance execution service adapter
- [ ] ST-510 Implement /live control page and API routes

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
12. ST-406
13. ST-407
14. ST-408
15. ST-409
16. ST-500
17. ST-501
18. ST-502
19. ST-503
20. ST-504
21. ST-505
22. ST-506
23. ST-507
24. ST-508
25. ST-509
26. ST-510
