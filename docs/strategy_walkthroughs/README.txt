Strategy Walkthroughs Index

This folder contains plain-text walkthroughs, generated PDF explainers, and
baseline CSV files used to create strategy visualizations.

Main documents
- basic_keltner_reversion.txt : Step-by-step explanation for Basic KC Reversion.
- bmsb.txt : Step-by-step explanation for BMSB strategy.
- donchian_breakout.txt : Step-by-step explanation for Donchian Breakout.
- ema_cross.txt : Step-by-step explanation for EMA Cross strategy.
- ema_trend_hold.txt : Step-by-step explanation for EMA Trend Hold strategy.
- emalyarovich_smas.txt : Step-by-step explanation for E. Malyarovich SMAs.
- k_davey_mom_keltner.txt : Step-by-step explanation for K. Davey Momentum+Keltner.
- rsi_reversion.txt : Step-by-step explanation for RSI Reversion.
- strategy_comparison.txt : Cross-strategy comparison and ranking.

Generated PDF explainers
- pdfs/basic_keltner_reversion.pdf
- pdfs/bmsb.pdf
- pdfs/donchian_breakout.pdf
- pdfs/ema_cross.pdf
- pdfs/ema_trend_hold.pdf
- pdfs/emalyarovich_smas.pdf
- pdfs/k_davey_mom_keltner.pdf
- pdfs/rsi_reversion.pdf
- pdfs/strategy_comparison.pdf

Generated baseline CSV files
- csv_baselines/basic_keltner_reversion.csv
- csv_baselines/bmsb.csv
- csv_baselines/ema_cross.csv
- csv_baselines/ema_trend_hold.csv
- csv_baselines/emalyarovich_smas.csv
- csv_baselines/k_davey_mom_keltner.csv
- csv_baselines/rsi_reversion.csv

Generated comparison metrics artifacts
- metrics/strategy_comparison.json
- metrics/strategy_comparison_report.txt

Notes
- CSV baselines are created from deterministic synthetic OHLCV data.
- Some strategies may produce no trades on the baseline dataset (for example,
  Donchian in current defaults), so no CSV may exist for that strategy.

Regeneration commands
1) Run self-tests:
   PYTHONPATH=. python3 scripts/test_strategies_selftest.py

2) Regenerate PDFs and baseline CSVs:
   PYTHONPATH=. python3 scripts/generate_strategy_pdfs.py

   This command also regenerates the unified deterministic strategy metrics JSON
   and the human-readable comparison report under metrics/.
