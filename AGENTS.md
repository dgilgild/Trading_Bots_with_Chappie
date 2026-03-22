# AGENTS.md

This repository is a lightweight Flask-based trading research app with
supporting scripts and backtest utilities. The notes below are for
agentic coding assistants so they can run the project safely and follow
existing conventions.

No Cursor or Copilot rules were found.

-----------------------------------------------------------------------
Quickstart
-----------------------------------------------------------------------
- Python dependencies: `pip install -r requirements.txt`
- Initialize the DB (creates data/market_data.db):
  `python scripts/init_db.py`
- Run the Flask app (web UI):
  `python web/app.py`

-----------------------------------------------------------------------
Build / Lint / Test Commands
-----------------------------------------------------------------------
There is no formal build or lint step configured. Use the scripts below
as the closest equivalents.

Install deps
- `pip install -r requirements.txt`

Run the web app
- `python web/app.py`

Run a backtest via UI
- Start the app, then use the form in the web UI to trigger a run.

Database and data utilities
- Init DB: `python scripts/init_db.py`
- Download data: `python scripts/download_data.py`
- Sanitize data: `python scripts/sanitize_data.py`

Tests (script-based)
- Data load smoke test: `python scripts/test_load_data.py`
- DB tables check: `python scripts/test_runs_table.py`

Single-test equivalents
- Single test is a script: `python scripts/test_load_data.py`
- Another single check: `python scripts/test_runs_table.py`

Notes
- There is no pytest or unittest suite in this repo at the moment.
- Some scripts rely on API credentials in `.env`.

-----------------------------------------------------------------------
Runtime Environment
-----------------------------------------------------------------------
- Python 3.10+ recommended (repo uses standard stdlib + pandas/numpy).
- `.env` keys used by `src/core/exchange.py`:
  - `BINANCE_API_KEY`
  - `BINANCE_API_SECRET`
- SQLite DB path: `data/market_data.db`

-----------------------------------------------------------------------
Code Style Guidelines
-----------------------------------------------------------------------
General
- Prefer clear, explicit code over clever shortcuts.
- Keep functions small and single-purpose.
- Avoid adding unnecessary dependencies.

Formatting
- Follow PEP 8 with 4-space indentation.
- Keep line length ~100 chars when practical.
- Use trailing commas in multi-line literals.

Imports
- Prefer absolute imports using `src.*` for app and library modules.
- Within a package, relative imports are OK for closely related modules.
- Do not mix `core.*` and `src.core.*` in the same runtime path.
- Order imports: stdlib, third-party, local.

Naming
- Modules, functions, and variables: `snake_case`.
- Classes: `PascalCase`.
- Constants: `UPPER_SNAKE_CASE` (ex: config values).
- Boolean flags: `is_*/has_*` when the meaning is not obvious.

Types
- Typing is optional but welcome for new or complex functions.
- Prefer `-> pd.DataFrame` and `Optional[...]` for clarity.
- Avoid type hints that require extra deps not already in requirements.

Error Handling
- Validate user inputs in Flask routes; return helpful error responses.
- Use `ValueError` for invalid parameter values in core utilities.
- Avoid bare `except:`; catch specific exceptions.
- In scripts, fail fast with clear prints and non-zero exit when needed.

Data and Time Handling
- Timestamps are milliseconds UTC in the SQLite DB.
- When converting dates, prefer explicit timezone handling.
- In pandas, keep `timestamp` as datetime for plotting and UI.

Backtesting and Trading Logic
- `check_signal()` returns `(signal, trigger)`; treat it as a tuple.
- Avoid signal evaluation on empty or too-short DataFrames.
- When adding commissions/slippage, keep cash accounting and reported
  `net_pnl` consistent.

Flask App Patterns
- Use `current_app.root_path` to build paths for outputs.
- Persist run metadata in `backtest_runs` (stats, params, chart paths).
- Keep routes thin; push logic into `src/` modules.

File and DB Hygiene
- Do not commit or expose `.env` secrets.
- Avoid writing large data files outside `data/` or `web/static/`.

-----------------------------------------------------------------------
Repo Map (High Level)
-----------------------------------------------------------------------
- `web/app.py`: Flask entry point and routes.
- `src/core/`: DB, data loaders, backtester V2, plotting helpers.
- `src/strategies/ema_cross/`: EMA cross strategy + backtest runner.
- `scripts/`: utility and smoke-test scripts.

-----------------------------------------------------------------------
When Adding New Code
-----------------------------------------------------------------------
- Keep behavior consistent with existing data formats (ms timestamps).
- Add or update scripts instead of inventing new ad-hoc entry points.
- If you add new config, put it in `src/strategies/.../config.py`.
- Update this file if you introduce linting, tests, or new commands.
