#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/daniel/Python_Code/chappie-backtesting-lab"
VENV_DIR="$PROJECT_DIR/.chappie_backtesting"

if [ ! -d "$PROJECT_DIR" ]; then
  echo "ERROR: Project directory not found: $PROJECT_DIR"
  exit 1
fi

cd "$PROJECT_DIR"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
  echo "ERROR: Backtesting venv not found at $VENV_DIR"
  echo "Create it first with: python3 -m venv .chappie_backtesting"
  exit 1
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo "=== CHAPPIE BACKTESTING SESSION ==="
echo "PWD: $(pwd)"
echo "Python: $(python -c 'import sys; print(sys.executable)')"
echo "Prefix: $(python -c 'import sys; print(sys.prefix)')"
echo "Branch: $(git branch --show-current)"
echo "--- git status ---"
git status -sb

echo
echo "Ready. Common commands:"
echo "  PYTHONPATH=. python web/app.py"
echo "  PYTHONPATH=. pytest tests/strategies -q"
echo "  PYTHONPATH=. python scripts/test_strategies_selftest.py"
