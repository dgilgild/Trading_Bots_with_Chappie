# Strategy Signal Contract v1

Owner: Strategy/Research
Contract Version: 1

## Input expectations
- Input data is OHLCV-like with at least:
  - `timestamp`
  - `open`
  - `high`
  - `low`
  - `close`
- Data should be time-ordered ascending.

## Signal function output
- Tuple shape:
  - `(signal, trigger)`
- `signal` allowed values:
  - `"LONG"`
  - `"SHORT"`
  - `"EXIT"`
  - `None`
- `trigger`:
  - short human-readable reason string
  - may be `None` (consumer should fallback safely)

## Consumer behavior contract
- Consumer must treat unknown/None signal as no-op.
- Consumer should persist trigger when available.
- Consumer must not assume strategy supports shorting unless configured.

## Compatibility notes
- Any new signal value requires contract bump and migration notes.
