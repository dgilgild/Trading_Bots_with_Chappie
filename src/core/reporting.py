from pathlib import Path

import pandas as pd
import quantstats as qs


def generate_quantstats_report(equity_dates, equity_values, output_dir, title):
    if not equity_dates or not equity_values:
        return None

    if len(equity_dates) < 2 or len(equity_values) < 2:
        return None

    equity_series = pd.Series(
        equity_values,
        index=pd.to_datetime(equity_dates, utc=True),
    ).sort_index()

    if equity_series.index.has_duplicates:
        equity_series = equity_series.groupby(level=0).last()

    returns = equity_series.pct_change().fillna(0.0)

    output_path = Path(output_dir) / "quantstats_report.html"
    try:
        qs.reports.html(returns, output=str(output_path), title=title)
        return str(output_path)
    except Exception as exc:
        equity = equity_series.values
        total_return = (equity[-1] / equity[0] - 1) if len(equity) > 1 else 0.0
        peak = equity[0]
        max_dd = 0.0
        for val in equity:
            if val > peak:
                peak = val
            drawdown = (val - peak) / peak if peak else 0.0
            if drawdown < max_dd:
                max_dd = drawdown

        output_path.write_text(
            "\n".join(
                [
                    "<html><head><title>QuantStats Report (Fallback)</title></head><body>",
                    f"<h1>{title}</h1>",
                    "<p>QuantStats report failed to render. Fallback summary below.</p>",
                    f"<p><strong>Total Return:</strong> {total_return:.2%}</p>",
                    f"<p><strong>Max Drawdown:</strong> {max_dd:.2%}</p>",
                    f"<pre>Exception: {exc}</pre>",
                    "</body></html>",
                ]
            ),
            encoding="utf-8",
        )
        return str(output_path)
