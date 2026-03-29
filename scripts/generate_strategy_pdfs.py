"""Generate strategy PDF explainers from walkthrough text and trade CSV data.

Usage:
    PYTHONPATH=. python3 scripts/generate_strategy_pdfs.py
"""

from __future__ import annotations

import textwrap
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

from src.core.analysis.strategy_metrics import generate_metrics_and_report


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs" / "strategy_walkthroughs"
PDF_DIR = DOCS_DIR / "pdfs"
CSV_DIR = DOCS_DIR / "csv_baselines"
TEMP_RUNS_DIR = DOCS_DIR / "_generated_runs"
METRICS_DIR = DOCS_DIR / "metrics"
METRICS_JSON_PATH = METRICS_DIR / "strategy_comparison.json"
METRICS_REPORT_PATH = METRICS_DIR / "strategy_comparison_report.txt"


def _add_text_page(pdf: PdfPages, title: str, body: str) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle(title, fontsize=18, y=0.98)
    wrapped = "\n".join(textwrap.wrap(body, width=105, break_long_words=False))
    fig.text(0.06, 0.95, wrapped, va="top", ha="left", fontsize=10, family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def create_strategy_pdf(strategy_name: str, walkthrough_path: Path, csv_path: Path | None, output_pdf: Path) -> None:
    text = walkthrough_path.read_text(encoding="utf-8")

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        _add_text_page(pdf, f"{strategy_name} - Strategy Walkthrough", text)

        if csv_path is None or not csv_path.exists():
            _add_text_page(
                pdf,
                f"{strategy_name} - Data note",
                "No trade CSV was produced for this strategy on the synthetic baseline dataset."
                "\nThis usually means the dataset did not trigger entries under default parameters.",
            )
            return

        trades = pd.read_csv(csv_path)
        if trades.empty:
            _add_text_page(pdf, f"{strategy_name} - Data note", "Trade CSV exists but has zero rows.")
            return

        trades["entry_time"] = pd.to_datetime(trades["entry_time"], errors="coerce")
        trades["exit_time"] = pd.to_datetime(trades["exit_time"], errors="coerce")
        trades["net_pnl"] = pd.to_numeric(trades["net_pnl"], errors="coerce").fillna(0.0)
        trades["bars_in_trade"] = pd.to_numeric(trades.get("bars_in_trade", 0), errors="coerce").fillna(0.0)

        win_rate = float((trades["net_pnl"] > 0).mean() * 100.0)
        total_net = float(trades["net_pnl"].sum())
        avg_pnl = float(trades["net_pnl"].mean())

        summary_text = (
            f"Trades: {len(trades)}\n"
            f"Total Net PnL: {total_net:.2f}\n"
            f"Average Net PnL per trade: {avg_pnl:.2f}\n"
            f"Win rate: {win_rate:.2f}%\n"
            f"First trigger: {trades.iloc[0].get('entry_trigger', 'n/a')}\n"
            f"Last trigger: {trades.iloc[-1].get('exit_trigger', 'n/a')}\n"
            f"Source CSV: {csv_path.relative_to(ROOT)}"
        )
        _add_text_page(pdf, f"{strategy_name} - Trade summary", summary_text)

        fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
        fig.suptitle(f"{strategy_name} - Trade analytics")

        equity_curve = trades["net_pnl"].cumsum()
        axes[0, 0].plot(range(1, len(equity_curve) + 1), equity_curve, color="#1f77b4")
        axes[0, 0].set_title("Cumulative Net PnL by trade")
        axes[0, 0].set_xlabel("Trade number")
        axes[0, 0].set_ylabel("PnL")
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].hist(trades["net_pnl"], bins=20, color="#2ca02c", alpha=0.8)
        axes[0, 1].set_title("Net PnL distribution")
        axes[0, 1].set_xlabel("Net PnL")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].grid(alpha=0.3)

        axes[1, 0].scatter(trades["bars_in_trade"], trades["net_pnl"], alpha=0.75, color="#ff7f0e")
        axes[1, 0].set_title("Duration vs Net PnL")
        axes[1, 0].set_xlabel("Bars in trade")
        axes[1, 0].set_ylabel("Net PnL")
        axes[1, 0].grid(alpha=0.3)

        trigger_counts = trades["entry_trigger"].astype(str).value_counts().head(8)
        axes[1, 1].barh(trigger_counts.index[::-1], trigger_counts.values[::-1], color="#9467bd")
        axes[1, 1].set_title("Top entry triggers")
        axes[1, 1].set_xlabel("Count")

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def _load_metrics_rows_from_json(metrics_json_path: Path) -> list[dict]:
    if not metrics_json_path.exists():
        return []

    payload = json.loads(metrics_json_path.read_text(encoding="utf-8"))
    strategies = payload.get("strategies", {})
    rows = []
    for strategy_name, metric in strategies.items():
        rows.append(
            {
                "strategy": strategy_name,
                "trades": int(metric.get("trades", 0)),
                "total_net_pnl": float(metric.get("total_net_pnl", 0.0)),
                "win_rate": float(metric.get("win_rate", 0.0)),
            }
        )
    return rows


def create_comparison_pdf(csv_by_strategy: dict[str, Path | None]) -> None:
    txt_path = DOCS_DIR / "strategy_comparison.txt"
    if not txt_path.exists():
        return

    text = txt_path.read_text(encoding="utf-8")
    out = PDF_DIR / "strategy_comparison.pdf"

    rows = _load_metrics_rows_from_json(METRICS_JSON_PATH)
    if not rows:
        for strategy, csv_path in csv_by_strategy.items():
            if not csv_path or not csv_path.exists():
                rows.append({"strategy": strategy, "trades": 0, "total_net_pnl": 0.0, "win_rate": 0.0})
                continue
            df = pd.read_csv(csv_path)
            if df.empty:
                rows.append({"strategy": strategy, "trades": 0, "total_net_pnl": 0.0, "win_rate": 0.0})
                continue
            pnl = pd.to_numeric(df["net_pnl"], errors="coerce").fillna(0.0)
            rows.append(
                {
                    "strategy": strategy,
                    "trades": int(len(df)),
                    "total_net_pnl": float(pnl.sum()),
                    "win_rate": float((pnl > 0).mean() * 100.0),
                }
            )
    metrics = pd.DataFrame(rows).sort_values("total_net_pnl", ascending=False)

    with PdfPages(out) as pdf:
        _add_text_page(pdf, "Strategy comparison", text)

        fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
        fig.suptitle("Synthetic baseline comparison")

        axes[0].barh(metrics["strategy"], metrics["total_net_pnl"], color="#1f77b4")
        axes[0].set_title("Total net PnL")
        axes[0].set_xlabel("PnL")
        axes[0].grid(alpha=0.3)

        axes[1].barh(metrics["strategy"], metrics["win_rate"], color="#2ca02c")
        axes[1].set_title("Win rate")
        axes[1].set_xlabel("Percent")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def main() -> int:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    csv_by_strategy, _payload = generate_metrics_and_report(
        temp_runs_dir=TEMP_RUNS_DIR,
        csv_dir=CSV_DIR,
        metrics_json_path=METRICS_JSON_PATH,
        metrics_report_path=METRICS_REPORT_PATH,
    )

    for walkthrough_path in sorted(DOCS_DIR.glob("*.txt")):
        if walkthrough_path.name == "strategy_comparison.txt":
            continue
        strategy_name = walkthrough_path.stem
        output_pdf = PDF_DIR / f"{strategy_name}.pdf"
        create_strategy_pdf(strategy_name, walkthrough_path, csv_by_strategy.get(strategy_name), output_pdf)

    create_comparison_pdf(csv_by_strategy)

    print("Generated PDFs in", PDF_DIR)
    print("Generated baseline CSVs in", CSV_DIR)
    print("Generated strategy metrics JSON in", METRICS_JSON_PATH)
    print("Generated strategy metrics report in", METRICS_REPORT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
