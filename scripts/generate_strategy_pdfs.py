"""Generate strategy PDF explainers from walkthrough text and trade CSV data.

Usage:
    PYTHONPATH=. python3 scripts/generate_strategy_pdfs.py
"""

from __future__ import annotations

import shutil
import textwrap
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

import src.strategies.basic_keltner_reversion.backtest_basic_keltner_reversion_v2 as bk_runner
import src.strategies.bmsb.backtest_bmsb_v2 as bmsb_runner
import src.strategies.donchian_breakout.backtest_donchian_breakout_v2 as don_runner
import src.strategies.ema_cross.backtest_ema_cross_v2 as ema_runner
import src.strategies.ema_trend_hold.backtest_ema_trend_hold_v2 as trend_runner
import src.strategies.emalyarovich_smas.backtest_emalyarovich_smas_v2 as sma_runner
import src.strategies.k_davey_mom_keltner.backtest_k_davey_mom_keltner_v2 as kd_runner
import src.strategies.rsi_reversion.backtest_rsi_reversion_v2 as rsi_runner


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs" / "strategy_walkthroughs"
PDF_DIR = DOCS_DIR / "pdfs"
CSV_DIR = DOCS_DIR / "csv_baselines"
TEMP_RUNS_DIR = DOCS_DIR / "_generated_runs"


@contextmanager
def patched_attr(module, attr_name, replacement):
    original = getattr(module, attr_name)
    setattr(module, attr_name, replacement)
    try:
        yield
    finally:
        setattr(module, attr_name, original)


def make_synthetic_ohlcv(rows: int = 430, freq: str = "D") -> pd.DataFrame:
    idx = pd.date_range("2021-01-01", periods=rows, freq=freq)
    x = np.arange(rows, dtype=float)
    trend = 100.0 + 0.05 * x
    wave = 2.5 * np.sin(x / 6.0) + 1.0 * np.cos(x / 17.0)
    close = trend + wave
    open_ = close + 0.2 * np.sin(x / 3.0)
    high = np.maximum(open_, close) + 0.6
    low = np.minimum(open_, close) - 0.6
    volume = np.full(rows, 1000.0)
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def build_jobs(base_path: Path):
    common = {
        "exchange": "binance",
        "symbol": "BTC/USDT",
        "timeframe": "1d",
        "start_date": "2021-01-01",
        "end_date": "2022-12-31",
        "use_clean": True,
        "run_id": "pdfgen",
        "generate_report": False,
        "generate_plots": False,
        "generate_equity": False,
        "base_path": str(base_path),
    }
    return [
        ("ema_cross", ema_runner, ema_runner.run_backtest_ema_cross_v2, {**common, "ema_fast": 20, "ema_slow": 50}),
        (
            "rsi_reversion",
            rsi_runner,
            rsi_runner.run_backtest_rsi_reversion_v2,
            {**common, "rsi_period": 14, "rsi_entry": 30, "rsi_exit": 50},
        ),
        (
            "donchian_breakout",
            don_runner,
            don_runner.run_backtest_donchian_breakout_v2,
            {**common, "donchian_lookback": 20},
        ),
        (
            "ema_trend_hold",
            trend_runner,
            trend_runner.run_backtest_ema_trend_hold_v2,
            {**common, "trend_ema": 200},
        ),
        (
            "bmsb",
            bmsb_runner,
            bmsb_runner.run_backtest_bmsb_v2,
            {**common, "sma_period": 20, "ema_period": 21, "tensignal_window": 3},
        ),
        (
            "emalyarovich_smas",
            sma_runner,
            sma_runner.run_backtest_emalyarovich_smas_v2,
            {**common, "sma_fast": 20, "sma_slow": 200, "slope_bars": 3},
        ),
        (
            "k_davey_mom_keltner",
            kd_runner,
            kd_runner.run_backtest_k_davey_mom_keltner_v2,
            {**common, "symbol": "MES", "position_mode": "fixed", "trade_size": 1.0},
        ),
        (
            "basic_keltner_reversion",
            bk_runner,
            bk_runner.run_backtest_basic_keltner_reversion_v2,
            {**common, "kc_ema_length": 20, "kc_atr_length": 20, "kc_atr_mult": 1.5},
        ),
    ]


def collect_trade_csvs() -> dict[str, Path | None]:
    TEMP_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    synthetic_df = make_synthetic_ohlcv()

    def fake_fetch(**_kwargs):
        return synthetic_df.copy()

    csv_by_strategy: dict[str, Path | None] = {}
    for strategy_name, module, fn, kwargs in build_jobs(TEMP_RUNS_DIR):
        with patched_attr(module, "fetch_ohlcv", fake_fetch):
            _stats, _chart, csv_rel = fn(**kwargs)

        if not csv_rel:
            csv_by_strategy[strategy_name] = None
            continue

        csv_abs = TEMP_RUNS_DIR / "static" / csv_rel
        if not csv_abs.exists():
            csv_by_strategy[strategy_name] = None
            continue

        dst = CSV_DIR / f"{strategy_name}.csv"
        shutil.copyfile(csv_abs, dst)
        csv_by_strategy[strategy_name] = dst

    return csv_by_strategy


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


def create_comparison_pdf(csv_by_strategy: dict[str, Path | None]) -> None:
    txt_path = DOCS_DIR / "strategy_comparison.txt"
    if not txt_path.exists():
        return

    text = txt_path.read_text(encoding="utf-8")
    out = PDF_DIR / "strategy_comparison.pdf"

    rows = []
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

    csv_by_strategy = collect_trade_csvs()

    for walkthrough_path in sorted(DOCS_DIR.glob("*.txt")):
        if walkthrough_path.name == "strategy_comparison.txt":
            continue
        strategy_name = walkthrough_path.stem
        output_pdf = PDF_DIR / f"{strategy_name}.pdf"
        create_strategy_pdf(strategy_name, walkthrough_path, csv_by_strategy.get(strategy_name), output_pdf)

    create_comparison_pdf(csv_by_strategy)

    print("Generated PDFs in", PDF_DIR)
    print("Generated baseline CSVs in", CSV_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
