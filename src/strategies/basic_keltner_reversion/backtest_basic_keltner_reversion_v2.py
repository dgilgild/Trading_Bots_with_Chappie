import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from src.core.data import fetch_ohlcv
from src.core.backtester_v2 import BacktesterV2
from src.core.reporting import generate_quantstats_report
from src.strategies.basic_keltner_reversion.strategy import keltner_reversion


def export_trades_csv(trades, output_dir, run_id, metadata):
    if not trades:
        print("[WARN] No trades to export")
        return None, None

    df = pd.DataFrame(trades)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)

    df["pnl_pct"] = ((df["exit_price"] - df["entry_price"]) / df["entry_price"] * 100).round(3)
    df["net_return_pct"] = ((df["net_pnl"] / df["position_size"]) * 100).round(3)
    df["result"] = df["net_pnl"].apply(lambda x: "WIN" if x > 0 else "LOSS")
    df["balance"] = df["cash_after_trade"].round(6)

    for key, value in metadata.items():
        df[key] = value

    filename = f"{run_id}_trades.csv"
    path = os.path.join(output_dir, filename)

    df = df[
        [
            "run_id", "exchange", "symbol", "timeframe", "ema_fast", "ema_slow",
            "side", "result", "position_size", "qty", "pyramid_level", "balance",
            "cash_after_trade", "entry_time", "exit_time", "entry_price", "exit_price",
            "stop_price", "take_profit_price", "commission_pct", "slippage_pct",
            "stop_loss_pct", "take_profit_pct", "gross_pnl", "commission_paid", "pnl_pct",
            "net_return_pct", "net_pnl", "entry_trigger", "exit_trigger", "bars_in_trade",
            "use_clean", "position_mode",
        ]
    ]

    df.to_csv(path, index=False)
    return path, f"backtests/basic_keltner_reversion/{run_id}/{filename}"


def run_backtest_basic_keltner_reversion_v2(
    exchange,
    symbol,
    timeframe,
    start_date,
    end_date,
    kc_ema_length=20,
    kc_atr_length=20,
    kc_atr_mult=1.5,
    use_clean=True,
    run_id=None,
    initial_balance=1000.0,
    position_mode="all_in",
    trade_size=100.0,
    position_pct=None,
    commission_pct=0.001,
    slippage_pct=0.001,
    allow_short=True,
    stop_loss_pct=0.02,
    take_profit_pct=0.03,
    pyramiding=1,
    generate_report=True,
    generate_plots=True,
    generate_equity=True,
    base_path=None,
):
    output_dir = os.path.join(base_path, "static", "backtests", "basic_keltner_reversion", run_id)
    os.makedirs(output_dir, exist_ok=True)

    df = fetch_ohlcv(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        limit=50000,
        use_clean=use_clean,
    )
    if df is None or df.empty:
        return {}, None, None

    bt = BacktesterV2(
        initial_capital=initial_balance,
        position_mode=position_mode,
        trade_size=trade_size,
        position_pct=position_pct,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
        allow_short=allow_short,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        pyramiding=pyramiding,
    )

    warmup = max(int(kc_ema_length), int(kc_atr_length)) + 1
    for i in range(warmup, len(df)):
        bar = df.iloc[i]
        timestamp = bar["timestamp"]
        price = bar["close"]

        bt.on_bar(high=bar["high"], low=bar["low"], timestamp=timestamp, bar_index=i)

        signal, trigger = keltner_reversion(
            df.iloc[: i + 1],
            ema_length=kc_ema_length,
            atr_length=kc_atr_length,
            atr_mult=kc_atr_mult,
        )

        bt.on_signal(signal, price, timestamp, trigger, i)

    stats = bt.stats()
    clean_stats = {k: v.item() if hasattr(v, "item") else v for k, v in stats.items()}

    equity = []
    equity_dates = []
    current_equity = bt.initial_capital
    for t in bt.trades:
        current_equity += t["net_pnl"]
        equity.append(current_equity)
        equity_dates.append(t["exit_time"])

    chart_rel = None
    if generate_equity:
        filename = f"equity_curve_{run_id}.png"
        full = os.path.join(output_dir, filename)
        plt.figure(figsize=(10, 5))
        plt.plot(equity_dates, equity)
        plt.title("Equity Curve - Basic Keltner Channel Reversion")
        plt.xlabel("Date")
        plt.ylabel("Equity ($)")
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(full)
        plt.close()
        chart_rel = f"backtests/basic_keltner_reversion/{run_id}/{filename}"

    if generate_report:
        generate_quantstats_report(equity_dates, equity, output_dir, f"Basic KC Reversion {symbol} {timeframe}")

    _, csv_rel = export_trades_csv(
        bt.trades,
        output_dir,
        run_id,
        metadata={
            "run_id": run_id,
            "exchange": exchange,
            "symbol": symbol,
            "timeframe": timeframe,
            "ema_fast": None,
            "ema_slow": None,
            "use_clean": use_clean,
            "initial_balance": initial_balance,
            "position_mode": position_mode,
            "trade_size": trade_size,
            "commission_pct": commission_pct,
            "slippage_pct": slippage_pct,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "allow_short": allow_short,
            "kc_ema_length": kc_ema_length,
            "kc_atr_length": kc_atr_length,
            "kc_atr_mult": kc_atr_mult,
        },
    )

    return clean_stats, chart_rel, csv_rel
