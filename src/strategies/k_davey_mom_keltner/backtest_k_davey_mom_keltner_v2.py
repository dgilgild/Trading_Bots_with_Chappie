import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from src.core.data import fetch_ohlcv
from src.core.backtester_v2 import BacktesterV2
from src.core.reporting import generate_quantstats_report
from src.core.ta import compute_atr
from src.strategies.k_davey_mom_keltner.strategy import (
    compute_keltner_stochastic,
    compute_position_size,
)


def export_trades_csv(trades, output_dir, run_id, metadata):
    if not trades:
        print("[WARN] No trades to export")
        return None, None

    df = pd.DataFrame(trades)

    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)

    df["pnl_pct"] = (
        (df["exit_price"] - df["entry_price"]) / df["entry_price"] * 100
    ).round(3)

    df["net_return_pct"] = (
        (df["net_pnl"] / df["position_size"]) * 100
    ).round(3)

    df["result"] = df["net_pnl"].apply(lambda x: "WIN" if x > 0 else "LOSS")
    df["balance"] = df["cash_after_trade"].round(6)

    for key, value in metadata.items():
        df[key] = value

    filename = f"{run_id}_trades.csv"
    path = os.path.join(output_dir, filename)

    df = df[
        [
            "run_id",
            "exchange",
            "symbol",
            "timeframe",
            "ema_fast",
            "ema_slow",
            "side",
            "result",
            "position_size",
            "qty",
            "pyramid_level",
            "balance",
            "cash_after_trade",
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "stop_price",
            "take_profit_price",
            "commission_pct",
            "slippage_pct",
            "stop_loss_pct",
            "take_profit_pct",
            "gross_pnl",
            "commission_paid",
            "pnl_pct",
            "net_return_pct",
            "net_pnl",
            "entry_trigger",
            "exit_trigger",
            "bars_in_trade",
            "use_clean",
            "position_mode",
        ]
    ]

    df.to_csv(path, index=False)

    DB_csv_path = f"backtests/k_davey_mom_keltner/{run_id}/{filename}"

    return path, DB_csv_path


def run_backtest_k_davey_mom_keltner_v2(
    exchange,
    symbol,
    timeframe,
    start_date,
    end_date,
    mom_length_long=40,
    mom_length_short=40,
    keltner_length=5,
    keltner_atr_mult=0.5,
    entry_threshold=30.0,
    exit_threshold=70.0,
    trend_ema=200,
    volatility_atr_period=14,
    volatility_sma_period=100,
    volatility_mult=0.8,
    use_position_sizing=True,
    base_equity=15000.0,
    sizing_factor=0.33,
    max_contracts=15,
    use_clean=True,
    run_id=None,
    initial_balance=1000.0,
    position_mode="fixed",
    trade_size=100.0,
    commission_pct=0.001,
    slippage_pct=0.001,
    allow_short=True,
    atr_period=14,
    atr_sl_mult_long=4.0,
    atr_sl_mult_short=1.0,
    pyramiding=1,
    position_pct=None,
    generate_report=True,
    generate_plots=True,
    generate_equity=True,
    base_path=None,
):

    output_dir = os.path.join(
        base_path,
        "static",
        "backtests",
        "k_davey_mom_keltner",
        run_id,
    )

    os.makedirs(output_dir, exist_ok=True)

    print("K. DAVEY MOM+KELTNER RUN ID:", run_id)
    print("OUTPUT DIR:", output_dir)

    if timeframe != "1d":
        print("[WARN] K. Davey strategy is designed for 1d bars")

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
        print("[WARN] No data returned for K. Davey backtest")
        return {}, None, None

    df["keltner_stoch"] = compute_keltner_stochastic(
        df,
        keltner_length,
        keltner_atr_mult,
        mamode="ema",
    )
    df["trend_ema"] = df["close"].ewm(span=int(trend_ema), adjust=False).mean()
    atr_series = compute_atr(df, atr_period)
    atr_vol = compute_atr(df, volatility_atr_period)
    atr_vol_avg = atr_vol.rolling(int(volatility_sma_period)).mean() if atr_vol is not None else None

    symbol_upper = symbol.upper()
    if base_equity is None:
        base_equity = 1500.0 if symbol_upper in {"MES", "MES=F"} else 15000.0

    contract_multiplier = 5.0 if symbol_upper in {"MES", "MES=F"} else 50.0

    bt = BacktesterV2(
        initial_capital=initial_balance,
        position_mode=position_mode,
        trade_size=trade_size,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
        allow_short=allow_short,
        stop_loss_pct=None,
        take_profit_pct=None,
        atr_sl_mult_long=atr_sl_mult_long,
        atr_sl_mult_short=atr_sl_mult_short,
        pnl_mode="futures",
        contract_multiplier=contract_multiplier,
        pyramiding=pyramiding,
        position_pct=position_pct,
    )

    pending_action = None

    start_index = max(
        mom_length_long,
        mom_length_short,
        keltner_length,
        atr_period,
        trend_ema,
        volatility_atr_period,
        volatility_sma_period,
    ) + 1
    for i in range(start_index, len(df)):
        current_bar = df.iloc[i]
        open_price = current_bar["open"]
        price = current_bar["close"]
        high = current_bar["high"]
        low = current_bar["low"]
        timestamp = current_bar["timestamp"]

        if pending_action is not None:
            if pending_action["type"] == "ENTRY":
                bt.on_signal(
                    pending_action["side"],
                    open_price,
                    timestamp,
                    pending_action["trigger"],
                    i,
                    atr_value=pending_action["atr_value"],
                )
            elif pending_action["type"] == "EXIT":
                bt.on_signal(
                    "EXIT",
                    open_price,
                    timestamp,
                    pending_action["trigger"],
                    i,
                )
            pending_action = None

        atr_value = None
        if atr_series is not None:
            atr_raw = atr_series.iloc[i]
            if pd.notna(atr_raw):
                atr_value = float(atr_raw)

        if bt.position is not None and atr_value is not None:
            bt.update_stop_from_avg(atr_value)

        bt.on_bar(high=high, low=low, timestamp=timestamp, bar_index=i)

        if atr_value is None:
            continue

        open_pnl = 0.0
        if bt.lots:
            for lot in bt.lots:
                entry_price = lot["entry_price"]
                qty = lot["qty"]
                if lot["side"] == "LONG":
                    open_pnl += (price - entry_price) * contract_multiplier * qty
                else:
                    open_pnl += (entry_price - price) * contract_multiplier * qty

        net_profit = bt.cash - initial_balance
        net_equity = base_equity + net_profit + open_pnl

        if position_pct is not None:
            capital_to_use = bt.cash * position_pct
            contracts = int(capital_to_use / (price * contract_multiplier))
            ncons = min(max(contracts, 1), int(max_contracts))
        else:
            ncons = compute_position_size(
                net_equity=net_equity,
                base_equity=base_equity,
                sizing_factor=sizing_factor,
                max_contracts=max_contracts,
                use_position_sizing=use_position_sizing,
            )
        bt.trade_size = float(ncons)

        keltner_stoch = current_bar["keltner_stoch"]

        if i >= len(df) - 1:
            continue

        long_cond = price > df["close"].iloc[i - int(mom_length_long)]
        short_cond = price < df["close"].iloc[i - int(mom_length_short)]
        trend_long = price > current_bar["trend_ema"]
        trend_short = price < current_bar["trend_ema"]

        volatility_ok = True
        if atr_vol is not None and atr_vol_avg is not None:
            vol_raw = atr_vol.iloc[i]
            vol_avg = atr_vol_avg.iloc[i]
            if pd.isna(vol_raw) or pd.isna(vol_avg):
                volatility_ok = False
            else:
                volatility_ok = float(vol_raw) > float(vol_avg) * float(volatility_mult)

        if bt.position is not None:
            if bt.position["side"] == "LONG" and keltner_stoch > exit_threshold:
                pending_action = {
                    "type": "EXIT",
                    "trigger": "Keltner stoch exit",
                }
            elif bt.position["side"] == "SHORT" and keltner_stoch < (100 - exit_threshold):
                pending_action = {
                    "type": "EXIT",
                    "trigger": "Keltner stoch exit",
                }
            if pending_action is not None:
                continue

        if bt.position is None or (bt.position["side"] == "LONG" and len(bt.lots) < bt.pyramiding):
            if long_cond and trend_long and volatility_ok and keltner_stoch < entry_threshold:
                pending_action = {
                    "type": "ENTRY",
                    "side": "LONG",
                    "trigger": "Momentum+Keltner long",
                    "atr_value": atr_value,
                }

        if allow_short and (bt.position is None or (bt.position["side"] == "SHORT" and len(bt.lots) < bt.pyramiding)):
            if short_cond and trend_short and volatility_ok and keltner_stoch > (100 - entry_threshold):
                pending_action = {
                    "type": "ENTRY",
                    "side": "SHORT",
                    "trigger": "Momentum+Keltner short",
                    "atr_value": atr_value,
                }

    stats = bt.stats()
    clean_stats = {k: v.item() if hasattr(v, "item") else v for k, v in stats.items()}

    equity = []
    equity_dates = []
    current_equity = bt.initial_capital

    for t in bt.trades:
        current_equity += t["net_pnl"]
        equity.append(current_equity)
        equity_dates.append(t["exit_time"])

    DB_equity_path = None
    if generate_equity:
        equity_filename = f"equity_curve_{run_id}.png"
        equity_path = os.path.join(output_dir, equity_filename)

        plt.figure(figsize=(10, 5))
        plt.plot(equity_dates, equity)
        plt.title("Equity Curve - K. Davey Momentum+Keltner")
        plt.xlabel("Date")
        plt.ylabel("Equity ($)")
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(equity_path)
        plt.close()

        DB_equity_path = f"backtests/k_davey_mom_keltner/{run_id}/{equity_filename}"

    if generate_report:
        generate_quantstats_report(
            equity_dates,
            equity,
            output_dir,
            title=f"K. Davey Momentum+Keltner {symbol} {timeframe}",
        )

    csv_path, DB_csv_path = export_trades_csv(
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
            "stop_loss_pct": None,
            "take_profit_pct": None,
            "allow_short": allow_short,
            "mom_length_long": mom_length_long,
            "mom_length_short": mom_length_short,
            "keltner_length": keltner_length,
            "keltner_atr_mult": keltner_atr_mult,
            "entry_threshold": entry_threshold,
            "exit_threshold": exit_threshold,
            "trend_ema": trend_ema,
            "volatility_atr_period": volatility_atr_period,
            "volatility_sma_period": volatility_sma_period,
            "volatility_mult": volatility_mult,
            "use_position_sizing": use_position_sizing,
            "base_equity": base_equity,
            "sizing_factor": sizing_factor,
            "max_contracts": max_contracts,
            "atr_period": atr_period,
            "atr_sl_mult_long": atr_sl_mult_long,
            "atr_sl_mult_short": atr_sl_mult_short,
        },
    )

    return clean_stats, DB_equity_path, DB_csv_path
