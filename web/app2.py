from flask import Flask, request, jsonify
from src.strategies.ema_cross.backtest_ema_cross import run_backtest_ema_cross
from src.visualization.plot_trades import plot_trades

app = Flask(__name__)


@app.route("/plot_trades", methods=["POST"])
def plot_trades_endpoint():
    payload = request.get_json()

    # -----------------------------
    # 1) Backtest params
    # -----------------------------
    exchange = payload["exchange"]
    symbol = payload["symbol"]
    timeframe = payload["timeframe"]
    start_date = payload.get("start_date")
    end_date = payload.get("end_date")

    ema_fast = payload["ema_fast"]
    ema_slow = payload["ema_slow"]
    use_clean = payload.get("use_clean", True)

    # -----------------------------
    # 2) Plot options
    # -----------------------------
    plot_mode = payload.get("plot_mode", "monthly")  # monthly | range
    months = payload.get("months")                   # ["2023-01", ...]
    date_from = payload.get("date_from")
    date_to = payload.get("date_to")

    min_bars_in_trade = payload.get("min_bars_in_trade", 0)

    show_entries = payload.get("show_entries", True)
    show_exits = payload.get("show_exits", True)
    show_trade_lines = payload.get("show_trade_lines", True)

    show_wins = payload.get("show_wins", True)
    show_losses = payload.get("show_losses", True)

    save_png = payload.get("save_png", True)

    # -----------------------------
    # 3) Run backtest
    # -----------------------------
    stats, chart_path, csv_path, df, trades = run_backtest_ema_cross(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        use_clean=use_clean,
        return_raw=True  # 👈 clave
    )

    # -----------------------------
    # 4) Plot trades
    # -----------------------------
    png_paths = plot_trades(
        df=df,
        trades=trades,
        plot_mode=plot_mode,
        months=months,
        date_from=date_from,
        date_to=date_to,
        min_bars_in_trade=min_bars_in_trade,
        show_entries=show_entries,
        show_exits=show_exits,
        show_trade_lines=show_trade_lines,
        show_wins=show_wins,
        show_losses=show_losses,
        save_png=save_png,
        title_prefix=f"EMA Cross {symbol} {timeframe}"
    )

    # -----------------------------
    # 5) Response
    # -----------------------------
    return jsonify({
        "status": "ok",
        "stats": stats,
        "equity_chart": chart_path,
        "trades_csv": csv_path,
        "trade_charts": png_paths
    })
