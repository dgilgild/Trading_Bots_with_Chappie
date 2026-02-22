from flask import Flask, request, jsonify
from src.strategies.ema_cross.backtest_ema_cross import run_backtest_ema_cross
from src.core.plotting.plot_trades import plot_trades
from src.core.plotting.plot_equity import plot_equity

app = Flask(__name__)


@app.route("/run_backtest", methods=["POST"])
def run_backtest():

    # -------- Params base --------
    exchange = request.form.get("exchange")
    symbol = request.form.get("symbol")
    timeframe = request.form.get("timeframe")

    start_date = request.form.get("start_date")
    end_date = request.form.get("end_date")

    ema_fast = int(request.form.get("ema_fast"))
    ema_slow = int(request.form.get("ema_slow"))

    chart_type = request.form.get("chart_type")          # equity | trades
    time_mode = request.form.get("time_mode")            # full | monthly | range

    show_ema_fast = request.form.get("show_ema_fast") == "on"
    show_ema_slow = request.form.get("show_ema_slow") == "on"

    # -------- Run backtest --------
    stats, equity_chart, csv_path, df, trades = run_backtest_ema_cross(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        return_raw=True   # 👈 clave
    )

    chart_paths = []

    # -------- Indicators --------
    indicators = {}
    if show_ema_fast:
        indicators[f"EMA {ema_fast}"] = df["close"].ewm(span=ema_fast, adjust=False).mean()
    if show_ema_slow:
        indicators[f"EMA {ema_slow}"] = df["close"].ewm(span=ema_slow, adjust=False).mean()

    # -------- Chart logic --------
    if chart_type == "equity":
        chart_paths.append(equity_chart)

    elif chart_type == "trades":

        if time_mode == "full":
            path = plot_trades(
                df=df,
                trades=trades,
                indicators=indicators,
                title="Trades – Full Range"
            )
            chart_paths.append(path)

        elif time_mode == "range":
            path = plot_trades(
                df=df,
                trades=trades,
                indicators=indicators,
                start_date=start_date,
                end_date=end_date,
                title="Trades – Custom Range"
            )
            chart_paths.append(path)

        elif time_mode == "monthly":
            df["year_month"] = df["timestamp"].dt.to_period("M")

            for ym in df["year_month"].unique():
                month_df = df[df["year_month"] == ym]

                path = plot_trades(
                    df=month_df,
                    trades=trades,
                    indicators=indicators,
                    title=f"Trades – {ym}"
                )
                chart_paths.append(path)

    return jsonify({
        "stats": stats,
        "charts": chart_paths,
        "csv_path": csv_path
    })


if __name__ == "__main__":
    app.run(debug=True)
