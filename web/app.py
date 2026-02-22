import json
import os
import uuid

from datetime import datetime, timezone
from flask import Flask, render_template, request, redirect, url_for, jsonify, current_app
from src.core.database import get_connection, init_db
from src.strategies.ema_cross.backtest_ema_cross import run_backtest_ema_cross

app = Flask(__name__,template_folder="templates",static_folder="static")
init_db()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run_backtest", methods=["POST"])
def run_backtest():
    strategy = request.form.get("strategy")
    exchange = request.form.get("exchange")
    symbol = request.form.get("symbol")
    timeframe = request.form.get("timeframe")
    use_clean = request.form.get("use_clean") == "1"

    ema_fast = int(request.form.get("ema_fast"))
    ema_slow = int(request.form.get("ema_slow"))

    start_date = request.form.get("start_date")
    end_date = request.form.get("end_date")

    params = {
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "use_clean": use_clean
    }

    # 1) Generar run_id
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    run_id = f"{timestamp}_{unique_id}"

    #2) Ejecutar backtest
    stats, chart_path, csv_path = run_backtest_ema_cross(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        run_id=run_id
    )

    #3) guardad en DB
    created_at = now.isoformat()

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO backtest_runs (
            run_id, strategy, exchange, symbol, timeframe,
            start_ts, end_ts,
            params_json, stats_json,
            chart_path, csv_path, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?, ?)
    """, (
        run_id, strategy, exchange, symbol, timeframe,
        None, None,
        json.dumps(params),
        json.dumps(stats),
        chart_path,
        csv_path,
        created_at
    ))

    conn.commit()
    conn.close()

    return redirect(url_for("results", run_id=run_id))

@app.route("/results/<run_id>")
def results(run_id):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT * FROM backtest_runs WHERE run_id = ?", (run_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return "Run not found", 404

    stats = json.loads(row["stats_json"])
    params = json.loads(row["params_json"])

    return render_template(
        "results.html",
        run=row,
        stats=stats,
        params=params
    )


@app.route("/history")
def history():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT run_id, strategy, exchange, symbol, timeframe, created_at
        FROM backtest_runs
        ORDER BY id DESC
        LIMIT 50
    """)
    runs = cur.fetchall()

    conn.close()
    return render_template("history.html", runs=runs)

@app.route("/charts/<strategy>")
def view_charts(strategy):
    charts_dir = os.path.join(current_app.static_folder,"charts",strategy)
    print(f"Looking for charts in: {charts_dir}")  # Debugging line
    if not os.path.exists(charts_dir):
        return f"No charts found for strategy {strategy} - Looking for charts in: {charts_dir}"

    images = [
        f"charts/{strategy}/{file}"
        for file in os.listdir(charts_dir)
        if file.endswith(".png")
    ]

    images.sort(reverse=True)


    return render_template("charts.html", strategy=strategy, images=images)

if __name__ == "__main__":
    app.run(debug=True)
