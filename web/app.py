from flask import Flask, render_template, request, redirect, url_for, jsonify
import json
from datetime import datetime, timezone

from src.core.database import get_connection, init_db
from src.strategies.ema_cross.backtest_ema_cross import run_backtest_ema_cross

app = Flask(__name__)
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

    stats, chart_path, csv_path = run_backtest_ema_cross(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        ema_fast=ema_fast,
        ema_slow=ema_slow
    )

    return jsonify({
        "stats": stats,
        "chart_path": chart_path,
        "csv_path": csv_path
    })


    created_at = datetime.now(timezone.utc).isoformat()

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO backtest_runs (
            strategy, exchange, symbol, timeframe,
            start_ts, end_ts,
            params_json, stats_json,
            chart_path, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        strategy, exchange, symbol, timeframe,
        None, None,
        json.dumps(params),
        json.dumps(stats),
        chart_path,
        created_at
    ))

    run_id = cur.lastrowid
    conn.commit()
    conn.close()

    return redirect(url_for("results", run_id=run_id))


@app.route("/results/<int:run_id>")
def results(run_id):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT * FROM backtest_runs WHERE id = ?", (run_id,))
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
        SELECT id, strategy, exchange, symbol, timeframe, created_at
        FROM backtest_runs
        ORDER BY id DESC
        LIMIT 50
    """)
    runs = cur.fetchall()

    conn.close()
    return render_template("history.html", runs=runs)


if __name__ == "__main__":
    app.run(debug=True)
