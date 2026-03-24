import os
import math
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd


def plot_trades(
    df,
    trades,
    indicators=None,
    start_date=None,
    end_date=None,
    title="Trades Overlay",
    output_dir="web/static/charts/trades",
    figsize=(20, 8)
):
    """
    df: DataFrame con ['timestamp','open','high','low','close']
    trades: lista de trades del Backtester
    indicators: dict opcional {'EMA_20': series, 'EMA_50': series}
    """

    os.makedirs(output_dir, exist_ok=True)
    filename = f"trades_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
    path = os.path.join(output_dir, filename)

    _, chart_created = plot_trades_candlestick(
        df=df,
        trades=trades,
        indicators=indicators,
        start_date=start_date,
        end_date=end_date,
        title=title,
        output_path=path,
        figsize=figsize,
    )

    if not chart_created:
        return None

    return f"/static/charts/trades/{filename}"


def _normalize_trigger(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text.replace("_", " ").title()


def _normalize_trades(trades):
    if isinstance(trades, pd.DataFrame):
        trades_df = trades.copy()
    else:
        trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        return trades_df

    for col in ["entry_time", "exit_time"]:
        if col in trades_df.columns:
            ts = pd.to_datetime(trades_df[col], utc=True, errors="coerce")
            trades_df[col] = ts.dt.tz_convert(None)

    return trades_df


def _prepare_ohlc(df, start_date=None, end_date=None):
    required_cols = {"timestamp", "open", "high", "low", "close"}
    if df is None or df.empty or not required_cols.issubset(df.columns):
        return None

    df_plot = df.copy()
    df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp"], errors="coerce")
    df_plot = df_plot.dropna(subset=["timestamp"]).sort_values("timestamp")
    df_plot.set_index("timestamp", inplace=True)

    if start_date:
        df_plot = df_plot[df_plot.index >= pd.to_datetime(start_date)]
    if end_date:
        df_plot = df_plot[df_plot.index <= pd.to_datetime(end_date)]

    if df_plot.empty:
        return None

    return df_plot


def _plot_single_candlestick_chart(
    df_plot,
    trades_df,
    indicators=None,
    title="Trades (Candlestick)",
    output_path=None,
    figsize=(20, 8),
):
    fig, ax = plt.subplots(figsize=figsize)

    x_values = mdates.date2num(df_plot.index.to_pydatetime())
    if len(x_values) > 1:
        candle_width = (x_values[1] - x_values[0]) * 0.7
    else:
        candle_width = 0.02

    for idx, row in zip(x_values, df_plot.itertuples()):
        color = "#2ca02c" if row.close >= row.open else "#d62728"
        ax.vlines(idx, row.low, row.high, color=color, linewidth=1, alpha=0.9)

        body_bottom = min(row.open, row.close)
        body_height = max(abs(row.close - row.open), 1e-9)
        rect = Rectangle(
            (idx - candle_width / 2, body_bottom),
            candle_width,
            body_height,
            facecolor=color,
            edgecolor=color,
            alpha=0.85,
        )
        ax.add_patch(rect)

    if indicators:
        window_start = df_plot.index.min()
        window_end = df_plot.index.max()
        for name, series in indicators.items():
            if series is None:
                continue
            series_to_plot = series.copy()
            if not isinstance(series_to_plot.index, pd.DatetimeIndex):
                series_to_plot.index = pd.to_datetime(series_to_plot.index, errors="coerce")
            series_to_plot = series_to_plot.dropna()
            series_to_plot = series_to_plot[
                (series_to_plot.index >= window_start) & (series_to_plot.index <= window_end)
            ]
            if not series_to_plot.empty:
                ax.plot(series_to_plot.index, series_to_plot.values, label=name, linewidth=1)

    has_full_conditions = True
    first_entry_label = True
    first_exit_label = True
    first_path_label = True
    has_markers = False
    window_start = df_plot.index.min()
    window_end = df_plot.index.max()

    for trade in trades_df.to_dict("records"):
        entry_time = trade.get("entry_time")
        exit_time = trade.get("exit_time")
        entry_price = trade.get("entry_price")
        exit_price = trade.get("exit_price")

        if (
            pd.isna(entry_time)
            or pd.isna(exit_time)
            or entry_price is None
            or exit_price is None
            or pd.isna(entry_price)
            or pd.isna(exit_price)
        ):
            continue

        entry_time = pd.to_datetime(entry_time)
        exit_time = pd.to_datetime(exit_time)

        entry_in_window = window_start <= entry_time <= window_end
        exit_in_window = window_start <= exit_time <= window_end
        if not entry_in_window and not exit_in_window:
            continue

        side = str(trade.get("side", "")).upper()
        entry_marker = "^" if side != "SHORT" else "v"
        entry_color = "#1f77b4" if side != "SHORT" else "#ff7f0e"
        net_pnl = pd.to_numeric(trade.get("net_pnl", 0.0), errors="coerce")
        if pd.isna(net_pnl):
            net_pnl = 0.0
        trade_line_color = "#17becf" if net_pnl >= 0 else "#9467bd"

        entry_trigger = _normalize_trigger(trade.get("entry_trigger"))
        exit_trigger = _normalize_trigger(trade.get("exit_trigger"))

        if entry_in_window and not entry_trigger:
            has_full_conditions = False
        if exit_in_window and not exit_trigger:
            has_full_conditions = False

        if entry_in_window:
            ax.scatter(
                entry_time,
                entry_price,
                marker=entry_marker,
                color=entry_color,
                s=70,
                zorder=4,
                label="Entry" if first_entry_label else None,
            )
            first_entry_label = False
            has_markers = True

            entry_label = entry_trigger or f"{side or 'Trade'} entry"
            ax.annotate(
                f"E: {entry_label}",
                (entry_time, entry_price),
                textcoords="offset points",
                xytext=(6, 8),
                fontsize=7,
                color=entry_color,
                alpha=0.95,
            )

        if exit_in_window:
            ax.scatter(
                exit_time,
                exit_price,
                marker="X",
                color="#d62728",
                s=65,
                zorder=4,
                label="Exit" if first_exit_label else None,
            )
            first_exit_label = False
            has_markers = True

            exit_label = exit_trigger or "Exit"
            ax.annotate(
                f"X: {exit_label}",
                (exit_time, exit_price),
                textcoords="offset points",
                xytext=(6, -12),
                fontsize=7,
                color="#b22222",
                alpha=0.95,
            )

        if entry_in_window and exit_in_window:
            ax.plot(
                [entry_time, exit_time],
                [entry_price, exit_price],
                color=trade_line_color,
                alpha=0.45,
                linewidth=1.0,
                label="Trade Path" if first_path_label else None,
            )
            first_path_label = False

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="best")

    fig.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)

    annotation_mode = "full" if has_full_conditions else "best_effort"
    return annotation_mode, True, has_markers


def plot_trades_candlestick(
    df,
    trades,
    indicators=None,
    start_date=None,
    end_date=None,
    title="Trades (Candlestick)",
    output_path=None,
    figsize=(20, 8),
):
    df_plot = _prepare_ohlc(df, start_date=start_date, end_date=end_date)
    if df_plot is None:
        return "missing_ohlc", False

    trades_df = _normalize_trades(trades)
    if trades_df.empty:
        return "missing_trades", False
    annotation_mode, chart_created, _ = _plot_single_candlestick_chart(
        df_plot=df_plot,
        trades_df=trades_df,
        indicators=indicators,
        title=title,
        output_path=output_path,
        figsize=figsize,
    )

    return annotation_mode, chart_created


def plot_trades_candlestick_windows(
    df,
    trades,
    indicators=None,
    start_date=None,
    end_date=None,
    title="Trades (Candlestick)",
    output_dir=None,
    filename_prefix="trades_graph",
    candles_per_chart=50,
    max_charts=10,
    figsize=(20, 8),
):
    df_plot = _prepare_ohlc(df, start_date=start_date, end_date=end_date)
    if df_plot is None:
        return "missing_ohlc", [], False, 0

    trades_df = _normalize_trades(trades)
    if trades_df.empty:
        return "missing_trades", [], False, 0

    window_size = max(int(candles_per_chart), 1)
    total_windows = int(math.ceil(len(df_plot) / window_size))
    chart_items = []
    has_full_conditions = True
    trade_windows = 0
    reached_cap = False

    for window_idx in range(total_windows):
        start_idx = window_idx * window_size
        end_idx = start_idx + window_size
        window_df = df_plot.iloc[start_idx:end_idx]
        if window_df.empty:
            continue

        window_start = window_df.index.min()
        window_end = window_df.index.max()
        has_trade_markers = (
            ((trades_df["entry_time"] >= window_start) & (trades_df["entry_time"] <= window_end))
            | ((trades_df["exit_time"] >= window_start) & (trades_df["exit_time"] <= window_end))
        ).any()
        if not has_trade_markers:
            continue

        trade_windows += 1
        if len(chart_items) >= max_charts:
            reached_cap = True
            break

        chart_filename = f"{filename_prefix}_w{len(chart_items) + 1:02d}.png"
        chart_path = os.path.join(output_dir, chart_filename) if output_dir else None
        window_title = (
            f"{title}\n"
            f"Window {trade_windows} | Candles {start_idx + 1}-{min(end_idx, len(df_plot))}"
        )
        annotation_mode, _, _ = _plot_single_candlestick_chart(
            df_plot=window_df,
            trades_df=trades_df,
            indicators=indicators,
            title=window_title,
            output_path=chart_path,
            figsize=figsize,
        )

        if annotation_mode == "best_effort":
            has_full_conditions = False

        chart_items.append(
            {
                "filename": chart_filename,
                "window_label": (
                    f"Window {trade_windows}: {window_start.strftime('%Y-%m-%d %H:%M')}"
                    f" - {window_end.strftime('%Y-%m-%d %H:%M')}"
                ),
            }
        )

    if not chart_items:
        return "missing_trades", [], False, trade_windows

    annotation_mode = "full" if has_full_conditions else "best_effort"
    return annotation_mode, chart_items, reached_cap, trade_windows
