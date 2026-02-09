import matplotlib.pyplot as plt
import pandas as pd


def plot_trades_by_date(
    df,
    trades,
    start_date,
    end_date,
    title="EMA Cross Trades"
):
    # -----------------------------
    # Datetime handling (TU CASO)
    # -----------------------------
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    df_plot = df.loc[start_date:end_date]

    if df_plot.empty:
        print("⚠️ No hay datos en ese rango de fechas")
        return

    # -----------------------------
    # Base chart
    # -----------------------------
    plt.figure(figsize=(18, 9))

    plt.plot(
        df_plot.index,
        df_plot["close"],
        label="Close",
        linewidth=1,
        alpha=0.9
    )

    plt.plot(
        df_plot.index,
        df_plot["ema_fast"],
        label="EMA Fast",
        linestyle="--",
        linewidth=1
    )

    plt.plot(
        df_plot.index,
        df_plot["ema_slow"],
        label="EMA Slow",
        linestyle="--",
        linewidth=1
    )

    # -----------------------------
    # Trades overlay
    # -----------------------------
    for t in trades:
        entry_time = pd.to_datetime(t["entry_time"])
        exit_time = pd.to_datetime(t["exit_time"])

        if entry_time < start_date or entry_time > end_date:
            continue

        # Entry
        plt.scatter(
            entry_time,
            t["entry_price"],
            marker="^",
            color="green",
            s=90,
            zorder=5,
            label="Entry" if "Entry" not in plt.gca().get_legend_handles_labels()[1] else ""
        )

        # Exit
        plt.scatter(
            exit_time,
            t["exit_price"],
            marker="v",
            color="red",
            s=90,
            zorder=5,
            label="Exit" if "Exit" not in plt.gca().get_legend_handles_labels()[1] else ""
        )

        # Trade line
        color = "green" if t["net_pnl"] > 0 else "red"

        plt.plot(
            [entry_time, exit_time],
            [t["entry_price"], t["exit_price"]],
            color=color,
            alpha=0.35,
            linewidth=1
        )

    # -----------------------------
    # Final touches
    # -----------------------------
    plt.title(f"{title}\n{start_date.date()} → {end_date.date()}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
