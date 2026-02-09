import os
import matplotlib.pyplot as plt
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

    df_plot = df.copy()
    df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp"])
    df_plot.set_index("timestamp", inplace=True)

    if start_date:
        df_plot = df_plot[df_plot.index >= pd.to_datetime(start_date)]
    if end_date:
        df_plot = df_plot[df_plot.index <= pd.to_datetime(end_date)]

    plt.figure(figsize=figsize)

    # Precio
    plt.plot(df_plot.index, df_plot["close"], label="Close", alpha=0.6)

    # Indicadores genéricos
    if indicators:
        for name, series in indicators.items():
            plt.plot(series.index, series.values, label=name, linewidth=1)

    # Trades
    for t in trades:
        entry = pd.to_datetime(t["entry_time"])
        exit_ = pd.to_datetime(t["exit_time"])

        if entry not in df_plot.index:
            continue

        plt.scatter(entry, t["entry_price"], marker="^", color="green", s=70)
        plt.scatter(exit_, t["exit_price"], marker="v", color="red", s=70)

        plt.plot(
            [entry, exit_],
            [t["entry_price"], t["exit_price"]],
            color="gray",
            alpha=0.4
        )

    plt.title(title)
    plt.legend()
    plt.grid(True)

    filename = f"trades_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
    path = os.path.join(output_dir, filename)

    plt.savefig(path)
    plt.close()

    return f"/static/charts/trades/{filename}"
