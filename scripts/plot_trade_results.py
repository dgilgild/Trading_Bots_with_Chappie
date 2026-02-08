import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def plot_trade_net_profits(trades, strategy_name):
    base_dir = "web/static/charts"
    output_dir = os.path.join(base_dir, strategy_name)
    os.makedirs(output_dir, exist_ok=True)

    net_pnls = [t["net_pnl"] for t in trades]
    x = np.arange(len(net_pnls))
    y = np.array(net_pnls)

    slope, intercept = np.polyfit(x, y, 1)
    regression = slope * x + intercept

    plt.figure()
    plt.scatter(x, y)
    plt.plot(x, regression)
    plt.axhline(0)

    plt.xlabel("Trade Number")
    plt.ylabel("Net Profit per Trade")
    plt.title("Trade Net Profit Scatter + Regression")

    output_path = os.path.join(output_dir, "trade_net_profit.png")
    plt.savefig(output_path)
    plt.close()

    return output_path

