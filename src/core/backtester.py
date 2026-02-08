import numpy as np

class Backtester:
    def __init__(
        self,
        initial_capital=1000,
        commission_per_trade=1.0,
        slippage_per_trade=1.0
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = None

        self.entry_price = None
        self.entry_time = None

        self.trades = []
        self.equity_curve = [initial_capital]

        self.commission = commission_per_trade
        self.slippage = slippage_per_trade

    def on_signal(self, signal, price, timestamp):
        print(f"{timestamp} | SIGNAL: {signal} @ {price}")

        if signal == "LONG" and self.position is None:
            self.position = "LONG"
            self.entry_price = price
            self.entry_time = timestamp
            print("   EXECUTED BUY")

        elif signal == "EXIT" and self.position == "LONG":
            exit_price = price

            gross_pnl = exit_price - self.entry_price
            costs = self.commission + self.slippage
            net_pnl = gross_pnl - costs

            self.capital += net_pnl
            #self.equity_curve.append(self.capital)

            self.trades.append({
                "entry_time": self.entry_time,
                "exit_time": timestamp,
                "entry_price": self.entry_price,
                "exit_price": exit_price,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl
            })

            print(f"   EXECUTED SELL | Net PnL: {net_pnl:.2f}")

            self.position = None
            self.entry_price = None
            self.entry_time = None

        self.equity_curve.append(self.capital)


    def stats(self):
        if not self.trades:
            return {}

        net_pnls = np.array([t["net_pnl"] for t in self.trades])
        wins = net_pnls[net_pnls > 0]
        losses = net_pnls[net_pnls < 0]

        total_trades = len(net_pnls)
        total_net_profit = net_pnls.sum()

        profit_factor = (
            wins.sum() / abs(losses.sum())
            if len(losses) > 0 else np.inf
        )

        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0

        win_rate = len(wins) / total_trades
        loss_rate = len(losses) / total_trades

        # Tharp Expectancy (normalizada)
        tharp_expectancy = (
            (avg_win * win_rate + avg_loss * loss_rate)
            / abs(avg_loss) if avg_loss != 0 else np.nan
        )

        # Max Drawdown
        equity = np.array(self.equity_curve)
        peaks = np.maximum.accumulate(equity)
        drawdowns = (equity - peaks) / peaks
        max_drawdown = drawdowns.min()

        # Equity Curve Slope (regresión lineal)
        x = np.arange(len(equity))
        slope = np.polyfit(x, equity, 1)[0]

        # -----------------------------
        # Trade Net Profit Slope
        # -----------------------------
        net_pnls = np.array([t["net_pnl"] for t in self.trades])

        x = np.arange(len(net_pnls))
        trade_pnl_slope = np.polyfit(x, net_pnls, 1)[0]

        return {
            "Total trades": total_trades,
            "Total Net Profit": total_net_profit,
            "Profit Factor": profit_factor,
            "Avg Trade Net Profit": avg_win,
            "Avg Trade Net Loss": avg_loss,
            "Tharp Expectancy": tharp_expectancy,
            "Max Drawdown (%)": max_drawdown * 100,
            "Equity Curve Slope": slope,
            "Trade Net Profit Slope": trade_pnl_slope
        }
