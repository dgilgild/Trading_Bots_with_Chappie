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

        self.position = None   # trade abierto (dict)
        self.trades = []

        self.equity_curve = [initial_capital]

        self.commission = commission_per_trade
        self.slippage = slippage_per_trade

    # -----------------------------
    # OPEN TRADE
    # -----------------------------
    def _open_trade(self, side, price, timestamp, trigger, bar_index):
        self.position = {
            "side": side,
            "entry_price": price,
            "entry_time": timestamp,
            "entry_trigger": trigger,
            "entry_index": bar_index,
        }
    
    # -----------------------------
    # CLOSE TRADE
    # -----------------------------
    def _close_trade(self, price, timestamp, trigger, bar_index):
        entry_price = self.position["entry_price"]

        gross_pnl = price - entry_price
        costs = self.commission + self.slippage
        net_pnl = gross_pnl - costs
        pnl_pct = net_pnl / entry_price * 100 if entry_price != 0 else 0

        bars_in_trade = bar_index - self.position["entry_index"]

        self.capital += net_pnl

        self.trades.append({
            "entry_time": self.position["entry_time"],
            "exit_time": timestamp,
            "side": self.position["side"],
            "entry_price": float(entry_price),
            "exit_price": float(price),
            "result": "WIN" if net_pnl > 0 else "LOSS",
            "net_pnl": float(net_pnl),
            "pnl_pct": float(pnl_pct),
            "entry_trigger": self.position["entry_trigger"],
            "exit_trigger": trigger,
            "bars_in_trade": bars_in_trade,
        })

        self.equity_curve.append(self.capital)
        self.position = None

    # -----------------------------
    # SIGNAL HANDLER
    # -----------------------------
    def on_signal(self, signal, price, timestamp, trigger=None, bar_index=None):

        if signal == "LONG" and self.position is None:
            self._open_trade(
                side="LONG",
                price=price,
                timestamp=timestamp,
                trigger=trigger,
                bar_index=bar_index
            )

        elif signal == "EXIT" and self.position is not None:
            self._close_trade(
                price=price,
                timestamp=timestamp,
                trigger=trigger,
                bar_index=bar_index
            )

    # -----------------------------
    # STATS
    # -----------------------------
    def stats(self):
        if not self.trades:
            return {}

        net_pnls = np.array([t["net_pnl"] for t in self.trades])

        wins = net_pnls[net_pnls > 0]
        losses = net_pnls[net_pnls < 0]

        total_trades = len(net_pnls)
        total_net_profit = float(net_pnls.sum())

        profit_factor = (
            wins.sum() / abs(losses.sum())
            if len(losses) > 0 else np.inf
        )

        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0

        win_rate = len(wins) / total_trades
        loss_rate = len(losses) / total_trades

        tharp_expectancy = (
            (avg_win * win_rate + avg_loss * loss_rate)
            / abs(avg_loss) if avg_loss != 0 else np.nan
        )

        equity = np.array(self.equity_curve)
        peaks = np.maximum.accumulate(equity)
        drawdowns = (equity - peaks) / peaks

        x = np.arange(len(equity))
        equity_slope = np.polyfit(x, equity, 1)[0]

        x = np.arange(len(net_pnls))
        trade_pnl_slope = np.polyfit(x, net_pnls, 1)[0]

        return {
            "Total trades": total_trades,
            "Total Net Profit": total_net_profit,
            "Profit Factor": profit_factor,
            "Avg Trade Net Profit": avg_win,
            "Avg Trade Net Loss": avg_loss,
            "Tharp Expectancy": float(tharp_expectancy),
            "Max Drawdown (%)": float(drawdowns.min() * 100),
            "Equity Curve Slope": float(equity_slope),
            "Trade Net Profit Slope": float(trade_pnl_slope),
        }
