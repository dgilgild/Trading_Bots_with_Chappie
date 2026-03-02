import numpy as np


class BacktesterV2:
    def __init__(
        self,
        initial_capital=1000,
        position_mode="all_in",     # "all_in" | "fixed"
        trade_size=100,
        commission_pct=0.001,       # 0.1%
        slippage_pct=0.01,          # 1%
        allow_short=False,
        stop_loss_pct=0.02
    ):

        self.initial_capital = initial_capital
        self.cash = initial_capital

        self.position_mode = position_mode
        self.trade_size = trade_size
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.allow_short = allow_short
        self.stop_loss_pct = stop_loss_pct

        self.position = None
        self.trades = []
        self.equity_curve = [initial_capital]

    # -------------------------------------------------
    # POSITION SIZE
    # -------------------------------------------------
    def _calculate_position_size(self):
        if self.position_mode == "all_in":
            return self.cash
        elif self.position_mode == "fixed":
            return min(self.trade_size, self.cash)
        else:
            raise ValueError("Invalid position_mode")

    # -------------------------------------------------
    # OPEN TRADE
    # -------------------------------------------------
    def _open_trade(self, side, price, timestamp, trigger, bar_index):

        position_size = self._calculate_position_size()

        if position_size <= 0:
            return

        # Apply slippage
        if side == "LONG":
            entry_price = price * (1 + self.slippage_pct)
            if self.stop_loss_pct is not None:
                self.stop_price = entry_price * (1 - self.stop_loss_pct)
            else:
                self.stop_price = None
        else:  # SHORT
            entry_price = price * (1 - self.slippage_pct)
            if self.stop_loss_pct is not None:
                self.stop_price = entry_price * (1 + self.stop_loss_pct)
            else:
                self.stop_price = None
        qty = position_size / entry_price

        commission_entry = position_size * self.commission_pct
        self.cash -= commission_entry

        self.position = {
            "side": side,
            "entry_price": entry_price,
            "entry_time": timestamp,
            "entry_trigger": trigger,
            "entry_index": bar_index,
            "position_size": position_size,
            "qty": qty,
            "commission_entry": commission_entry,
            "stop_price": self.stop_price,  # 👈 agregar
        }

    # -------------------------------------------------
    # CLOSE TRADE
    # -------------------------------------------------
    def _close_trade(self, price, timestamp, trigger, bar_index):

        side = self.position["side"]
        entry_price = self.position["entry_price"]
        qty = self.position["qty"]
        position_size = self.position["position_size"]

        # Apply slippage
        if side == "LONG":
            exit_price = price * (1 - self.slippage_pct)
            gross_pnl = (exit_price - entry_price) * qty
        else:  # SHORT
            exit_price = price * (1 + self.slippage_pct)
            gross_pnl = (entry_price - exit_price) * qty

        commission_exit = (position_size + gross_pnl) * self.commission_pct
        total_commission = (
            self.position["commission_entry"] + commission_exit
        )

        net_pnl = gross_pnl - commission_exit

        bars_in_trade = bar_index - self.position["entry_index"]

        self.cash += net_pnl
        self.equity_curve.append(self.cash)

        self.trades.append({
            "entry_time": self.position["entry_time"],
            "exit_time": timestamp,
            "side": side,
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "position_size": float(position_size),
            "gross_pnl": float(gross_pnl),
            "commission_paid": float(total_commission),
            "net_pnl": float(net_pnl),
            "result": "WIN" if net_pnl > 0 else "LOSS",
            "entry_trigger": self.position["entry_trigger"],
            "exit_trigger": trigger,
            "bars_in_trade": bars_in_trade,
        })

        self.position = None

    # -------------------------------------------------
    # SIGNAL HANDLER
    # -------------------------------------------------
    def on_signal(self, signal, price, timestamp, trigger=None, bar_index=None):

        if signal == "LONG":

            if self.position is None:
                self._open_trade(
                    side="LONG",
                    price=price,
                    timestamp=timestamp,
                    trigger=trigger,
                    bar_index=bar_index
                )

            elif self.position["side"] == "SHORT":
                self._close_trade(price, timestamp, trigger, bar_index)
                self._open_trade(
                    side="LONG",
                    price=price,
                    timestamp=timestamp,
                    trigger=trigger,
                    bar_index=bar_index
                )

        elif signal == "EXIT":

            if self.position is not None:
                self._close_trade(
                    price=price,
                    timestamp=timestamp,
                    trigger=trigger,
                    bar_index=bar_index
                )

                if self.allow_short:
                    self._open_trade(
                        side="SHORT",
                        price=price,
                        timestamp=timestamp,
                        trigger=trigger,
                        bar_index=bar_index
                    )

    # -------------------------------------------------
    # BAR UPDATE (STOP CHECK)
    # -------------------------------------------------
    def on_bar(self, high, low, timestamp, bar_index):

        if self.position is None:
            return

        if self.stop_loss_pct is None:
            return

        side = self.position["side"]
        stop_price = self.position["stop_price"]

        # LONG STOP
        if side == "LONG" and low <= stop_price:
            self._close_trade(
                price=stop_price,
                timestamp=timestamp,
                trigger="stop_loss",
                bar_index=bar_index
            )

        # SHORT STOP
        elif side == "SHORT" and high >= stop_price:
            self._close_trade(
                price=stop_price,
                timestamp=timestamp,
                trigger="stop_loss",
                bar_index=bar_index
            )

    # -------------------------------------------------
    # STATS (compatible)
    # -------------------------------------------------
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