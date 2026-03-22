import numpy as np


class BacktesterV2:
    def __init__(
        self,
        initial_capital=1000.0,
        position_mode="all_in",     # "all_in" | "fixed"
        trade_size=100.0,
        commission_pct=0.001,       # 0.1%
        slippage_pct=0.01,          # 1%
        allow_short=True,
        stop_loss_pct=0.02,
        take_profit_pct=None,
        atr_sl_mult=None,
        atr_tp_mult=None,
        atr_sl_mult_long=None,
        atr_sl_mult_short=None,
        pnl_mode="spot",
        contract_multiplier=1.0,
        commission_per_contract=None,
        pyramiding=1,
        position_pct=None,
    ):

        self.initial_capital = initial_capital
        self.cash = initial_capital

        self.position_mode = position_mode
        self.trade_size = trade_size
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.allow_short = allow_short
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.atr_sl_mult_long = atr_sl_mult_long
        self.atr_sl_mult_short = atr_sl_mult_short
        self.pnl_mode = pnl_mode
        self.contract_multiplier = contract_multiplier
        self.commission_per_contract = commission_per_contract
        self.pyramiding = max(int(pyramiding), 1)
        self.position_pct = position_pct

        self.position = None
        self.lots = []
        self.trades = []
        self.equity_curve = [initial_capital]

    @staticmethod
    def _normalize_trigger(trigger, fallback):
        if trigger is None:
            return fallback
        text = str(trigger).strip()
        if not text or text.lower() == "nan":
            return fallback
        return text

    # -------------------------------------------------
    # POSITION SIZE
    # -------------------------------------------------
    def _calculate_position_size(self, price=None):
        if self.position_pct is not None:
            if self.pnl_mode == "futures":
                if price is None:
                    return 0
                capital_per_entry = self.cash * self.position_pct
                contracts = int(capital_per_entry / (price * self.contract_multiplier))
                return max(contracts, 1)
            return self.cash * self.position_pct

        if self.position_mode == "all_in":
            return self.cash / self.pyramiding
        elif self.position_mode == "fixed":
            return min(self.trade_size, self.cash / self.pyramiding)
        elif self.position_mode == "contracts":
            return self.trade_size
        else:
            raise ValueError("Invalid position_mode")

    # -------------------------------------------------
    # OPEN TRADE
    # -------------------------------------------------
    def _open_trade(self, side, price, timestamp, trigger, bar_index, atr_value=None):

        position_size = self._calculate_position_size(price=price)

        if position_size <= 0:
            return

        if self.pnl_mode == "spot" and position_size * self.pyramiding > self.cash:
            return

        # Apply slippage
        if side == "LONG":
            entry_price = price * (1 + self.slippage_pct)
            if atr_value is not None and (
                self.atr_sl_mult_long is not None or self.atr_sl_mult is not None
            ):
                sl_mult = self.atr_sl_mult_long
                if sl_mult is None:
                    sl_mult = self.atr_sl_mult
                self.stop_price = entry_price - atr_value * sl_mult
            elif self.stop_loss_pct is not None:
                self.stop_price = entry_price * (1 - self.stop_loss_pct)
            else:
                self.stop_price = None
            if atr_value is not None and self.atr_tp_mult is not None:
                self.take_profit_price = entry_price + atr_value * self.atr_tp_mult
            elif self.take_profit_pct is not None:
                self.take_profit_price = entry_price * (1 + self.take_profit_pct)
            else:
                self.take_profit_price = None
        else:  # SHORT
            entry_price = price * (1 - self.slippage_pct)
            if atr_value is not None and (
                self.atr_sl_mult_short is not None or self.atr_sl_mult is not None
            ):
                sl_mult = self.atr_sl_mult_short
                if sl_mult is None:
                    sl_mult = self.atr_sl_mult
                self.stop_price = entry_price + atr_value * sl_mult
            elif self.stop_loss_pct is not None:
                self.stop_price = entry_price * (1 + self.stop_loss_pct)
            else:
                self.stop_price = None
            if atr_value is not None and self.atr_tp_mult is not None:
                self.take_profit_price = entry_price - atr_value * self.atr_tp_mult
            elif self.take_profit_pct is not None:
                self.take_profit_price = entry_price * (1 - self.take_profit_pct)
            else:
                self.take_profit_price = None
        entry_trigger = self._normalize_trigger(trigger, f"{side.lower()}_entry_signal")

        if self.pnl_mode == "futures":
            qty = float(position_size)
            if self.commission_per_contract is not None:
                commission_entry = qty * self.commission_per_contract
            else:
                notional = entry_price * self.contract_multiplier * qty
                commission_entry = notional * self.commission_pct
            self.cash -= commission_entry
        else:
            qty = position_size / entry_price
            commission_entry = position_size * self.commission_pct
            self.cash -= commission_entry

        if self.position is None:
            self.position = {
                "side": side,
                "stop_price": self.stop_price,
                "take_profit_price": self.take_profit_price,
            }
        else:
            if side == "LONG" and self.stop_price is not None:
                if self.position["stop_price"] is None:
                    self.position["stop_price"] = self.stop_price
                else:
                    self.position["stop_price"] = max(
                        self.position["stop_price"],
                        self.stop_price,
                    )
            elif side == "SHORT" and self.stop_price is not None:
                if self.position["stop_price"] is None:
                    self.position["stop_price"] = self.stop_price
                else:
                    self.position["stop_price"] = min(
                        self.position["stop_price"],
                        self.stop_price,
                    )

            if self.position["take_profit_price"] is None:
                self.position["take_profit_price"] = self.take_profit_price

        pyramid_level = len(self.lots) + 1
        lot = {
            "side": side,
            "entry_price": entry_price,
            "entry_time": timestamp,
            "entry_trigger": entry_trigger,
            "entry_index": bar_index,
            "position_size": position_size,
            "qty": qty,
            "commission_entry": commission_entry,
            "stop_price": self.stop_price,
            "take_profit_price": self.take_profit_price,
            "pyramid_level": pyramid_level,
        }
        self.lots.append(lot)

    def average_entry_price(self):
        if not self.lots:
            return None
        total_qty = sum(lot["qty"] for lot in self.lots)
        if total_qty == 0:
            return None
        weighted = sum(lot["qty"] * lot["entry_price"] for lot in self.lots)
        return weighted / total_qty

    def update_stop_from_avg(self, atr_value):
        if self.position is None or atr_value is None:
            return
        avg_price = self.average_entry_price()
        if avg_price is None:
            return
        side = self.position["side"]
        if side == "LONG":
            sl_mult = self.atr_sl_mult_long if self.atr_sl_mult_long is not None else self.atr_sl_mult
            if sl_mult is None:
                return
            self.position["stop_price"] = avg_price - atr_value * sl_mult
        elif side == "SHORT":
            sl_mult = self.atr_sl_mult_short if self.atr_sl_mult_short is not None else self.atr_sl_mult
            if sl_mult is None:
                return
            self.position["stop_price"] = avg_price + atr_value * sl_mult

    # -------------------------------------------------
    # CLOSE TRADE
    # -------------------------------------------------
    def _close_trade(self, price, timestamp, trigger, bar_index):
        if not self.lots:
            return

        side = self.position["side"]
        exit_trigger = self._normalize_trigger(trigger, f"{side.lower()}_exit_signal")

        if side == "LONG":
            exit_price = price * (1 - self.slippage_pct)
        else:
            exit_price = price * (1 + self.slippage_pct)

        for lot in self.lots:
            entry_price = lot["entry_price"]
            qty = lot["qty"]
            position_size = lot["position_size"]

            if self.pnl_mode == "futures":
                if side == "LONG":
                    gross_pnl = (exit_price - entry_price) * self.contract_multiplier * qty
                else:
                    gross_pnl = (entry_price - exit_price) * self.contract_multiplier * qty

                if self.commission_per_contract is not None:
                    commission_exit = qty * self.commission_per_contract
                else:
                    notional = exit_price * self.contract_multiplier * qty
                    commission_exit = notional * self.commission_pct
            else:
                if side == "LONG":
                    gross_pnl = (exit_price - entry_price) * qty
                else:
                    gross_pnl = (entry_price - exit_price) * qty
                commission_exit = (position_size + gross_pnl) * self.commission_pct

            total_commission = lot["commission_entry"] + commission_exit
            net_pnl = gross_pnl - total_commission
            bars_in_trade = bar_index - lot["entry_index"]

            self.cash += gross_pnl - commission_exit
            self.equity_curve.append(self.cash)

            self.trades.append({
                "entry_time": lot["entry_time"],
                "exit_time": timestamp,
                "side": side,
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "position_size": float(position_size),
                "qty": float(qty),
                "gross_pnl": float(gross_pnl),
                "commission_paid": float(total_commission),
                "net_pnl": float(net_pnl),
                "result": "WIN" if net_pnl > 0 else "LOSS",
                "entry_trigger": lot["entry_trigger"],
                "exit_trigger": exit_trigger,
                "bars_in_trade": bars_in_trade,
                "stop_price": self.position["stop_price"],
                "take_profit_price": self.position["take_profit_price"],
                "cash_after_trade": float(self.cash),
                "pyramid_level": lot["pyramid_level"],
            })

        self.position = None
        self.lots = []

    # -------------------------------------------------
    # SIGNAL HANDLER
    # -------------------------------------------------
    def on_signal(self, signal, price, timestamp, trigger=None, bar_index=None, atr_value=None):

        if signal == "LONG":

            if self.position is None:
                self._open_trade(
                    side="LONG",
                    price=price,
                    timestamp=timestamp,
                    trigger=trigger,
                    bar_index=bar_index,
                    atr_value=atr_value,
                )

            elif self.position["side"] == "SHORT":
                self._close_trade(price, timestamp, trigger, bar_index)
                self._open_trade(
                    side="LONG",
                    price=price,
                    timestamp=timestamp,
                    trigger=trigger,
                    bar_index=bar_index,
                    atr_value=atr_value,
                )
            elif self.position["side"] == "LONG":
                if len(self.lots) < self.pyramiding:
                    self._open_trade(
                        side="LONG",
                        price=price,
                        timestamp=timestamp,
                        trigger=trigger,
                        bar_index=bar_index,
                        atr_value=atr_value,
                    )

        elif signal == "SHORT":

            if not self.allow_short:
                return

            if self.position is None:
                self._open_trade(
                    side="SHORT",
                    price=price,
                    timestamp=timestamp,
                    trigger=trigger,
                    bar_index=bar_index,
                    atr_value=atr_value,
                )

            elif self.position["side"] == "LONG":
                self._close_trade(price, timestamp, trigger, bar_index)
                self._open_trade(
                    side="SHORT",
                    price=price,
                    timestamp=timestamp,
                    trigger=trigger,
                    bar_index=bar_index,
                    atr_value=atr_value,
                )
            elif self.position["side"] == "SHORT":
                if len(self.lots) < self.pyramiding:
                    self._open_trade(
                        side="SHORT",
                        price=price,
                        timestamp=timestamp,
                        trigger=trigger,
                        bar_index=bar_index,
                        atr_value=atr_value,
                    )

        elif signal == "EXIT":

            if self.position is not None:
                self._close_trade(
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

        side = self.position["side"]
        stop_price = self.position["stop_price"]
        take_profit_price = self.position["take_profit_price"]

        # LONG
        if side == "LONG":
            if stop_price is not None and low <= stop_price:
                self._close_trade(
                    price=stop_price,
                    timestamp=timestamp,
                    trigger="stop_loss",
                    bar_index=bar_index
                )
                return

            if take_profit_price is not None and high >= take_profit_price:
                self._close_trade(
                    price=take_profit_price,
                    timestamp=timestamp,
                    trigger="take_profit",
                    bar_index=bar_index
                )
                return

        # SHORT
        if side == "SHORT":
            if stop_price is not None and high >= stop_price:
                self._close_trade(
                    price=stop_price,
                    timestamp=timestamp,
                    trigger="stop_loss",
                    bar_index=bar_index
                )
                return

            if take_profit_price is not None and low <= take_profit_price:
                self._close_trade(
                    price=take_profit_price,
                    timestamp=timestamp,
                    trigger="take_profit",
                    bar_index=bar_index
                )
                return

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

        if len(equity) > 1:
            x = np.arange(len(equity))
            equity_slope = np.polyfit(x, equity, 1)[0]
        else:
            equity_slope = 0.0

        if len(net_pnls) > 1:
            x = np.arange(len(net_pnls))
            trade_pnl_slope = np.polyfit(x, net_pnls, 1)[0]
        else:
            trade_pnl_slope = 0.0

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
