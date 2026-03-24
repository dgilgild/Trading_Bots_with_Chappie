def check_signal(rsi_value, entry_level, exit_level, current_side=None):
    if rsi_value is None:
        return None, None

    if rsi_value < entry_level:
        return "LONG", f"RSI {rsi_value:.2f} below {entry_level}"

    if current_side == "LONG" and rsi_value > exit_level:
        return "EXIT", f"RSI {rsi_value:.2f} above {exit_level}"

    return None, None
