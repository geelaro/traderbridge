"""Close-price research replay; no orders, sizing, fees or broker state."""

from copy import deepcopy
from math import isfinite

import pandas as pd

from utils.trade_actions import TradeAction, resolve_action


def replay_actions(strategy, df: pd.DataFrame) -> pd.DataFrame:
    """Resolve four intents using strategy exits and a hypothetical position.

    A close and a new entry cannot occur on the same bar. Repeated entry
    signals do not reset the entry date. Replay a private strategy copy so
    stateful exit implementations cannot affect a later engine run.
    """
    strategy = deepcopy(strategy)
    position = 0
    entry_index = None
    entry_price = highest = lowest = 0.0
    records = []
    for i in range(len(df)):
        action, reason = TradeAction.HOLD, ""
        before = position
        price = float(df["Close"].iloc[i])
        valid = isfinite(price) and price > 0
        if valid and i >= strategy.min_bars:
            if position:
                highest = max(highest, price)
                lowest = min(lowest, price)
                exit_now, reason = strategy.check_exit(
                    df, i, entry_price, highest,
                    lowest_since_entry=lowest,
                    position={"direction": "LONG" if position > 0 else "SHORT",
                              "date": df.index[entry_index], "price": entry_price},
                )
                if exit_now:
                    action = TradeAction.CLOSE_LONG if position > 0 else TradeAction.CLOSE_SHORT
                    position, entry_index = 0, None
            else:
                action = resolve_action(
                    strategy.entry_signal(df, i), long_only=strategy.long_only,
                )
                if action in (TradeAction.OPEN_LONG, TradeAction.OPEN_SHORT):
                    position = 1 if action == TradeAction.OPEN_LONG else -1
                    entry_index = i
                    entry_price = highest = lowest = price
                    reason = "entry_signal"
        records.append({
            "action": action.value if valid else None,
            "action_reason": reason if valid else "invalid_price",
            "position_before": before,
            "position_after": position,
            "entry_index": entry_index,
        })
    return pd.DataFrame(records, index=df.index, columns=[
        "action", "action_reason", "position_before", "position_after", "entry_index",
    ])
