"""Explicit strategy intents, independent of broker BUY/SELL sides."""

from enum import Enum
import json


class TradeAction(str, Enum):
    HOLD = "hold"
    OPEN_LONG = "open_long"
    OPEN_SHORT = "open_short"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"


ACTION_LABELS = {
    TradeAction.HOLD: "观望/持有",
    TradeAction.OPEN_LONG: "做多",
    TradeAction.OPEN_SHORT: "做空",
    TradeAction.CLOSE_LONG: "多头平仓",
    TradeAction.CLOSE_SHORT: "空头平仓",
}


def resolve_action(signal: int, position: float = 0, *, long_only: bool = True) -> TradeAction:
    """Resolve a raw direction against a known signed position; never reverse."""
    if position > 0:
        return TradeAction.CLOSE_LONG if signal == -1 else TradeAction.HOLD
    if position < 0:
        return TradeAction.CLOSE_SHORT if signal == 1 else TradeAction.HOLD
    if signal == 1:
        return TradeAction.OPEN_LONG
    if signal == -1 and not long_only:
        return TradeAction.OPEN_SHORT
    return TradeAction.HOLD


def action_label(result: dict) -> str:
    """Keep old signal-history rows readable without inventing position state."""
    result = restore_action(result)
    if "action" in result:
        return ACTION_LABELS.get(result["action"], "数据不可用")
    return {1: "买入", -1: "卖出", 0: "观望"}.get(result.get("signal"), "数据不可用")


def has_action(result: dict) -> bool:
    result = restore_action(result)
    if "action" in result:
        return result["action"] in {
            TradeAction.OPEN_LONG, TradeAction.OPEN_SHORT,
            TradeAction.CLOSE_LONG, TradeAction.CLOSE_SHORT,
        }
    return result.get("signal", 0) != 0


def restore_action(result: dict) -> dict:
    """Read optional research metadata from existing signal-history storage."""
    if "action" in result:
        return result
    indicators = result.get("indicators") or {}
    if isinstance(indicators, str):
        try:
            indicators = json.loads(indicators)
        except (TypeError, ValueError):
            return result
    if isinstance(indicators, dict) and "_action" in indicators:
        return {**result, "action": indicators["_action"],
                "action_reason": indicators.get("_action_reason", "")}
    return result
