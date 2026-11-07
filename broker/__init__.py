from typing import TYPE_CHECKING

from .base import (
    Account,
    Broker,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from .mock import MockBroker

if TYPE_CHECKING:
    from .futu import FutuBroker


def __getattr__(name):
    if name == "FutuBroker":
        from .futu import FutuBroker

        globals()[name] = FutuBroker
        return FutuBroker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Broker",
    "MockBroker",
    "FutuBroker",
    "Order",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "Position",
    "Account",
]
