"""Broker abstraction — common data types and the abstract Broker interface.

Every broker adapter (Mock, QMT, Futu, IB, etc.) MUST implement this interface.
The LiveTrader only depends on this file — never on a concrete broker.

Order lifecycle
--------------
  PENDING → SUBMITTED → PARTIAL → FILLED
                    ↘ REJECTED
                    ↘ CANCELLED

Position tracking
-----------------
  The broker is the source of truth for positions.
  LiveTrader compares broker positions against strategy signals.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(str, Enum):
    PENDING = "PENDING"       # created locally, not yet sent
    SUBMITTED = "SUBMITTED"   # sent to broker, awaiting ack
    PARTIAL = "PARTIAL"       # partially filled
    FILLED = "FILLED"         # fully filled
    REJECTED = "REJECTED"     # broker rejected
    CANCELLED = "CANCELLED"   # cancelled


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Order:
    """Standardised order — returned by submit_order, tracked by LiveTrader."""

    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    order_id: str = ""               # set by broker on submit; auto-generated fallback
    price: Optional[float] = None    # None for market orders
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: int = 0
    avg_fill_price: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = ""
    broker_data: dict = field(default_factory=dict)  # raw broker response


@dataclass
class Position:
    """Current holding in one symbol."""

    symbol: str
    quantity: int           # >0 = long, <0 = short
    avg_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float = 0.0


@dataclass
class Account:
    """Account-level snapshot."""

    total_equity: float       # 总权益
    available_cash: float     # 可用资金
    frozen_margin: float      # 占用保证金
    total_pnl: float          # 累计盈亏
    currency: str = "USD"


# ---------------------------------------------------------------------------
# Abstract broker
# ---------------------------------------------------------------------------


class Broker(ABC):
    """Every broker adapter must implement these methods.

    Subclass contract
    -----------------
    1. `get_account()` → Account
    2. `get_positions()` → List[Position]
    3. `submit_order(order)` → Order (with updated status/order_id)
    4. `cancel_order(order_id)` → bool
    5. `get_order(order_id)` → Order

    Optional overrides
    ------------------
    6. `connect()` / `disconnect()` — for brokers that need sessions
    7. `on_order_update(callback)` — for async/push-based brokers
    8. `warmup(symbols)` / `refresh_prices(symbols)` / `last_prices` — market data

    Safety rules for implementations:
    - Never raise from submit_order — return Order with REJECTED status on failure
    - get_account/get_positions must raise when a complete broker snapshot is unavailable;
      never substitute fabricated cash or an empty position list for a query failure
    - get_positions MUST return the broker's ground-truth, not local cache
    - All monetary values in the account's base currency
    """

    def __init__(self):
        self.last_prices: Dict[str, float] = {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier: 'mock', 'qmt', 'futu', 'ib'."""
        ...

    @abstractmethod
    def get_account(self) -> Account:
        """Return current account snapshot (real-time query)."""
        ...

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Return all current positions (real-time query)."""
        ...

    @abstractmethod
    def submit_order(self, order: Order) -> Order:
        """Send an order to the broker.

        Returns an Order with:
        - `order_id` set (broker-assigned or local fallback)
        - `status` updated (SUBMITTED, REJECTED, or FILLED for mock)
        """
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order.  Returns True if successful."""
        ...

    def get_order(self, order_id: str) -> Optional[Order]:
        """Query a specific order's status.  Optional — default returns None."""
        return None

    def connect(self):
        """Optional: establish session / login."""
        pass

    def disconnect(self):
        """Optional: tear down session."""
        pass

    def warmup(self, symbols: List[str]) -> bool:
        """Optional: pre-connect / pre-load data for symbols. Returns True on success."""
        return True

    def refresh_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Optional: fetch latest prices for symbols, return {symbol: price}."""
        return {}

    def get_historical_kline(self, symbol: str, start: str, end: str):
        """Optional: fetch daily OHLCV kline from broker. Returns empty DataFrame by default."""
        import pandas as pd
        return pd.DataFrame()

    def __repr__(self):
        return f"<Broker:{self.name}>"
