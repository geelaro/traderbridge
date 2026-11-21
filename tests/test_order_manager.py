"""Tests for live/order_manager.py — OrderManager order generation and submission."""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from broker import Order, OrderSide, OrderStatus, OrderType, Position, Account
from engine.execution import ExecutionConfig, ExecutionModel, ExecutionTiming, ExecutionStyle
from live.order_manager import OrderManager
from utils.signal_gate import SignalGate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class FakePosition:
    symbol: str
    quantity: int
    avg_price: float = 100.0
    market_value: float = 10000.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@pytest.fixture
def mock_broker():
    broker = MagicMock()
    broker.last_prices = {"AAPL": 195.0, "NVDA": 850.0}
    broker.submit_order.return_value = MagicMock(status=OrderStatus.FILLED)
    broker.get_order.return_value = MagicMock(status=OrderStatus.FILLED)
    return broker


@pytest.fixture
def mock_cache():
    cache = MagicMock()
    return cache


@pytest.fixture
def mock_execution_model():
    config = ExecutionConfig(
        timing=ExecutionTiming.NEXT_OPEN,
        style=ExecutionStyle.MARKET,
        slippage_pct=0.0001,
        commission_rate=0.0003,
    )
    return ExecutionModel(config)


@pytest.fixture
def mock_notifier():
    return MagicMock()


@pytest.fixture
def mock_gate():
    gate = SignalGate()
    return gate


@pytest.fixture
def mock_risk_ctrl():
    rc = MagicMock()
    rc.calc_position_size.return_value = 100
    rc.passes_risk.return_value = True
    return rc


@pytest.fixture
def account():
    return Account(total_equity=100_000.0, available_cash=90_000.0, frozen_margin=0, total_pnl=0)


@pytest.fixture
def order_mgr(mock_broker, mock_cache, mock_execution_model, mock_notifier, mock_gate, mock_risk_ctrl):
    return OrderManager(
        broker=mock_broker,
        cache=mock_cache,
        execution_model=mock_execution_model,
        notifier=mock_notifier,
        gate=mock_gate,
        risk_ctrl=mock_risk_ctrl,
        dry_run=False,
    )


# ---------------------------------------------------------------------------
# generate_orders
# ---------------------------------------------------------------------------


class TestGenerateOrders:
    def test_creates_buy_order_for_buy_signal_without_position(self, order_mgr, account):
        signals = [{"symbol": "AAPL", "signal": 1, "price": 195.0, "atr": 5.0, "strategy": "weekly_macd"}]
        orders = order_mgr.generate_orders(signals, {}, account)
        assert len(orders) == 1
        assert orders[0].symbol == "AAPL"
        assert orders[0].side == OrderSide.BUY

    def test_creates_sell_order_for_sell_signal_with_position(self, order_mgr, account):
        signals = [{"symbol": "AAPL", "signal": -1, "price": 195.0, "atr": 5.0, "strategy": "weekly_macd"}]
        positions = {"AAPL": FakePosition(symbol="AAPL", quantity=50, market_value=9750.0)}
        orders = order_mgr.generate_orders(signals, positions, account)
        assert len(orders) == 1
        assert orders[0].symbol == "AAPL"
        assert orders[0].side == OrderSide.SELL
        assert orders[0].quantity == 50

    def test_skips_hold_signal(self, order_mgr, account):
        signals = [{"symbol": "AAPL", "signal": 0, "price": 195.0, "atr": 5.0, "strategy": "weekly_macd"}]
        orders = order_mgr.generate_orders(signals, {}, account)
        assert len(orders) == 0

    def test_blocks_buy_when_orphan_true(self, order_mgr, account):
        order_mgr.gate = SignalGate()
        signals = [{"symbol": "AAPL", "signal": 1, "price": 195.0, "atr": 5.0, "strategy": "weekly_macd", "orphan": True}]
        orders = order_mgr.generate_orders(signals, {}, account)
        assert len(orders) == 0

    def test_blocks_buy_when_gate_rejects(self, order_mgr, account):
        order_mgr.gate = SignalGate(trading_paused=True, pause_reason="test pause")
        signals = [{"symbol": "AAPL", "signal": 1, "price": 195.0, "atr": 5.0, "strategy": "weekly_macd"}]
        orders = order_mgr.generate_orders(signals, {}, account)
        assert len(orders) == 0

    def test_blocks_buy_when_risk_fails(self, order_mgr, account):
        order_mgr.risk_ctrl.passes_risk.return_value = False
        signals = [{"symbol": "AAPL", "signal": 1, "price": 195.0, "atr": 5.0, "strategy": "weekly_macd"}]
        orders = order_mgr.generate_orders(signals, {}, account)
        assert len(orders) == 0

    def test_signal_dedup_prefers_nonzero_over_hold(self, order_mgr, account):
        signals = [
            {"symbol": "AAPL", "signal": 0, "price": 195.0, "atr": 5.0, "strategy": "turtle_trading"},
            {"symbol": "AAPL", "signal": 1, "price": 195.0, "atr": 5.0, "strategy": "weekly_macd"},
        ]
        orders = order_mgr.generate_orders(signals, {}, account)
        assert len(orders) == 1
        assert orders[0].side == OrderSide.BUY

    def test_no_buy_when_already_have_position(self, order_mgr, account):
        signals = [{"symbol": "AAPL", "signal": 1, "price": 195.0, "atr": 5.0, "strategy": "weekly_macd"}]
        positions = {"AAPL": FakePosition(symbol="AAPL", quantity=50, market_value=9750.0)}
        orders = order_mgr.generate_orders(signals, positions, account)
        assert len(orders) == 0

    def test_short_entry_when_no_position(self, order_mgr, account):
        """Short research signals never open a broker position."""
        signals = [{"symbol": "AAPL", "signal": -1, "price": 195.0, "atr": 5.0, "strategy": "turtle_trading"}]
        orders = order_mgr.generate_orders(signals, {}, account)
        assert orders == []
        order_mgr.cache.log_ops.assert_called_with(
            "gate_reject", symbol="AAPL", detail="SHORT_ENTRY_RESEARCH_ONLY", level="INFO",
        )

    def test_no_sell_when_long_only_strategy(self, order_mgr, account):
        """Long-only strategy with signal=-1 on long position → close long."""
        pos = FakePosition("AAPL", 100, 190.0)
        signals = [{"symbol": "AAPL", "signal": -1, "price": 195.0, "atr": 5.0, "strategy": "weekly_macd"}]
        orders = order_mgr.generate_orders(signals, {"AAPL": pos}, account)
        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL  # close long

    def test_zero_qty_signal_skipped(self, order_mgr, account):
        order_mgr.risk_ctrl.calc_position_size.return_value = 0
        signals = [{"symbol": "AAPL", "signal": 1, "price": 195.0, "atr": 5.0, "strategy": "weekly_macd"}]
        orders = order_mgr.generate_orders(signals, {}, account)
        assert len(orders) == 0

    def test_multiple_symbols_generate_separate_orders(self, order_mgr, account):
        order_mgr.gate.max_total_exposure_pct = 0.99
        signals = [
            {"symbol": "AAPL", "signal": 1, "price": 195.0, "atr": 5.0, "strategy": "weekly_macd"},
            {"symbol": "NVDA", "signal": 1, "price": 850.0, "atr": 30.0, "strategy": "weekly_macd"},
        ]
        orders = order_mgr.generate_orders(signals, {}, account)
        assert len(orders) == 2
        symbols = {o.symbol for o in orders}
        assert symbols == {"AAPL", "NVDA"}


# ---------------------------------------------------------------------------
# submit_and_wait
# ---------------------------------------------------------------------------


class TestSubmitAndWait:
    def test_immediate_fill_returns_immediately(self, order_mgr):
        filled_order = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET,
            quantity=100, status=OrderStatus.FILLED, avg_fill_price=195.0,
        )
        order_mgr.broker.submit_order.return_value = filled_order
        order = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100,
        )
        result = order_mgr.submit_and_wait(order)
        assert result.status == OrderStatus.FILLED
        order_mgr.broker.get_order.assert_not_called()

    def test_immediate_rejected_returns_immediately(self, order_mgr):
        rejected = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET,
            quantity=100, status=OrderStatus.REJECTED,
        )
        order_mgr.broker.submit_order.return_value = rejected
        order = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100,
        )
        result = order_mgr.submit_and_wait(order)
        assert result.status == OrderStatus.REJECTED

    def test_timeout_cancels_order(self, order_mgr):
        order_mgr.execution_model.config = ExecutionConfig(market_timeout_seconds=1)
        submitted = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET,
            quantity=100, status=OrderStatus.SUBMITTED, order_id="ord-1",
        )
        order_mgr.broker.submit_order.return_value = submitted
        order_mgr.broker.get_order.return_value = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET,
            quantity=100, status=OrderStatus.SUBMITTED, order_id="ord-1",
        )
        order = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100,
        )
        result = order_mgr.submit_and_wait(order)
        assert result.status == OrderStatus.CANCELLED

    def test_eventual_fill_before_timeout(self, order_mgr):
        submitted = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET,
            quantity=100, status=OrderStatus.SUBMITTED, order_id="ord-1",
        )
        order_mgr.broker.submit_order.return_value = submitted
        filled = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET,
            quantity=100, status=OrderStatus.FILLED, order_id="ord-1", avg_fill_price=195.0,
        )
        order_mgr.broker.get_order.return_value = filled
        order = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100,
        )
        result = order_mgr.submit_and_wait(order)
        assert result.status == OrderStatus.FILLED


# ---------------------------------------------------------------------------
# record_slippage / log_order
# ---------------------------------------------------------------------------


class TestRecordSlippage:
    def test_records_slippage(self, order_mgr):
        order = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET,
            quantity=100, avg_fill_price=196.0, order_id="ord-1",
        )
        order_mgr.record_slippage(order, 195.0)
        order_mgr.cache.log_ops.assert_called()

    def test_skips_when_signal_price_zero(self, order_mgr):
        order = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET,
            quantity=100, avg_fill_price=196.0, order_id="ord-1",
        )
        order_mgr.record_slippage(order, 0)
        order_mgr.cache.log_ops.assert_not_called()

    def test_skips_when_fill_price_zero(self, order_mgr):
        order = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET,
            quantity=100, avg_fill_price=0, order_id="ord-1",
        )
        order_mgr.record_slippage(order, 195.0)
        order_mgr.cache.log_ops.assert_not_called()
