"""Tests for broker/middleware.py — RetryingBroker."""

import time
from unittest.mock import MagicMock

import pytest

from broker import Order, OrderSide, OrderStatus, OrderType
from broker.middleware import CircuitOpenError, RetryingBroker


def _real_broker():
    """Minimal mocked broker matching the Broker interface."""
    b = MagicMock()
    b.name = "mock"
    b.get_account = MagicMock(return_value="account")
    b.get_positions = MagicMock(return_value=[])
    b.submit_order = MagicMock(return_value=Order(
        symbol="AAPL", side=OrderSide.SELL, order_type=OrderType.MARKET,
        quantity=10, status=OrderStatus.FILLED, order_id="ord1",
    ))
    b.cancel_order = MagicMock(return_value=True)
    b.get_order = MagicMock(return_value=None)
    b.warmup = MagicMock(return_value=True)
    b.last_prices = {}
    return b


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Skip real backoff sleeps in tests."""
    monkeypatch.setattr(time, "sleep", lambda *_: None)


class TestPassthrough:
    def test_name_includes_inner(self):
        inner = _real_broker()
        rb = RetryingBroker(inner)
        assert rb.name == "retrying:mock"

    def test_happy_path_no_retry(self):
        inner = _real_broker()
        rb = RetryingBroker(inner)
        assert rb.get_account() == "account"
        assert inner.get_account.call_count == 1


class TestRetry:
    def test_transient_error_retries(self):
        inner = _real_broker()
        inner.get_positions.side_effect = [
            ConnectionError("boom"),
            ConnectionError("boom"),
            ["pos"],
        ]
        rb = RetryingBroker(inner, max_retries=3)
        assert rb.get_positions() == ["pos"]
        assert inner.get_positions.call_count == 3

    def test_exhausts_retries_raises_last(self):
        inner = _real_broker()
        inner.get_positions.side_effect = TimeoutError("offline")
        rb = RetryingBroker(inner, max_retries=2)
        with pytest.raises(TimeoutError):
            rb.get_positions()
        # 1 initial + 2 retries = 3 calls
        assert inner.get_positions.call_count == 3

    def test_non_retryable_immediate(self):
        """ValueError isn't in the retryable set — re-raised immediately."""
        inner = _real_broker()
        inner.submit_order.side_effect = ValueError("bad input")
        rb = RetryingBroker(inner, max_retries=3)
        with pytest.raises(ValueError):
            rb.submit_order(Order(
                symbol="X", side=OrderSide.BUY,
                order_type=OrderType.MARKET, quantity=1,
            ))
        assert inner.submit_order.call_count == 1  # no retry

    def test_submit_order_timeout_is_not_retried(self):
        """A timeout may follow broker acceptance, so retrying could duplicate."""
        inner = _real_broker()
        inner.submit_order.side_effect = TimeoutError("response lost")
        rb = RetryingBroker(inner, max_retries=3)
        order = Order(
            symbol="X", side=OrderSide.BUY,
            order_type=OrderType.MARKET, quantity=1,
        )

        with pytest.raises(TimeoutError, match="response lost"):
            rb.submit_order(order)

        inner.submit_order.assert_called_once_with(order)


class TestCircuitBreaker:
    def test_opens_after_threshold(self):
        inner = _real_broker()
        inner.get_positions.side_effect = ConnectionError("nope")
        rb = RetryingBroker(inner, max_retries=0, cb_threshold=3,
                            cb_cooldown=60)
        # 3 failed calls — each tries once (max_retries=0)
        for _ in range(3):
            with pytest.raises(ConnectionError):
                rb.get_positions()
        # 4th call rejected by circuit breaker — no inner call
        inner.get_positions.reset_mock()
        with pytest.raises(CircuitOpenError):
            rb.get_positions()
        inner.get_positions.assert_not_called()

    def test_circuit_resets_after_cooldown(self, monkeypatch):
        """After cooldown, a half-open trial call recovers the circuit."""
        inner = _real_broker()
        inner.get_positions.side_effect = ConnectionError("nope")
        rb = RetryingBroker(inner, max_retries=0, cb_threshold=2,
                            cb_cooldown=0.0)  # zero cooldown for the test
        for _ in range(2):
            with pytest.raises(ConnectionError):
                rb.get_positions()
        # Circuit is open but cooldown=0 → next call is half-open trial
        # Make it succeed now
        inner.get_positions.side_effect = None
        inner.get_positions.return_value = ["pos"]
        assert rb.get_positions() == ["pos"]

    def test_success_resets_failure_counter(self):
        inner = _real_broker()
        inner.get_positions.side_effect = [
            ConnectionError("a"), ConnectionError("b"),
            ["ok"], ConnectionError("c"),
        ]
        rb = RetryingBroker(inner, max_retries=0, cb_threshold=3,
                            cb_cooldown=60)
        with pytest.raises(ConnectionError):
            rb.get_positions()
        with pytest.raises(ConnectionError):
            rb.get_positions()
        # Third call succeeds → counter reset
        assert rb.get_positions() == ["ok"]
        # Fourth call fails — but counter just got reset, no circuit
        with pytest.raises(ConnectionError):
            rb.get_positions()
        # Counter is at 1 not 3 — circuit still closed
        inner.get_positions.side_effect = None
        inner.get_positions.return_value = ["ok2"]
        assert rb.get_positions() == ["ok2"]
