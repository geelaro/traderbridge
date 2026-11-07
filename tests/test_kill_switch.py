"""Tests for live/kill_switch.py — manual emergency liquidation."""

from unittest.mock import MagicMock

import pytest

from broker import Order, OrderSide, OrderStatus, OrderType, Position
from live.kill_switch import KillSwitch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _position(symbol, qty, avg=100.0):
    return Position(
        symbol=symbol, quantity=qty, avg_price=avg,
        market_value=qty * avg, unrealized_pnl=0.0,
    )


def _make_filled_order(order: Order, order_id="ord_test") -> Order:
    """Construct an Order in FILLED state to use as submit_order return."""
    return Order(
        symbol=order.symbol,
        side=order.side,
        order_type=order.order_type,
        quantity=order.quantity,
        order_id=order_id,
        status=OrderStatus.FILLED,
        filled_qty=order.quantity,
        avg_fill_price=100.0,
    )


@pytest.fixture
def broker():
    b = MagicMock()
    b.get_positions = MagicMock(return_value=[])
    # submit_order echoes back a FILLED version of the input
    b.submit_order = MagicMock(side_effect=lambda o: _make_filled_order(o))
    return b


@pytest.fixture
def risk_ctrl():
    rc = MagicMock()
    rc.trading_paused = False
    rc.pause_reason = ""
    rc.persist_state = MagicMock()
    return rc


@pytest.fixture
def notifier():
    nf = MagicMock()
    nf.available = True
    nf.kill_switch_card = MagicMock(return_value=True)
    return nf


@pytest.fixture
def kill_switch(broker, risk_ctrl, notifier, temp_cache):
    return KillSwitch(broker, risk_ctrl, notifier, temp_cache)


# ===================================================================
# trigger
# ===================================================================


class TestTrigger:
    def test_empty_positions(self, kill_switch, broker):
        result = kill_switch.trigger("test reason")
        assert result["status"] == "no_positions"
        assert result["n_positions"] == 0
        assert result["orders"] == []
        # No orders submitted
        broker.submit_order.assert_not_called()

    def test_submits_sell_for_long_position(self, kill_switch, broker):
        broker.get_positions.return_value = [_position("AAPL", 100)]
        result = kill_switch.trigger("闪崩")
        assert result["status"] == "triggered"
        assert result["n_positions"] == 1
        broker.submit_order.assert_called_once()
        submitted = broker.submit_order.call_args.args[0]
        assert submitted.symbol == "AAPL"
        assert submitted.side == OrderSide.SELL
        assert submitted.quantity == 100
        assert submitted.order_type == OrderType.MARKET

    def test_submits_buy_for_short_position(self, kill_switch, broker):
        broker.get_positions.return_value = [_position("AAPL", -50)]
        kill_switch.trigger("空头平仓")
        submitted = broker.submit_order.call_args.args[0]
        assert submitted.side == OrderSide.BUY
        assert submitted.quantity == 50

    def test_skips_zero_quantity(self, kill_switch, broker):
        broker.get_positions.return_value = [_position("AAPL", 0)]
        result = kill_switch.trigger("test")
        assert result["n_positions"] == 1  # listed but skipped
        broker.submit_order.assert_not_called()

    def test_multiple_positions(self, kill_switch, broker):
        broker.get_positions.return_value = [
            _position("AAPL", 100), _position("MSFT", 50), _position("TSLA", -30),
        ]
        result = kill_switch.trigger("全部平仓")
        assert result["n_positions"] == 3
        assert len(result["orders"]) == 3
        assert broker.submit_order.call_count == 3

    def test_requires_non_empty_reason(self, kill_switch):
        with pytest.raises(ValueError):
            kill_switch.trigger("")
        with pytest.raises(ValueError):
            kill_switch.trigger("   ")

    def test_pauses_trading(self, kill_switch, broker, risk_ctrl):
        broker.get_positions.return_value = [_position("AAPL", 100)]
        kill_switch.trigger("闪崩")
        assert risk_ctrl.trading_paused is True
        assert "Kill Switch" in risk_ctrl.pause_reason

    def test_dry_run_is_read_only(
            self, kill_switch, broker, risk_ctrl, notifier, temp_cache):
        broker.get_positions.return_value = [_position("AAPL", 100)]
        result = kill_switch.trigger("dry test", dry_run=True)
        assert result["dry_run"] is True
        assert result["status"] == "preview"
        assert result["orders"] == [{
            "symbol": "AAPL", "side": "SELL", "qty": 100,
            "status": "DRY_RUN", "order_id": "",
        }]
        broker.submit_order.assert_not_called()
        assert risk_ctrl.trading_paused is False
        risk_ctrl.persist_state.assert_not_called()
        assert kill_switch.is_active is False
        assert temp_cache.load_alert_history(
            days=1, alert_type="kill_switch",
        ) == []
        assert temp_cache.query_decisions(days=1, decision_type="kill_switch") == []
        notifier.kill_switch_card.assert_not_called()

    def test_broker_exception_recorded_as_error(self, kill_switch, broker):
        broker.get_positions.return_value = [
            _position("AAPL", 100), _position("FAIL", 50),
        ]

        def submit(order):
            if order.symbol == "FAIL":
                raise RuntimeError("API down")
            return _make_filled_order(order)

        broker.submit_order.side_effect = submit
        result = kill_switch.trigger("test")
        # AAPL ok, FAIL errored
        assert len(result["orders"]) == 1
        assert len(result["errors"]) == 1
        assert result["errors"][0]["symbol"] == "FAIL"

    def test_audit_history_recorded(self, kill_switch, broker, temp_cache):
        broker.get_positions.return_value = [_position("AAPL", 100)]
        kill_switch.trigger("test reason")
        rows = temp_cache.load_alert_history(days=1, alert_type="kill_switch")
        assert len(rows) == 1
        assert rows[0]["payload"]["reason"] == "test reason"
        assert rows[0]["payload"]["n_positions"] == 1

    def test_notifier_called_on_trigger(self, kill_switch, broker, notifier):
        broker.get_positions.return_value = [_position("AAPL", 100)]
        kill_switch.trigger("test")
        notifier.kill_switch_card.assert_called_once()

    def test_notifier_unavailable_does_not_crash(self, kill_switch, broker, notifier):
        broker.get_positions.return_value = [_position("AAPL", 100)]
        notifier.available = False
        result = kill_switch.trigger("test")  # should not raise
        assert result["status"] == "triggered"
        notifier.kill_switch_card.assert_not_called()


# ===================================================================
# Idempotency
# ===================================================================


class TestIdempotency:
    def test_re_trigger_while_active_is_no_op(self, kill_switch, broker):
        broker.get_positions.return_value = [_position("AAPL", 100)]
        kill_switch.trigger("first")
        broker.submit_order.reset_mock()

        broker.get_positions.return_value = [_position("MSFT", 50)]
        result = kill_switch.trigger("second")
        # Second trigger should be a no-op
        assert result["status"] == "already_active"
        broker.submit_order.assert_not_called()


# ===================================================================
# State + reset
# ===================================================================


class TestBrokerOrderOutcomes:
    """Broker may return non-FILLED order statuses — Kill Switch must
    record them in audit but still mark itself active."""

    def test_rejected_order_recorded_but_active_set(
            self, kill_switch, broker, temp_cache):
        """Broker returns REJECTED — error path inside audit, kill switch
        is still active (don't keep retrying)."""
        broker.get_positions.return_value = [_position("AAPL", 100)]

        def submit(order):
            rej = _make_filled_order(order, order_id="rej_1")
            rej.status = OrderStatus.REJECTED
            return rej

        broker.submit_order.side_effect = submit
        result = kill_switch.trigger("test")
        # Order recorded with REJECTED status — audit visible to user
        assert result["status"] == "triggered"
        assert len(result["orders"]) == 1
        assert result["orders"][0]["status"] == "REJECTED"
        # Active flag still set — no auto-retry
        assert kill_switch.is_active is True

    def test_partial_fill_recorded(self, kill_switch, broker):
        broker.get_positions.return_value = [_position("AAPL", 100)]

        def submit(order):
            o = _make_filled_order(order, order_id="part_1")
            o.status = OrderStatus.PARTIAL
            o.filled_qty = 60
            return o

        broker.submit_order.side_effect = submit
        result = kill_switch.trigger("test")
        assert result["orders"][0]["status"] == "PARTIAL"


class TestRobustness:
    """Failure modes that MUST not silently break Kill Switch."""

    def test_broker_get_positions_failure_still_pauses_and_notifies(
            self, kill_switch, broker, risk_ctrl, notifier, temp_cache):
        """Broker outage during trigger() must not skip the safety steps.

        If get_positions raises, we still want:
          - risk_ctrl.trading_paused = True (so daemon doesn't open new pos)
          - kill_switch.is_active = True (idempotency flag set)
          - Audit history written
          - Feishu notification fired
          - errors[] populated with the broker exception
        """
        broker.get_positions.side_effect = RuntimeError("broker offline")
        result = kill_switch.trigger("emergency")
        # Trigger completes — not exception
        assert result["status"] in ("triggered", "no_positions")
        assert result["errors"]
        assert "broker offline" in result["errors"][0]["error"]
        # Safety steps all happened
        assert risk_ctrl.trading_paused is True
        assert kill_switch.is_active is True
        notifier.kill_switch_card.assert_called_once()
        # Audit recorded
        rows = temp_cache.load_alert_history(days=1, alert_type="kill_switch")
        assert len(rows) == 1
        assert "broker offline" in rows[0]["payload"]["errors"][0]["error"]


class TestAtomicStatePersistence:
    """The active flag + reason + timestamp must be stored atomically so a
    mid-write crash can't leave 'active=1 with empty reason'."""

    def test_state_loaded_from_atomic_json_blob(
            self, kill_switch, broker, temp_cache):
        from live.kill_switch import _KEY_STATE, _KEY_ACTIVE
        broker.get_positions.return_value = [_position("AAPL", 100)]
        kill_switch.trigger("test reason")
        # Atomic JSON exists
        blob = temp_cache.load_risk_state(_KEY_STATE)
        assert blob is not None
        import json as _j
        parsed = _j.loads(blob)
        assert parsed["active"] is True
        assert parsed["reason"] == "test reason"
        assert parsed["triggered_at"]

    def test_atomic_state_takes_precedence_over_legacy(
            self, kill_switch, broker, temp_cache):
        """If JSON blob says inactive, legacy "1" mirror is ignored."""
        from live.kill_switch import _KEY_STATE, _KEY_ACTIVE
        import json as _j
        # Synthesise a stale legacy "active" marker
        temp_cache.save_risk_state(_KEY_ACTIVE, "1")
        # But the atomic blob says inactive
        temp_cache.save_risk_state(_KEY_STATE, _j.dumps({
            "active": False, "reason": "", "triggered_at": "",
        }))
        assert kill_switch.is_active is False

    def test_legacy_state_fallback_when_blob_missing(
            self, kill_switch, broker, temp_cache):
        """Old installations with only the three legacy keys still work."""
        from live.kill_switch import _KEY_ACTIVE, _KEY_REASON, _KEY_TRIGGERED_AT
        temp_cache.save_risk_state(_KEY_ACTIVE, "1")
        temp_cache.save_risk_state(_KEY_REASON, "legacy")
        temp_cache.save_risk_state(_KEY_TRIGGERED_AT, "2026-06-01T10:00:00")
        # No _KEY_STATE blob — should fall back
        state = kill_switch.get_state()
        assert state["active"] is True
        assert state["reason"] == "legacy"

    def test_malformed_blob_falls_back_to_legacy(
            self, kill_switch, temp_cache):
        from live.kill_switch import _KEY_STATE, _KEY_ACTIVE
        temp_cache.save_risk_state(_KEY_STATE, "not valid json {{{")
        temp_cache.save_risk_state(_KEY_ACTIVE, "1")
        # Should fall back gracefully to legacy key
        assert kill_switch.is_active is True


class TestStateAndReset:
    def test_initial_not_active(self, kill_switch):
        assert kill_switch.is_active is False
        state = kill_switch.get_state()
        assert state["active"] is False
        assert state["reason"] == ""

    def test_active_after_trigger(self, kill_switch, broker):
        broker.get_positions.return_value = [_position("AAPL", 100)]
        kill_switch.trigger("闪崩")
        assert kill_switch.is_active is True
        state = kill_switch.get_state()
        assert state["active"] is True
        assert state["reason"] == "闪崩"
        assert state["triggered_at"]

    def test_reset_clears_state(self, kill_switch, broker, risk_ctrl):
        broker.get_positions.return_value = [_position("AAPL", 100)]
        kill_switch.trigger("test")
        assert kill_switch.is_active is True

        kill_switch.reset()
        assert kill_switch.is_active is False
        assert risk_ctrl.trading_paused is False
        assert risk_ctrl.pause_reason == ""

    def test_reset_when_not_active_is_safe(self, kill_switch):
        # Should not raise
        kill_switch.reset()
        assert kill_switch.is_active is False

    def test_reset_recorded_in_audit(self, kill_switch, broker, temp_cache):
        broker.get_positions.return_value = [_position("AAPL", 100)]
        kill_switch.trigger("test")
        kill_switch.reset("checked positions")
        rows = temp_cache.load_alert_history(days=1, alert_type="kill_switch_reset")
        assert len(rows) == 1
        assert "checked positions" in rows[0]["payload"]["reason"]

    def test_can_trigger_again_after_reset(self, kill_switch, broker):
        broker.get_positions.return_value = [_position("AAPL", 100)]
        kill_switch.trigger("first")
        kill_switch.reset()
        broker.submit_order.reset_mock()

        # Re-trigger should now work
        broker.get_positions.return_value = [_position("MSFT", 50)]
        result = kill_switch.trigger("second")
        assert result["status"] == "triggered"
        broker.submit_order.assert_called_once()
