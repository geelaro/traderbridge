"""Tests for live/risk_controller.py — RiskController directly."""

from datetime import date
from unittest.mock import MagicMock

import pytest

from broker import Order, OrderSide
from live.risk_controller import RiskController
from utils.risk import RiskLimits


@pytest.fixture
def risk_limits():
    return RiskLimits()


@pytest.fixture
def mock_cache():
    cache = MagicMock()
    cache.load_risk_state.return_value = None
    cache.load_all_entry_prices.return_value = {}
    return cache


@pytest.fixture
def mock_broker():
    broker = MagicMock()
    broker.last_prices = {}
    return broker


@pytest.fixture
def mock_notifier():
    return MagicMock()


@pytest.fixture
def controller(risk_limits, mock_cache, mock_broker, mock_notifier):
    mock_cache.load_risk_state.return_value = None
    mock_cache.load_all_entry_prices.return_value = {}
    return RiskController(
        risk=risk_limits,
        cache=mock_cache,
        broker=mock_broker,
        notifier=mock_notifier,
    )


@pytest.fixture
def account():
    acc = MagicMock()
    acc.total_equity = 100_000.0
    return acc


# ---------------------------------------------------------------------------
# init_risk
# ---------------------------------------------------------------------------


class TestInitRisk:
    def test_sets_day_start_equity_on_new_day(self, controller, account):
        controller.risk._date = ""
        controller.risk._day_start_equity = 0.0
        controller.init_risk(account)
        assert controller.risk._day_start_equity == 100_000.0
        assert controller.risk._date == date.today().isoformat()

    def test_preserves_equity_on_same_day(self, controller, account):
        today = date.today().isoformat()
        controller.risk._date = today
        controller.risk._day_start_equity = 90_000.0
        account.total_equity = 85_000.0
        controller.init_risk(account)
        assert controller.risk._day_start_equity == 90_000.0

    def test_updates_peak_equity(self, controller, account):
        controller.risk._peak_equity = 90_000.0
        account.total_equity = 110_000.0
        controller.init_risk(account)
        assert controller.risk._peak_equity == 110_000.0

    def test_does_not_lower_peak_equity(self, controller, account):
        controller.risk._peak_equity = 120_000.0
        account.total_equity = 110_000.0
        controller.init_risk(account)
        assert controller.risk._peak_equity == 120_000.0


# ---------------------------------------------------------------------------
# check_global
# ---------------------------------------------------------------------------


class TestCheckGlobal:
    def test_daily_loss_triggers_pause(self, controller, account):
        controller.risk._date = date.today().isoformat()
        controller.risk._day_start_equity = 100_000.0
        controller.risk._peak_equity = 120_000.0
        account.total_equity = 93_000.0  # -7%, exceeds 5% cap
        result = controller.check_global(account, {})
        assert result is False
        assert controller.trading_paused is True
        assert "日内亏损超限" in controller.pause_reason

    def test_total_drawdown_triggers_pause(self, controller, account):
        controller.risk._date = date.today().isoformat()
        controller.risk._day_start_equity = 50_000.0  # low so daily loss doesn't trigger
        controller.risk._peak_equity = 100_000.0
        account.total_equity = 65_000.0  # -35% from peak, but above daily loss threshold
        result = controller.check_global(account, {})
        assert result is False
        assert controller.trading_paused is True

    def test_passes_when_within_limits(self, controller, account):
        controller.risk._date = date.today().isoformat()
        controller.risk._day_start_equity = 100_000.0
        controller.risk._peak_equity = 120_000.0
        account.total_equity = 99_000.0
        result = controller.check_global(account, {})
        assert result is True
        assert controller.trading_paused is False

    def test_resets_pause_on_new_day(self, controller, account):
        controller.trading_paused = True
        controller.pause_reason = "old reason"
        controller.risk._date = "2020-01-01"
        controller.risk._day_start_equity = 100_000.0
        controller.risk._peak_equity = 120_000.0
        account.total_equity = 99_000.0
        result = controller.check_global(account, {})
        assert result is True
        assert controller.trading_paused is False

    def test_consecutive_losses_triggers_pause(self, controller, account):
        controller.risk._date = date.today().isoformat()
        controller.risk._day_start_equity = 100_000.0
        controller.risk._peak_equity = 120_000.0
        controller.risk._consecutive_losses = 3
        controller.risk.max_consecutive_losses = 3
        account.total_equity = 99_000.0
        result = controller.check_global(account, {})
        assert result is False
        assert "连续亏损熔断" in controller.pause_reason

    def test_exposure_cap_triggers_pause(self, controller, account):
        controller.risk._date = date.today().isoformat()
        controller.risk._day_start_equity = 100_000.0
        controller.risk._peak_equity = 120_000.0
        account.total_equity = 100_000.0
        pos = MagicMock()
        pos.market_value = 90_000.0
        result = controller.check_global(account, {"AAPL": pos})
        assert result is False
        assert "敞口超限" in controller.pause_reason

    def test_pause_state_is_persisted_for_dashboard(self, controller, account):
        controller.risk._date = date.today().isoformat()
        controller.risk._day_start_equity = 100_000.0
        controller.risk._peak_equity = 120_000.0
        account.total_equity = 93_000.0

        assert controller.check_global(account, {}) is False

        calls = controller.cache.save_risk_state_many.call_args.args[0]
        assert calls["trading_paused"] == "True"
        assert "日内亏损超限" in calls["pause_reason"]
        assert "risk_operational_snapshot" in calls

    def test_operational_snapshot_contains_utilisation_and_limits(self, controller, account):
        controller.risk._day_start_equity = 100_000.0
        controller.risk._peak_equity = 120_000.0
        controller.risk._daily_trade_count = 2
        controller.risk._consecutive_losses = 1
        account.total_equity = 95_000.0
        pos = MagicMock()
        pos.market_value = 20_000.0

        snapshot = controller.risk_snapshot(account, {"AAPL": pos})

        assert snapshot["daily_pnl_pct"] == -5.0
        assert snapshot["drawdown_pct"] == -20.83
        assert snapshot["exposure_pct"] == pytest.approx(21.05)
        assert snapshot["daily_trade_count"] == 2
        assert snapshot["limits"]["max_daily_trades"] == controller.risk.max_daily_trades
        assert snapshot["paused"] is False


# ---------------------------------------------------------------------------
# passes_risk
# ---------------------------------------------------------------------------


class TestPassesRisk:
    def test_daily_loss_cap_exceeded(self, controller, account):
        controller.risk._day_start_equity = 100_000.0
        account.total_equity = 90_000.0  # -10%, exceeds 5%
        signal = {"symbol": "AAPL", "price": 195.0}
        result = controller.passes_risk(signal, 10, account)
        assert result is False

    def test_consecutive_losses_blocks(self, controller, account):
        controller.risk._consecutive_losses = 4
        controller.risk.max_consecutive_losses = 3
        account.total_equity = 100_000.0
        signal = {"symbol": "AAPL", "price": 195.0}
        result = controller.passes_risk(signal, 10, account)
        assert result is False

    def test_daily_trade_count_cap(self, controller, account):
        controller.risk._daily_trade_count = 5
        controller.risk.max_daily_trades = 5
        account.total_equity = 100_000.0
        signal = {"symbol": "AAPL", "price": 195.0}
        result = controller.passes_risk(signal, 10, account)
        assert result is False

    def test_min_order_value_rejects(self, controller, account):
        controller.risk._day_start_equity = 100_000.0
        account.total_equity = 100_000.0
        signal = {"symbol": "AAPL", "price": 4.0}
        result = controller.passes_risk(signal, 10, account)
        assert result is False  # $40 < $500 min

    def test_slippage_rejects(self, controller, account):
        controller.risk._day_start_equity = 100_000.0
        account.total_equity = 100_000.0
        controller.broker.last_prices = {"AAPL": 195.0}
        signal = {"symbol": "AAPL", "price": 220.0}  # 12.8% slippage > 2%
        result = controller.passes_risk(signal, 10, account)
        assert result is False


# ---------------------------------------------------------------------------
# persist_state / restore_state
# ---------------------------------------------------------------------------


class TestStatePersistence:
    def test_persist_state_writes_related_keys_atomically(self, controller):
        controller.risk._date = "2025-01-15"
        controller.risk._day_start_equity = 100_000.0
        controller.risk._peak_equity = 110_000.0
        controller.persist_state()
        values = controller.cache.save_risk_state_many.call_args.args[0]
        assert values["date"] == "2025-01-15"
        assert values["peak_equity"] == "110000.0"
        assert "trading_paused" in values
        controller.cache.save_risk_state.assert_not_called()

    def test_restore_state_loads_from_cache(self, risk_limits, mock_cache, mock_broker, mock_notifier):
        mock_cache.load_risk_state.side_effect = lambda key: {
            "date": date.today().isoformat(),
            "day_start_equity": "95000.0",
            "peak_equity": "105000.0",
            "consecutive_losses": "1",
            "daily_trade_count": "2",
        }.get(key)
        mock_cache.load_all_entry_prices.return_value = {"AAPL": (195.0, "2025-01-15")}
        ctrl = RiskController(
            risk=risk_limits,
            cache=mock_cache,
            broker=mock_broker,
            notifier=mock_notifier,
        )
        assert ctrl.risk._day_start_equity == 95000.0
        assert ctrl.risk._consecutive_losses == 1
        assert ctrl.risk._daily_trade_count == 2
        assert ctrl.risk._peak_equity == 105000.0
        assert "AAPL" in ctrl.entry_prices
        assert ctrl.entry_prices["AAPL"] == 195.0

    def test_restore_state_different_day_keeps_historical_resets_daily(self, risk_limits, mock_cache, mock_broker, mock_notifier):
        """Cross-day restore: peak_equity & consecutive_losses persist (historical state);
        day_start_equity & daily_trade_count reset (day-scoped)."""
        mock_cache.load_risk_state.side_effect = lambda key: {
            "date": "2020-01-01",
            "day_start_equity": "95000.0",
            "peak_equity": "105000.0",
            "consecutive_losses": "1",
            "daily_trade_count": "2",
        }.get(key)
        mock_cache.load_all_entry_prices.return_value = {}
        ctrl = RiskController(
            risk=risk_limits,
            cache=mock_cache,
            broker=mock_broker,
            notifier=mock_notifier,
        )
        # Day-scoped state — reset on rollover
        assert ctrl.risk._day_start_equity == 0.0
        assert ctrl.risk._daily_trade_count == 0
        # Historical state — survives day rollover (P1-3 fix)
        assert ctrl.risk._peak_equity == 105000.0
        assert ctrl.risk._consecutive_losses == 1
        assert ctrl.trading_paused is False
        assert ctrl.pause_reason == ""

    def test_restore_state_same_day_restores_pause(self, risk_limits, mock_cache, mock_broker, mock_notifier):
        mock_cache.load_risk_state.side_effect = lambda key: {
            "date": date.today().isoformat(),
            "day_start_equity": "95000.0",
            "peak_equity": "105000.0",
            "consecutive_losses": "1",
            "daily_trade_count": "2",
            "trading_paused": "True",
            "pause_reason": "连续亏损熔断 (3/3)",
        }.get(key)
        mock_cache.load_all_entry_prices.return_value = {}

        ctrl = RiskController(
            risk=risk_limits,
            cache=mock_cache,
            broker=mock_broker,
            notifier=mock_notifier,
        )

        assert ctrl.trading_paused is True
        assert ctrl.pause_reason == "连续亏损熔断 (3/3)"


# ---------------------------------------------------------------------------
# on_trade_filled
# ---------------------------------------------------------------------------


class TestOnTradeFilled:
    def test_buy_updates_entry_prices(self, controller):
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=MagicMock(),
            quantity=10,
            avg_fill_price=195.0,
        )
        controller.on_trade_filled(order)
        assert controller.entry_prices.get("AAPL") == 195.0
        controller.cache.save_entry_price.assert_called_once()

    def test_sell_with_profit_resets_consecutive_losses(self, controller):
        controller.entry_prices["AAPL"] = 100.0
        controller.risk._consecutive_losses = 2
        order = Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            order_type=MagicMock(),
            quantity=10,
            avg_fill_price=110.0,
        )
        controller.on_trade_filled(order)
        assert controller.risk._consecutive_losses == 0

    def test_sell_with_loss_increments_consecutive_losses(self, controller):
        controller.entry_prices["NVDA"] = 200.0
        controller.risk._consecutive_losses = 1
        order = Order(
            symbol="NVDA",
            side=OrderSide.SELL,
            order_type=MagicMock(),
            quantity=10,
            avg_fill_price=180.0,
        )
        controller.on_trade_filled(order)
        assert controller.risk._consecutive_losses == 2

    def test_increments_daily_trade_count(self, controller):
        controller.risk._daily_trade_count = 1
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=MagicMock(),
            quantity=10,
            avg_fill_price=195.0,
        )
        controller.on_trade_filled(order)
        assert controller.risk._daily_trade_count == 2
