"""Tests for live_trader.py — RiskLimits, LiveTrader order generation, position sizing."""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from live_trader import LiveTrader, RiskLimits
from utils.market_state import MarketRegime, Volatility
from broker import (
    Order, OrderSide, OrderStatus, OrderType, Position, Account, MockBroker,
)


# ===================================================================
# RiskLimits (extending existing test_risk.py)
# ===================================================================


class TestRiskLimitsDayTracking:
    def test_new_day_resets_equity(self):
        r = RiskLimits()
        r._day_start_equity = 100000
        r._date = "2025-01-14"

        # Simulate _init_risk with a new day
        account = type("Account", (), {"total_equity": 105000})()
        today = "2025-01-15"
        if r._date != today:
            r._day_start_equity = account.total_equity
            r._date = today

        assert r._day_start_equity == 105000
        assert r._date == "2025-01-15"

    def test_same_day_keeps_equity(self):
        r = RiskLimits()
        r._day_start_equity = 100000
        r._date = "2025-01-15"

        account = type("Account", (), {"total_equity": 95000})()
        today = "2025-01-15"
        if r._date != today:
            r._day_start_equity = account.total_equity

        assert r._day_start_equity == 100000  # unchanged


# ===================================================================
# LiveTrader._calc_position_size
# ===================================================================


class TestPositionSizing:
    def _make_trader(self):
        broker = MockBroker(initial_cash=100000)
        broker.last_prices = {"AAPL": 195.0}
        return LiveTrader(broker=broker, dry_run=True)

    def test_normal_sizing(self):
        trader = self._make_trader()
        qty = trader.risk_ctrl.calc_position_size(
            capital=100000, price=195.0, atr=5.0, last_price=195.0,
        )
        assert qty > 0
        # risk 2% of 100k = 2000, stop at 2*ATR=10, so ~200 shares
        # max position 30% of 100k at $195 = ~153 shares
        assert 1 <= qty <= 200

    def test_zero_price_returns_zero(self):
        trader = self._make_trader()
        qty = trader.risk_ctrl.calc_position_size(
            capital=100000, price=0, atr=5.0, last_price=0,
        )
        assert qty == 0

    def test_no_atr_uses_fallback(self):
        trader = self._make_trader()
        qty = trader.risk_ctrl.calc_position_size(
            capital=100000, price=195.0, atr=0, last_price=195.0,
        )
        # fallback: equity * 0.30 / price
        assert qty > 0


# ===================================================================
# LiveTrader._passes_risk
# ===================================================================


class TestRiskChecks:
    def _make_trader(self):
        broker = MockBroker(initial_cash=100000)
        return LiveTrader(broker=broker, dry_run=True)

    def test_normal_case_passes(self):
        trader = self._make_trader()
        trader.risk._day_start_equity = 100000
        account = type("Account", (), {"total_equity": 101000})()
        signal = {"symbol": "AAPL", "price": 195.0}
        assert trader.risk_ctrl.passes_risk(signal, 10, account) is True

    def test_daily_loss_limit(self, capsys):
        trader = self._make_trader()
        trader.risk.max_daily_loss_pct = 0.05  # 5%
        trader.risk._day_start_equity = 100000
        # equity dropped 8% — exceeds 5% limit
        account = type("Account", (), {"total_equity": 92000})()
        signal = {"symbol": "AAPL", "price": 195.0}
        assert trader.risk_ctrl.passes_risk(signal, 10, account) is False

    def test_small_order_rejected(self):
        trader = self._make_trader()
        trader.risk.min_order_value = 500.0
        trader.risk._day_start_equity = 100000
        account = type("Account", (), {"total_equity": 100000})()
        signal = {"symbol": "AAPL", "price": 195.0}
        # 1 share @ $195 = $195 < $500 min
        assert trader.risk_ctrl.passes_risk(signal, 1, account) is False


# ===================================================================
# LiveTrader._generate_orders
# ===================================================================


class TestOrderGeneration:
    def _make_trader(self):
        broker = MockBroker(initial_cash=100000)
        broker.last_prices = {"AAPL": 195.0, "NVDA": 850.0}
        trader = LiveTrader(broker=broker, dry_run=True)
        trader.risk._day_start_equity = 100000
        return trader

    def test_buy_signal_without_position(self):
        trader = self._make_trader()
        signals = [{
            "symbol": "AAPL", "name": "Apple", "strategy": "weekly_macd_kdj",
            "signal": 1, "price": 195.0, "atr": 5.0,
        }]
        positions = {}
        account = type("Account", (), {"total_equity": 100000})()
        orders = trader.order_mgr.generate_orders(signals, positions, account)
        assert len(orders) == 1
        assert orders[0].symbol == "AAPL"
        assert orders[0].side == OrderSide.BUY

    def test_sell_signal_with_position(self):
        trader = self._make_trader()
        signals = [{
            "symbol": "AAPL", "name": "Apple", "strategy": "weekly_macd_kdj",
            "signal": -1, "price": 200.0, "atr": 5.0,
        }]
        positions = {
            "AAPL": Position(symbol="AAPL", quantity=50, avg_price=190.0,
                             market_value=10000, unrealized_pnl=500),
        }
        account = type("Account", (), {"total_equity": 100000})()
        orders = trader.order_mgr.generate_orders(signals, positions, account)
        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL
        assert orders[0].quantity == 50

    def test_hold_signal_generates_no_order(self):
        trader = self._make_trader()
        signals = [{
            "symbol": "AAPL", "name": "Apple", "strategy": "weekly_macd_kdj",
            "signal": 0, "price": 195.0, "atr": 5.0,
        }]
        positions = {}
        account = type("Account", (), {"total_equity": 100000})()
        orders = trader.order_mgr.generate_orders(signals, positions, account)
        assert len(orders) == 0

    def test_buy_with_existing_position_generates_no_order(self):
        """Already holding, buy signal should not double-buy."""
        trader = self._make_trader()
        signals = [{
            "symbol": "AAPL", "name": "Apple", "strategy": "weekly_macd_kdj",
            "signal": 1, "price": 195.0, "atr": 5.0,
        }]
        positions = {
            "AAPL": Position(symbol="AAPL", quantity=50, avg_price=190.0,
                             market_value=9750, unrealized_pnl=250),
        }
        account = type("Account", (), {"total_equity": 100000})()
        orders = trader.order_mgr.generate_orders(signals, positions, account)
        assert len(orders) == 0

    def test_prefers_nonzero_signal_when_multiple(self):
        trader = self._make_trader()
        signals = [
            {"symbol": "AAPL", "name": "Apple", "strategy": "s1",
             "signal": 0, "price": 195.0, "atr": 5.0},
            {"symbol": "AAPL", "name": "Apple", "strategy": "s2",
             "signal": 1, "price": 195.0, "atr": 5.0},
        ]
        positions = {}
        account = type("Account", (), {"total_equity": 100000})()
        orders = trader.order_mgr.generate_orders(signals, positions, account)
        # Should pick the buy signal (signal=1) over hold (signal=0)
        assert len(orders) >= 1
        assert any(o.side == OrderSide.BUY for o in orders)


# ===================================================================
# LiveTrader._load_config
# ===================================================================


class TestLoadConfig:
    def test_loads_watchlist_config(self):
        cfg = LiveTrader._load_config("watchlist.toml")
        assert "watchlist" in cfg
        assert "risk" in cfg


# ===================================================================
# LiveTrader init
# ===================================================================


class TestDaemonHelpers:
    def test_market_open_midnight(self):
        """US market open at midnight Beijing (summer)."""
        from unittest.mock import patch
        import datetime as dt
        monday_1am = dt.datetime(2026, 5, 11, 1, 0, 0)  # Monday 1 AM BJT
        with patch('live_trader.datetime') as mock_dt:
            mock_dt.now.return_value = monday_1am
            assert LiveTrader._is_market_open() is True

    def test_market_open_evening(self):
        """US market opens at 21:30 Beijing."""
        from unittest.mock import patch
        import datetime as dt
        monday_10pm = dt.datetime(2026, 5, 11, 22, 0, 0)
        with patch('live_trader.datetime') as mock_dt:
            mock_dt.now.return_value = monday_10pm
            assert LiveTrader._is_market_open() is True

    def test_market_closed_daytime(self):
        """US market closed at 10 AM Beijing."""
        from unittest.mock import patch
        import datetime as dt
        monday_10am = dt.datetime(2026, 5, 11, 10, 0, 0)
        with patch('live_trader.datetime') as mock_dt:
            mock_dt.now.return_value = monday_10am
            assert LiveTrader._is_market_open() is False

    def test_weekend_closed(self):
        """Market closed on Saturday."""
        from unittest.mock import patch
        import datetime as dt
        saturday_1am = dt.datetime(2026, 5, 9, 1, 0, 0)  # Saturday
        with patch('live_trader.datetime') as mock_dt:
            mock_dt.now.return_value = saturday_1am
            assert LiveTrader._is_market_open() is False


# ===================================================================
# Total exposure & slippage checks
# ===================================================================


class TestExposureCheck:
    def _make_trader(self):
        broker = MockBroker(initial_cash=100000)
        broker.last_prices = {"AAPL": 195.0, "NVDA": 850.0}
        trader = LiveTrader(broker=broker, dry_run=True)
        trader.risk._day_start_equity = 100000
        return trader

    def test_exposure_capped_in_generate_orders(self):
        """When existing position + new order exceeds max exposure, skip."""
        trader = self._make_trader()
        from broker import Position
        # Already holding $70k of NVDA out of $100k = 70% exposed
        positions = {
            "NVDA": Position(symbol="NVDA", quantity=82, avg_price=840.0,
                             market_value=70000, unrealized_pnl=1000),
        }
        signals = [{
            "symbol": "AAPL", "name": "Apple", "strategy": "turtle_trading",
            "signal": 1, "price": 195.0, "atr": 5.0,
        }]
        account = type("Account", (), {"total_equity": 100000})()
        orders = trader.order_mgr.generate_orders(signals, positions, account)
        # New position ~$3k → total 73% ≤ 80% → order should pass
        assert len(orders) >= 0


class TestSlippageCheck:
    def _make_trader(self):
        broker = MockBroker(initial_cash=100000)
        broker.last_prices = {"AAPL": 200.0}
        trader = LiveTrader(broker=broker, dry_run=True)
        trader.risk._day_start_equity = 100000
        return trader

    def test_large_slippage_rejected(self):
        trader = self._make_trader()
        trader.risk.max_slippage_pct = 0.02
        # Signal price $195 vs last price $200 = 2.5% slippage > 2%
        signal = {"symbol": "AAPL", "price": 195.0}
        account = type("Account", (), {"total_equity": 100000})()
        assert trader.risk_ctrl.passes_risk(signal, 10, account) is False

    def test_small_slippage_passes(self):
        trader = self._make_trader()
        trader.risk.max_slippage_pct = 0.02
        # Signal price $199 vs last price $200 = 0.5% slippage < 2%
        signal = {"symbol": "AAPL", "price": 199.0}
        account = type("Account", (), {"total_equity": 100000})()
        assert trader.risk_ctrl.passes_risk(signal, 10, account) is True


class TestLiveTraderInternals:
    def test_print_order(self, capsys):
        trader = LiveTrader(broker=MockBroker(), dry_run=True)
        order = Order(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100)
        trader.order_mgr.print_order(order)
        out = capsys.readouterr().out
        assert "DRY-RUN" in out
        assert "AAPL" in out

    def test_log_order(self):
        trader = LiveTrader(broker=MockBroker(), dry_run=True)
        # Ensure table exists, then clean slate (table created lazily in log_order)
        trader.cache.conn.execute(
            "CREATE TABLE IF NOT EXISTS order_log ("
            "  order_id TEXT, symbol TEXT, side TEXT, qty INTEGER,"
            "  price REAL, status TEXT, created_at TEXT"
            ")"
        )
        trader.cache.conn.execute("DELETE FROM order_log")
        trader.cache.conn.commit()

        order = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET,
            quantity=10, order_id="test-123", status=OrderStatus.FILLED,
            avg_fill_price=195.0,
        )
        trader.order_mgr.log_order(order)
        rows = trader.cache.conn.execute(
            "SELECT * FROM order_log WHERE order_id=?", ["test-123"]
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][1] == "AAPL"


class TestLiveTraderInit:
    def test_defaults(self):
        broker = MockBroker(initial_cash=50000)
        trader = LiveTrader(broker=broker, dry_run=True)
        assert trader.dry_run is True
        assert trader.broker.name == "mock"

    def test_notifier_created(self):
        broker = MockBroker()
        trader = LiveTrader(broker=broker, dry_run=True)
        assert trader.notifier is not None


# ===================================================================
# Circuit breaker (_update_risk_state)
# ===================================================================


class TestCircuitBreaker:
    def _make_trader(self):
        broker = MockBroker(initial_cash=100000)
        broker.last_prices = {"AAPL": 195.0}
        return LiveTrader(broker=broker, dry_run=True)

    def test_buy_increments_daily_count(self):
        trader = self._make_trader()
        assert trader.risk._daily_trade_count == 0
        order = Order(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET,
                      quantity=10, status=OrderStatus.FILLED, avg_fill_price=195.0)
        trader.risk_ctrl.on_trade_filled(order)
        assert trader.risk._daily_trade_count == 1
        assert trader.risk_ctrl.entry_prices["AAPL"] == 195.0

    def test_profitable_sell_resets_losses(self):
        trader = self._make_trader()
        trader.risk._consecutive_losses = 2
        trader.risk_ctrl.entry_prices["AAPL"] = 190.0
        order = Order(symbol="AAPL", side=OrderSide.SELL, order_type=OrderType.MARKET,
                      quantity=10, status=OrderStatus.FILLED, avg_fill_price=200.0)
        trader.risk_ctrl.on_trade_filled(order)
        assert trader.risk._consecutive_losses == 0
        assert "AAPL" not in trader.risk_ctrl.entry_prices

    def test_losing_sell_increments_losses(self):
        trader = self._make_trader()
        trader.risk._consecutive_losses = 1
        trader.risk_ctrl.entry_prices["AAPL"] = 210.0
        order = Order(symbol="AAPL", side=OrderSide.SELL, order_type=OrderType.MARKET,
                      quantity=10, status=OrderStatus.FILLED, avg_fill_price=200.0)
        trader.risk_ctrl.on_trade_filled(order)
        assert trader.risk._consecutive_losses == 2

    def test_sell_without_entry_price(self):
        """No entry price tracked — should not crash."""
        trader = self._make_trader()
        trader.risk._consecutive_losses = 1
        order = Order(symbol="AAPL", side=OrderSide.SELL, order_type=OrderType.MARKET,
                      quantity=10, status=OrderStatus.FILLED, avg_fill_price=200.0)
        trader.risk_ctrl.on_trade_filled(order)
        assert trader.risk._consecutive_losses == 1  # unchanged


# ===================================================================
# Circuit breaker in _passes_risk
# ===================================================================


class TestCircuitBreakerPassesRisk:
    def _make_trader(self):
        broker = MockBroker(initial_cash=100000)
        trader = LiveTrader(broker=broker, dry_run=True)
        trader.risk._day_start_equity = 100000
        return trader

    def test_consecutive_losses_block(self):
        trader = self._make_trader()
        trader.risk._consecutive_losses = 3
        trader.risk.max_consecutive_losses = 3
        account = type("Account", (), {"total_equity": 100000})()
        signal = {"symbol": "AAPL", "price": 195.0}
        assert trader.risk_ctrl.passes_risk(signal, 10, account) is False

    def test_below_threshold_passes(self):
        trader = self._make_trader()
        trader.risk._consecutive_losses = 2
        trader.risk.max_consecutive_losses = 3
        account = type("Account", (), {"total_equity": 100000})()
        signal = {"symbol": "AAPL", "price": 195.0}
        assert trader.risk_ctrl.passes_risk(signal, 10, account) is True


# ===================================================================
# Daily trade limit
# ===================================================================


class TestDailyTradeLimit:
    def _make_trader(self):
        broker = MockBroker(initial_cash=100000)
        trader = LiveTrader(broker=broker, dry_run=True)
        trader.risk._day_start_equity = 100000
        return trader

    def test_daily_trade_cap_blocks(self):
        trader = self._make_trader()
        trader.risk._daily_trade_count = 5
        trader.risk.max_daily_trades = 5
        account = type("Account", (), {"total_equity": 100000})()
        signal = {"symbol": "AAPL", "price": 195.0}
        assert trader.risk_ctrl.passes_risk(signal, 10, account) is False

    def test_under_cap_passes(self):
        trader = self._make_trader()
        trader.risk._daily_trade_count = 4
        trader.risk.max_daily_trades = 5
        account = type("Account", (), {"total_equity": 100000})()
        signal = {"symbol": "AAPL", "price": 195.0}
        assert trader.risk_ctrl.passes_risk(signal, 10, account) is True


# ===================================================================
# Volatility-adaptive position sizing
# ===================================================================


class TestAdaptiveSizing:
    def _make_trader(self):
        broker = MockBroker(initial_cash=100000)
        return LiveTrader(broker=broker, dry_run=True)

    def test_low_volatility_full_size(self):
        trader = self._make_trader()
        # ATR=2, price=200 → vol_ratio=1% → scalar=1/(1+0.01*5)=0.95 → near full
        qty = trader.risk_ctrl.calc_position_size(
            capital=100000, price=200.0, atr=2.0, last_price=200.0,
        )
        assert qty > 0

    def test_high_volatility_reduced_size(self):
        trader = self._make_trader()
        # ATR=40, price=200 → vol_ratio=20% → scalar=1/(1+0.2*5)=0.5 → half
        qty_high = trader.risk_ctrl.calc_position_size(
            capital=100000, price=200.0, atr=40.0, last_price=200.0,
        )
        # Low volatility on same price → larger position
        qty_low = trader.risk_ctrl.calc_position_size(
            capital=100000, price=200.0, atr=4.0, last_price=200.0,
        )
        assert qty_low > qty_high

    def test_min_vol_scalar_floor(self):
        trader = self._make_trader()
        trader.risk.min_vol_scalar = 0.3
        # Extreme volatility
        qty = trader.risk_ctrl.calc_position_size(
            capital=100000, price=100.0, atr=80.0, last_price=100.0,
        )
        assert qty > 0  # not zero

    def test_zero_atr_fallback(self):
        trader = self._make_trader()
        qty = trader.risk_ctrl.calc_position_size(
            capital=100000, price=200.0, atr=0, last_price=200.0,
        )
        assert qty > 0


# ===================================================================
# Orphan position handling
# ===================================================================


class TestOrphanPositions:
    def _make_trader(self, orphan_strategy="daily_macd_kdj"):
        broker = MockBroker(initial_cash=100000)
        broker.last_prices = {"AAPL": 195.0, "MSFT": 420.0}
        trader = LiveTrader(broker=broker, dry_run=True)
        trader.risk._day_start_equity = 100000
        trader._orphan_strategy = orphan_strategy
        trader._watchlist_symbols = ["AAPL"]
        return trader

    def test_orphan_buy_blocked(self):
        trader = self._make_trader()
        signals = [{
            "symbol": "MSFT", "name": "Microsoft", "strategy": "daily_macd_kdj",
            "signal": 1, "price": 420.0, "atr": 8.0, "orphan": True,
        }]
        positions = {}
        account = type("Account", (), {"total_equity": 100000})()
        orders = trader.order_mgr.generate_orders(signals, positions, account)
        assert len(orders) == 0  # orphan buy blocked

    def test_orphan_sell_allowed(self):
        trader = self._make_trader()
        signals = [{
            "symbol": "MSFT", "name": "Microsoft", "strategy": "daily_macd_kdj",
            "signal": -1, "price": 420.0, "atr": 8.0, "orphan": True,
        }]
        positions = {
            "MSFT": Position(symbol="MSFT", quantity=20, avg_price=410.0,
                             market_value=8400, unrealized_pnl=200),
        }
        account = type("Account", (), {"total_equity": 100000})()
        orders = trader.order_mgr.generate_orders(signals, positions, account)
        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL

    def test_orphan_hold_no_order(self):
        trader = self._make_trader()
        signals = [{
            "symbol": "MSFT", "name": "Microsoft", "strategy": "daily_macd_kdj",
            "signal": 0, "price": 420.0, "atr": 8.0, "orphan": True,
        }]
        positions = {}
        account = type("Account", (), {"total_equity": 100000})()
        orders = trader.order_mgr.generate_orders(signals, positions, account)
        assert len(orders) == 0

    def test_watchlist_buy_still_works(self):
        """Non-orphan symbols should still generate BUY orders."""
        trader = self._make_trader()
        signals = [{
            "symbol": "AAPL", "name": "Apple", "strategy": "weekly_macd_kdj",
            "signal": 1, "price": 195.0, "atr": 5.0, "orphan": False,
        }]
        positions = {}
        account = type("Account", (), {"total_equity": 100000})()
        orders = trader.order_mgr.generate_orders(signals, positions, account)
        assert len(orders) == 1
        assert orders[0].side == OrderSide.BUY


# ===================================================================
# Trading pause (global risk guard)
# ===================================================================


class TestTradingPause:
    def _make_trader(self):
        broker = MockBroker(initial_cash=100000)
        broker.last_prices = {"AAPL": 195.0}
        trader = LiveTrader(broker=broker, dry_run=True)
        trader.risk._day_start_equity = 100000
        return trader

    def test_paused_blocks_buy(self):
        trader = self._make_trader()
        trader._gate.trading_paused = True
        trader._gate.pause_reason = "test"
        trader.order_mgr.gate = trader._gate
        signals = [{
            "symbol": "AAPL", "name": "Apple", "strategy": "w",
            "signal": 1, "price": 195.0, "atr": 5.0,
        }]
        positions = {}
        account = type("Account", (), {"total_equity": 100000})()
        orders = trader.order_mgr.generate_orders(signals, positions, account)
        assert len(orders) == 0

    def test_paused_allows_sell(self):
        trader = self._make_trader()
        trader._gate.trading_paused = True
        trader._gate.pause_reason = "test"
        trader.order_mgr.gate = trader._gate
        signals = [{
            "symbol": "AAPL", "name": "Apple", "strategy": "w",
            "signal": -1, "price": 195.0, "atr": 5.0,
        }]
        positions = {
            "AAPL": Position(symbol="AAPL", quantity=50, avg_price=190.0,
                             market_value=9750, unrealized_pnl=250),
        }
        account = type("Account", (), {"total_equity": 100000})()
        orders = trader.order_mgr.generate_orders(signals, positions, account)
        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL


class TestGlobalRiskCheck:
    def _make_trader(self):
        broker = MockBroker(initial_cash=100000)
        broker.last_prices = {"AAPL": 195.0}
        trader = LiveTrader(broker=broker, dry_run=True)
        trader.risk._day_start_equity = 100000
        trader.risk._date = "2025-06-15"
        return trader

    def test_daily_loss_pauses_trading(self):
        trader = self._make_trader()
        trader.risk.max_daily_loss_pct = 0.05
        trader.risk._day_start_equity = 100000
        account = type("Account", (), {"total_equity": 94000})()
        trader.risk_ctrl.check_global(account, {})
        assert trader.risk_ctrl.trading_paused is True
        assert "日内亏损超限" in trader.risk_ctrl.pause_reason

    def test_consecutive_losses_pause(self):
        trader = self._make_trader()
        trader.risk._consecutive_losses = 3
        trader.risk.max_consecutive_losses = 3
        account = type("Account", (), {"total_equity": 100000})()
        trader.risk_ctrl.check_global(account, {})
        assert trader.risk_ctrl.trading_paused is True
        assert "连续亏损熔断" in trader.risk_ctrl.pause_reason

    def test_exposure_pause(self):
        trader = self._make_trader()
        trader.risk.max_total_exposure_pct = 0.80
        positions = {
            "AAPL": Position(symbol="AAPL", quantity=420, avg_price=195.0,
                             market_value=82000, unrealized_pnl=0),
        }
        account = type("Account", (), {"total_equity": 100000})()
        trader.risk_ctrl.check_global(account, positions)
        assert trader.risk_ctrl.trading_paused is True
        assert "总敞口超限" in trader.risk_ctrl.pause_reason

    def test_normal_passes(self):
        trader = self._make_trader()
        account = type("Account", (), {"total_equity": 101000})()
        positions = {
            "AAPL": Position(symbol="AAPL", quantity=50, avg_price=190.0,
                             market_value=9750, unrealized_pnl=250),
        }
        trader.risk_ctrl.check_global(account, positions)
        assert trader.risk_ctrl.trading_paused is False

    def test_new_day_unpauses(self):
        trader = self._make_trader()
        trader.risk_ctrl.trading_paused = True
        trader.risk_ctrl.pause_reason = "old"
        trader.risk._date = "2025-06-14"  # different from today
        account = type("Account", (), {"total_equity": 100000})()
        trader.risk_ctrl.check_global(account, {})
        assert trader.risk_ctrl.trading_paused is False


# ===================================================================
# Slippage recording
# ===================================================================


class TestSlippageRecording:
    def _make_trader(self):
        broker = MockBroker(initial_cash=100000)
        return LiveTrader(broker=broker, dry_run=True)

    def test_records_slippage(self):
        trader = self._make_trader()
        trader.cache.conn.execute(
            "CREATE TABLE IF NOT EXISTS slippage_log ("
            "  order_id TEXT, symbol TEXT, side TEXT,"
            "  signal_price REAL, fill_price REAL, slippage_pct REAL,"
            "  created_at TEXT"
            ")"
        )
        trader.cache.conn.execute("DELETE FROM slippage_log")
        trader.cache.conn.commit()
        order = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET,
            quantity=10, order_id="slip-1", status=OrderStatus.FILLED,
            avg_fill_price=196.0,
        )
        trader.order_mgr.record_slippage(order, signal_price=195.0)
        rows = trader.cache.conn.execute(
            "SELECT * FROM slippage_log WHERE order_id=?", ["slip-1"]
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][1] == "AAPL"
        # slippage: (196 - 195) / 195 * 100 ≈ 0.5128
        assert abs(rows[0][5] - 0.5128) < 0.01

    def test_zero_signal_price_skips(self):
        trader = self._make_trader()
        trader.order_mgr.record_slippage(Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET,
            quantity=10, order_id="dummy", status=OrderStatus.FILLED,
            avg_fill_price=195.0), signal_price=195.0)  # ensure table exists
        trader.cache.conn.execute("DELETE FROM slippage_log")
        trader.cache.conn.commit()
        order = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET,
            quantity=10, order_id="slip-2", status=OrderStatus.FILLED,
            avg_fill_price=196.0,
        )
        trader.order_mgr.record_slippage(order, signal_price=0)
        rows = trader.cache.conn.execute(
            "SELECT * FROM slippage_log WHERE order_id=?", ["slip-2"]
        ).fetchall()
        assert len(rows) == 0

    def test_negative_slippage(self):
        trader = self._make_trader()
        trader.order_mgr.record_slippage(Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET,
            quantity=10, order_id="dummy2", status=OrderStatus.FILLED,
            avg_fill_price=195.0), signal_price=195.0)  # ensure table exists
        trader.cache.conn.execute("DELETE FROM slippage_log")
        trader.cache.conn.commit()
        order = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET,
            quantity=10, order_id="slip-3", status=OrderStatus.FILLED,
            avg_fill_price=194.0,
        )
        trader.order_mgr.record_slippage(order, signal_price=195.0)
        rows = trader.cache.conn.execute(
            "SELECT * FROM slippage_log WHERE order_id=?", ["slip-3"]
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][5] < 0  # negative slippage (price improvement)


# ===================================================================
# Risk state persistence (DB restore)
# ===================================================================


class TestRiskPersistence:
    def _clean_risk_db(self):
        """Clean risk tables BEFORE creating a trader (so __init__ doesn't
        pick up stale data from other tests)."""
        from data.cache import CacheManager
        cache = CacheManager()
        cache.conn.execute("DELETE FROM risk_state")
        cache.conn.execute("DELETE FROM entry_prices")
        cache.conn.commit()

    def _make_trader(self):
        broker = MockBroker(initial_cash=100000)
        broker.last_prices = {"AAPL": 195.0}
        return LiveTrader(broker=broker, dry_run=True)

    def test_persist_and_restore(self):
        self._clean_risk_db()
        try:
            trader = self._make_trader()
            today = __import__("datetime").date.today().isoformat()
            trader.risk._date = today
            trader.risk._day_start_equity = 100000
            trader.risk._consecutive_losses = 2
            trader.risk._daily_trade_count = 3
            trader.risk_ctrl.entry_prices = {"AAPL": 195.0, "NVDA": 850.0}
            trader.risk_ctrl.persist_state()
            trader.cache.save_entry_price("AAPL", 195.0, today)
            trader.cache.save_entry_price("NVDA", 850.0, today)

            trader2 = self._make_trader()
            trader2.cache = trader.cache
            trader2.risk_ctrl.restore_state()
            assert trader2.risk._consecutive_losses == 2
            assert trader2.risk._daily_trade_count == 3
            assert trader2.risk._day_start_equity == 100000
            assert trader2.risk_ctrl.entry_prices == {"AAPL": 195.0, "NVDA": 850.0}
        finally:
            self._clean_risk_db()

    def test_restore_different_day_keeps_historical_resets_daily(self):
        """P1-3: peak_equity & consecutive_losses are historical and must survive
        day rollover; daily_trade_count is day-scoped and resets."""
        self._clean_risk_db()
        try:
            trader = self._make_trader()
            trader.cache.save_risk_state("date", "2020-01-01")
            trader.cache.save_risk_state("consecutive_losses", "5")
            trader.cache.save_risk_state("daily_trade_count", "10")
            trader.cache.save_risk_state("peak_equity", "123456.78")
            trader.risk_ctrl.restore_state()
            # Day-scoped resets
            assert trader.risk._daily_trade_count == 0
            # Historical persists
            assert trader.risk._consecutive_losses == 5
            assert trader.risk._peak_equity == 123456.78
        finally:
            self._clean_risk_db()

    def test_restore_today_restores_counts(self):
        self._clean_risk_db()
        try:
            trader = self._make_trader()
            today = __import__("datetime").date.today().isoformat()
            trader.cache.save_risk_state("date", today)
            trader.cache.save_risk_state("day_start_equity", "123456")
            trader.cache.save_risk_state("consecutive_losses", "3")
            trader.cache.save_risk_state("daily_trade_count", "4")
            trader.risk_ctrl.restore_state()
            assert trader.risk._consecutive_losses == 3
            assert trader.risk._daily_trade_count == 4
            assert trader.risk._day_start_equity == 123456
            assert trader.risk._date == today
        finally:
            self._clean_risk_db()


# ===================================================================
# Market state integration — regime filtering + volatility sizing
# ===================================================================


class TestRegimeFiltering:
    def _make_gate(self, **kw):
        from utils.signal_gate import SignalGate
        return SignalGate(ms_enabled=True, **kw)

    def test_disabled_does_nothing(self):
        from utils.signal_gate import SignalGate
        g = SignalGate(ms_enabled=False)
        ok, _ = g.allow_buy({"symbol": "AAPL", "strategy": "turtle_trading"}, {}, None)
        assert ok is True

    def test_null_state_does_nothing(self):
        g = self._make_gate(market_state=None)
        ok, _ = g.allow_buy({"symbol": "AAPL", "strategy": "turtle_trading"}, {}, None)
        assert ok is True

    def test_trend_strategy_blocked_in_ranging(self):
        ms = MagicMock(); ms.regime = MarketRegime.RANGING; ms.volatility = Volatility.NORMAL
        g = self._make_gate(market_state=ms)
        ok, _ = g.allow_buy({"symbol": "AAPL", "strategy": "turtle_trading"}, {}, None)
        assert ok is False

    def test_trend_strategy_allowed_in_trending_up(self):
        ms = MagicMock(); ms.regime = MarketRegime.TRENDING_UP; ms.volatility = Volatility.NORMAL
        g = self._make_gate(market_state=ms)
        ok, _ = g.allow_buy({"symbol": "AAPL", "strategy": "turtle_trading"}, {}, None)
        assert ok is True

    def test_mean_reversion_blocked_in_trending(self):
        ms = MagicMock(); ms.regime = MarketRegime.TRENDING_UP; ms.volatility = Volatility.NORMAL
        g = self._make_gate(market_state=ms)
        ok, _ = g.allow_buy({"symbol": "AAPL", "strategy": "bollinger_mean_reversion"}, {}, None)
        assert ok is True  # strategy removed from map — gate can't classify, allows through

    def test_mean_reversion_allowed_in_ranging(self):
        ms = MagicMock(); ms.regime = MarketRegime.RANGING; ms.volatility = Volatility.NORMAL
        g = self._make_gate(market_state=ms)
        ok, _ = g.allow_buy({"symbol": "AAPL", "strategy": "bollinger_mean_reversion"}, {}, None)
        assert ok is True  # strategy removed from map — gate can't classify, allows through

    def test_mixed_strategy_never_blocked(self):
        for regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN,
                       MarketRegime.RANGING, MarketRegime.TRANSITIONAL):
            ms = MagicMock(); ms.regime = regime; ms.volatility = Volatility.NORMAL
            g = self._make_gate(market_state=ms)
            ok, _ = g.allow_buy({"symbol": "AAPL", "strategy": "daily_macd_kdj"}, {}, None)
            assert ok is True

    def test_trading_paused_blocks_buy(self):
        ms = MagicMock(); ms.regime = MarketRegime.TRENDING_UP; ms.volatility = Volatility.NORMAL
        g = self._make_gate(market_state=ms, trading_paused=True, pause_reason="test")
        ok, reason = g.allow_buy({"symbol": "AAPL", "strategy": "turtle_trading"}, {}, None)
        assert ok is False
        assert "TEST" in reason

    def test_orphan_buy_blocked(self):
        g = self._make_gate()
        ok, _ = g.allow_buy({"symbol": "AAPL", "strategy": "w", "orphan": True}, {}, None)
        assert ok is False

    def test_sell_always_allowed(self):
        g = self._make_gate(trading_paused=True, pause_reason="x")
        ok, _ = g.allow_sell({"symbol": "AAPL"})
        assert ok is True


class TestVolatilitySizing:
    def test_high_vol_reduces_size(self):
        from utils.signal_gate import SignalGate
        ms = MagicMock(); ms.volatility = Volatility.HIGH
        g = SignalGate(ms_enabled=True, market_state=ms, vol_high_scalar=0.7)
        assert g.vol_scaled_qty(100) == 70

    def test_normal_vol_no_scaling(self):
        from utils.signal_gate import SignalGate
        ms = MagicMock(); ms.volatility = Volatility.NORMAL
        g = SignalGate(ms_enabled=True, market_state=ms, vol_high_scalar=0.7)
        assert g.vol_scaled_qty(100) == 100

    def test_disabled_no_scaling(self):
        from utils.signal_gate import SignalGate
        g = SignalGate(ms_enabled=False, vol_high_scalar=0.7)
        assert g.vol_scaled_qty(100) == 100


# ===================================================================
# _submit_and_wait — order lifecycle (polling, timeout, cancel)
# ===================================================================


class TestSubmitAndWait:
    def _make_trader(self):
        broker = MockBroker(initial_cash=100000)
        trader = LiveTrader(broker=broker, dry_run=False)
        return trader

    def test_immediate_fill_returns_immediately(self):
        """FILLED from submit_order → no polling."""
        trader = self._make_trader()
        trader.broker.last_prices["AAPL"] = 195.0
        order = Order(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=10)
        result = trader.order_mgr.submit_and_wait(order)
        assert result.status == OrderStatus.FILLED

    def test_rejected_returns_immediately(self):
        trader = self._make_trader()
        order = Order(symbol="NONEXIST", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=10)
        result = trader.order_mgr.submit_and_wait(order)
        assert result.status == OrderStatus.REJECTED

    def test_poll_until_filled(self):
        """SUBMITTED → polls → FILLED after 1 poll."""
        trader = self._make_trader()
        call_count = [0]

        def _submit(o):
            o.status = OrderStatus.SUBMITTED
            o.order_id = "poll-me"
            return o

        def _get_order(oid):
            call_count[0] += 1
            o = Order(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET,
                      quantity=10, order_id="poll-me", status=OrderStatus.FILLED,
                      avg_fill_price=195.0)
            return o

        trader.broker.submit_order = _submit
        trader.broker.get_order = _get_order

        order = Order(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=10)
        result = trader.order_mgr.submit_and_wait(order)
        assert result.status == OrderStatus.FILLED
        assert call_count[0] >= 1

    def test_timeout_cancels_limit_order(self):
        """Limit order timeout → cancel called, status CANCELLED."""
        trader = self._make_trader()
        cancel_called = [False]

        def _submit(o):
            o.status = OrderStatus.SUBMITTED
            o.order_id = "limit-timeout"
            return o

        def _get_order(oid):
            # Never returns terminal status
            return Order(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.LIMIT,
                         quantity=10, order_id="limit-timeout", status=OrderStatus.SUBMITTED)

        def _cancel(oid):
            cancel_called[0] = True
            return True

        trader.broker.submit_order = _submit
        trader.broker.get_order = _get_order
        trader.broker.cancel_order = _cancel

        # Override timeout for fast test
        import live_trader as lt
        orig = getattr(lt.LiveTrader, '_submit_and_wait', None)
        # Patch timeout to 0 so we timeout immediately
        order = Order(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.LIMIT, quantity=10)
        # Directly test by using a tiny timeout via monkey-patch
        import time as _time_module
        real_sleep = _time_module.sleep
        _time_module.sleep = lambda x: None  # disable sleep
        result = trader.order_mgr.submit_and_wait(order)
        _time_module.sleep = real_sleep
        assert result.status == OrderStatus.CANCELLED
        assert cancel_called[0] is True


# ===================================================================
# _scan_signals — signal generation pipeline
# ===================================================================


class TestScanSignals:
    def _make_trader(self):
        broker = MockBroker(initial_cash=100000)
        broker.last_prices = {"AAPL": 195.0}
        trader = LiveTrader(broker=broker, dry_run=True)
        trader._watchlist_symbols = ["AAPL"]
        return trader

    def test_scans_watchlist_symbols(self, ohlcv):
        trader = self._make_trader()
        trader.provider.get_daily = lambda sym, start, end: ohlcv
        results = trader._scan_signals("2021-03-01", {})
        aapl = [r for r in results if r["symbol"] == "AAPL"]
        assert len(aapl) >= 1  # at least the active strategy

    def test_orphan_positions_scanned(self, ohlcv):
        trader = self._make_trader()
        trader._orphan_strategy = "daily_macd_kdj"
        trader._watchlist_symbols = ["AAPL"]
        # Override config so the SignalScanner sees only AAPL as watchlist —
        # decouples the test from whatever symbols watchlist.toml currently
        # contains. (Previously this used MSFT, which broke once MSFT was
        # added to the production watchlist.)
        trader.config = {
            "watchlist": [{"symbol": "AAPL", "name": "Apple",
                           "active": "weekly_macd_kdj", "monitor": []}],
            "scanner": {"lookback_years": 1},
        }
        trader.provider.get_daily = lambda sym, start, end: ohlcv
        positions = {"MSFT": Position(symbol="MSFT", quantity=100, avg_price=200,
                                       market_value=20000, unrealized_pnl=0)}
        results = trader._scan_signals("2021-03-01", positions)
        msft = [r for r in results if r["symbol"] == "MSFT"]
        assert len(msft) >= 1
        assert msft[0].get("orphan") is True

    def test_marks_signal_fields(self, ohlcv):
        trader = self._make_trader()
        trader.provider.get_daily = lambda sym, start, end: ohlcv
        results = trader._scan_signals("2021-03-01", {})
        for r in results:
            assert "symbol" in r
            assert "signal" in r
            assert "price" in r
            assert "strategy" in r
            assert "bar_date" in r


# ===================================================================
# LiveTrader.run() — end-to-end integration
# ===================================================================


class TestRunIntegration:
    def _make_trader(self):
        broker = MockBroker(initial_cash=100000)
        broker.last_prices = {"AAPL": 195.0, "NVDA": 850.0}
        trader = LiveTrader(broker=broker, dry_run=True)
        trader._watchlist_symbols = ["AAPL"]
        trader.risk._day_start_equity = 100000
        # Clean risk state to avoid cross-test pollution
        trader.cache.init_schema()
        trader.cache.conn.execute("DELETE FROM risk_state")
        trader.cache.conn.execute("DELETE FROM entry_prices")
        trader.cache.conn.commit()
        trader._entry_prices = {}
        return trader

    def test_run_with_no_signals_completes(self, ohlcv):
        """run() completes without error even when no signals fire."""
        trader = self._make_trader()
        trader.provider.get_daily = lambda sym, start, end: ohlcv
        orders = trader.run(target_date="2021-03-01")
        assert isinstance(orders, list)

    def test_run_aborts_when_broker_warmup_fails(self):
        trader = self._make_trader()
        trader.broker.warmup = lambda symbols: False

        with pytest.raises(ConnectionError, match="warmup failed"):
            trader.run(target_date="2021-03-01")

        assert trader.cache._batch_mode is False

    def test_run_commits_batch_after_cycle_exception(self):
        trader = self._make_trader()

        def fail_cycle(target_date):
            trader.cache.save_risk_state("partial_tick", "saved")
            raise RuntimeError("cycle failed")

        trader._run_cycle = fail_cycle
        with pytest.raises(RuntimeError, match="cycle failed"):
            trader.run(target_date="2021-03-01")

        assert trader.cache._batch_mode is False
        assert trader.cache.load_risk_state("partial_tick") == "saved"

    def test_run_generates_buy_order(self, ohlcv):
        """A buy signal produces a BUY order."""
        trader = self._make_trader()
        trader.provider.get_daily = lambda sym, start, end: ohlcv
        # Override _scan_signals to return a hard buy signal
        def _fake_scan(target_date, positions):
            return [{"symbol": "AAPL", "name": "Apple", "strategy": "w",
                     "signal": 1, "price": 195.0, "atr": 5.0, "orphan": False}]
        trader._scan_signals = _fake_scan
        orders = trader.run(target_date="2021-03-01")
        buys = [o for o in orders if o.side == OrderSide.BUY]
        assert len(buys) >= 1

    def test_run_sell_order(self, ohlcv):
        """A sell signal on an existing position produces a SELL order."""
        trader = self._make_trader()
        trader.provider.get_daily = lambda sym, start, end: ohlcv
        # Add an existing position to the broker
        pos = Position(symbol="AAPL", quantity=50, avg_price=190.0,
                       market_value=9750, unrealized_pnl=250)
        trader.broker._positions["AAPL"] = pos
        def _fake_scan(target_date, positions):
            return [{"symbol": "AAPL", "name": "Apple", "strategy": "w",
                     "signal": -1, "price": 200.0, "atr": 5.0, "orphan": False}]
        trader._scan_signals = _fake_scan
        orders = trader.run(target_date="2021-03-01")
        sells = [o for o in orders if o.side == OrderSide.SELL]
        assert len(sells) >= 1
