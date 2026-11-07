"""Tests for broker/futu.py — symbol mapping, broker interface contract."""

from unittest.mock import patch, MagicMock

import pytest
from broker.futu import (
    FutuBroker,
    _to_futu_symbol,
    _to_market,
    _from_futu_side,
    _to_futu_side,
    _from_futu_status,
    _to_futu_order_type,
)
from broker.base import (
    Order, OrderSide, OrderStatus, OrderType, Account, Position,
)
from futu import TrdSide, OrderType as FutuOrderType, OrderStatus as FutuOrderStatus, TrdMarket


# ===================================================================
# Symbol mapping
# ===================================================================


class TestSymbolMapping:
    def test_us_stock(self):
        assert _to_futu_symbol("AAPL") == "US.AAPL"
        assert _to_futu_symbol("NVDA") == "US.NVDA"

    def test_hk_stock(self):
        assert _to_futu_symbol("HK.00700") == "HK.00700"

    def test_cn_etf(self):
        assert _to_futu_symbol("510300") == "SH.510300"
        assert _to_futu_symbol("159919") == "SZ.159919"

    def test_already_prefixed(self):
        assert _to_futu_symbol("US.TSLA") == "US.TSLA"
        assert _to_futu_symbol("SH.510050") == "SH.510050"

    def test_shenzhen_code(self):
        """6-digit codes starting with 0/1/2/3 are SZSE."""
        assert _to_futu_symbol("000001") == "SZ.000001"
        assert _to_futu_symbol("300750") == "SZ.300750"


class TestMarketMapping:
    def test_us_market(self):
        assert _to_market("AAPL") == TrdMarket.US

    def test_hk_market(self):
        assert _to_market("HK.00700") == TrdMarket.HK

    def test_cn_market(self):
        assert _to_market("510300") == TrdMarket.CN


# ===================================================================
# Side / type / status mapping
# ===================================================================


class TestSideMapping:
    def test_buy(self):
        assert _to_futu_side(OrderSide.BUY) == TrdSide.BUY
        assert _from_futu_side(TrdSide.BUY) == OrderSide.BUY

    def test_sell(self):
        assert _to_futu_side(OrderSide.SELL) == TrdSide.SELL
        assert _from_futu_side(TrdSide.SELL) == OrderSide.SELL


class TestOrderTypeMapping:
    def test_market(self):
        assert _to_futu_order_type(OrderType.MARKET) == FutuOrderType.MARKET

    def test_limit(self):
        assert _to_futu_order_type(OrderType.LIMIT) == FutuOrderType.NORMAL


class TestStatusMapping:
    def test_filled_all(self):
        assert _from_futu_status(FutuOrderStatus.FILLED_ALL) == OrderStatus.FILLED

    def test_filled_part(self):
        assert _from_futu_status(FutuOrderStatus.FILLED_PART) == OrderStatus.PARTIAL

    def test_submitted(self):
        assert _from_futu_status(FutuOrderStatus.SUBMITTED) == OrderStatus.SUBMITTED

    def test_cancelled(self):
        assert _from_futu_status(FutuOrderStatus.CANCELLED_ALL) == OrderStatus.CANCELLED

    def test_failed(self):
        assert _from_futu_status(FutuOrderStatus.FAILED) == OrderStatus.REJECTED


# ===================================================================
# FutuBroker interface
# ===================================================================


class TestFutuBrokerInterface:
    def test_initialization(self):
        broker = FutuBroker(host="127.0.0.1", port=11111, initial_cash=10000)
        assert broker.name == "futu"
        assert broker._initial_cash == 10000
        assert broker._host == "127.0.0.1"
        assert broker._port == 11111

    def test_implements_broker_interface(self):
        from broker.base import Broker
        assert issubclass(FutuBroker, Broker)

    def test_get_account_no_connection(self):
        """Without a connected trade context, account state is unavailable."""
        broker = FutuBroker(initial_cash=10000)
        with pytest.raises(ConnectionError, match="no trade contexts"):
            broker.get_account()

    def test_get_positions_no_connection(self):
        broker = FutuBroker()
        with pytest.raises(ConnectionError, match="no trade contexts"):
            broker.get_positions()

    def test_warmup_fails_if_any_market_context_is_missing(self):
        broker = FutuBroker()
        with patch.object(
                broker, "_get_trade_ctx",
                side_effect=lambda symbol: object() if symbol == "AAPL" else None):
            assert broker.warmup(["AAPL", "HK.00700"]) is False

    def test_get_account_api_failure_raises(self):
        broker = FutuBroker()
        ctx = MagicMock()
        ctx.accinfo_query.return_value = (-1, "account unavailable")
        broker._trade_ctxs[TrdMarket.US] = ctx

        with pytest.raises(ConnectionError, match="account snapshot unavailable"):
            broker.get_account()

    def test_get_account_returns_valid_snapshot(self):
        import pandas as pd

        broker = FutuBroker(initial_cash=10000)
        ctx = MagicMock()
        ctx.accinfo_query.return_value = (0, pd.DataFrame([{
            "total_assets": 12500, "cash": 8000, "frozen_cash": 500,
        }]))
        broker._trade_ctxs[TrdMarket.US] = ctx

        account = broker.get_account()

        assert isinstance(account, Account)
        assert account.total_equity == 12500
        assert account.available_cash == 8000
        assert account.frozen_margin == 500
        assert account.total_pnl == 2500

    def test_get_positions_api_failure_raises(self):
        broker = FutuBroker()
        ctx = MagicMock()
        ctx.position_list_query.return_value = (-1, "positions unavailable")
        broker._trade_ctxs[TrdMarket.US] = ctx

        with pytest.raises(ConnectionError, match="positions snapshot unavailable"):
            broker.get_positions()

    def test_get_positions_does_not_return_partial_market_snapshot(self):
        import pandas as pd

        broker = FutuBroker()
        us_ctx = MagicMock()
        us_ctx.position_list_query.return_value = (0, pd.DataFrame([{
            "code": "US.AAPL", "qty": 10, "cost_price": 100,
        }]))
        hk_ctx = MagicMock()
        hk_ctx.position_list_query.return_value = (-1, "market unavailable")
        broker._trade_ctxs[TrdMarket.US] = us_ctx
        broker._trade_ctxs[TrdMarket.HK] = hk_ctx

        with pytest.raises(ConnectionError, match="positions snapshot unavailable"):
            broker.get_positions()

    def test_cancel_order_no_connection(self):
        broker = FutuBroker()
        assert broker.cancel_order("nonexistent") is False

    def test_get_order_no_connection(self):
        broker = FutuBroker()
        assert broker.get_order("nonexistent") is None

    def test_submit_order_without_connection(self):
        """Without FutuOpenD, order should be REJECTED gracefully."""
        broker = FutuBroker()
        order = Order(
            symbol="AAPL", side=OrderSide.BUY,
            order_type=OrderType.MARKET, quantity=10,
        )
        # Mock _get_trade_ctx to simulate FutuOpenD not running
        with patch.object(broker, '_get_trade_ctx', return_value=None):
            result = broker.submit_order(order)
        assert result.status == OrderStatus.REJECTED
        assert "error" in result.broker_data

    def test_last_prices_dict(self):
        broker = FutuBroker()
        assert broker.last_prices == {}
        broker.last_prices["AAPL"] = 195.0
        assert broker.last_prices["AAPL"] == 195.0

    def test_disconnect_cleanup(self):
        broker = FutuBroker()
        broker.disconnect()  # should not raise


class TestRefreshPrices:
    def test_no_connection_returns_empty(self):
        broker = FutuBroker()
        broker.connect = lambda: None  # prevent real connection
        broker._quote_ctx = None
        result = broker.refresh_prices(["AAPL"])
        assert isinstance(result, dict)
        assert result == {}

    def test_mocked_prices_filled(self):
        """refresh_prices with mocked quote ctx fills last_prices."""
        broker = FutuBroker()
        broker._quote_ctx = MagicMock()
        # Mock get_market_snapshot to return a DataFrame-like response
        import pandas as pd
        data = pd.DataFrame([{"code": "US.AAPL", "last_price": 195.0}])
        broker._quote_ctx.get_market_snapshot.return_value = (0, data)
        result = broker.refresh_prices(["AAPL"])
        assert result.get("AAPL") == 195.0
        # Second disconnect is also safe
        broker.disconnect()
