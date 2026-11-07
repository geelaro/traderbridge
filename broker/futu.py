"""FutuBroker — 富途 OpenAPI 券商适配器.

Supports simulated trading (模拟盘) via FutuOpenD gateway.

Setup
-----
1. Install FutuOpenD (https://www.futunn.com/download/openAPI)
2. Launch FutuOpenD, log in with your Futu account
3. In FutuOpenD settings, enable "模拟盘" trading
4. Default simulated trading port: 11111

Usage
-----
    from broker.futu import FutuBroker
    from live_trader import LiveTrader

    broker = FutuBroker(host='127.0.0.1', port=11111, initial_cash=10000)
    trader = LiveTrader(broker=broker)
    trader.run()
"""

from typing import Dict, List, Optional
import logging
import time

from futu import (
    OpenSecTradeContext,
    OpenQuoteContext,
    SimAccType,
    TrdEnv,
    TrdMarket,
    TrdSide,
    OrderType as FutuOrderType,
    OrderStatus as FutuOrderStatus,
    ModifyOrderOp,
)

from .base import (
    Account,
    Broker,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)


# ---------------------------------------------------------------------------
# Symbol mapping: traderbridge format ↔ Futu format
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def _to_futu_symbol(symbol: str) -> str:
    """Convert traderbridge symbol to Futu format (US.AAPL, HK.00700, SH.510300)."""
    s = symbol.upper().strip()
    if s.startswith("US."):
        return s
    if s.startswith("HK."):
        return s
    if s[:2] in ("SH", "SZ"):
        if "." not in s:
            return f"{s[:2]}.{s[2:]}"
        return s
    if s.isdigit() and len(s) == 6:
        # A-share code
        prefix = "SH" if s[0] in ("5", "6", "9") else "SZ"
        return f"{prefix}.{s}"
    # Default: US stock
    return f"US.{s}"


def _to_market(symbol: str) -> int:
    """Map Futu symbol to TrdMarket enum."""
    s = _to_futu_symbol(symbol)
    prefix = s.split(".")[0].upper()
    if prefix in ("US", "USO", "USN"):
        return TrdMarket.US
    if prefix == "HK":
        return TrdMarket.HK
    if prefix in ("SH", "SZ"):
        return TrdMarket.CN
    return TrdMarket.US  # default


def _from_futu_side(futu_side) -> OrderSide:
    return OrderSide.BUY if futu_side == TrdSide.BUY else OrderSide.SELL


def _to_futu_side(side: OrderSide):
    return TrdSide.BUY if side == OrderSide.BUY else TrdSide.SELL


def _to_futu_order_type(ot: OrderType):
    return FutuOrderType.MARKET if ot == OrderType.MARKET else FutuOrderType.NORMAL


def _from_futu_status(futu_status) -> OrderStatus:
    """Map Futu order status to our OrderStatus enum."""
    mapping = {
        FutuOrderStatus.WAITING_SUBMIT: OrderStatus.PENDING,
        FutuOrderStatus.SUBMITTING: OrderStatus.SUBMITTED,
        FutuOrderStatus.SUBMITTED: OrderStatus.SUBMITTED,
        FutuOrderStatus.FILLED_PART: OrderStatus.PARTIAL,
        FutuOrderStatus.FILLED_ALL: OrderStatus.FILLED,
        FutuOrderStatus.CANCELLED_PART: OrderStatus.CANCELLED,
        FutuOrderStatus.CANCELLED_ALL: OrderStatus.CANCELLED,
        FutuOrderStatus.FAILED: OrderStatus.REJECTED,
        FutuOrderStatus.DISABLED: OrderStatus.REJECTED,
        FutuOrderStatus.DELETED: OrderStatus.CANCELLED,
    }
    return mapping.get(futu_status, OrderStatus.REJECTED)


# ---------------------------------------------------------------------------
# FutuBroker
# ---------------------------------------------------------------------------


class FutuBroker(Broker):
    """Futu OpenAPI broker adapter — simulated trading.

    Parameters
    ----------
    host : str
        FutuOpenD host (default 127.0.0.1).
    port : int
        FutuOpenD port (default 11111 for simulated trading).
    initial_cash : float
        Reference initial cash for P&L tracking (account ground truth from Futu).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11111,
        initial_cash: float = 10000.0,
        sim_acc_type: SimAccType = SimAccType.STOCK_AND_OPTION,
    ):
        super().__init__()
        self._host = host
        self._port = port
        self._initial_cash = initial_cash
        self._sim_acc_type = sim_acc_type
        self._acc_ids: Dict[int, int] = {}  # resolved lazily per market
        self._quote_ctx: Optional[OpenQuoteContext] = None
        self._trade_ctxs: Dict[int, OpenSecTradeContext] = {}
        self.last_prices: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self):
        """Establish connections to FutuOpenD.

        Raises ConnectionError if FutuOpenD is not reachable.
        """
        try:
            self._quote_ctx = OpenQuoteContext(host=self._host, port=self._port)
            self._quote_ctx.start()
        except Exception as e:
            raise ConnectionError(
                f"无法连接 FutuOpenD ({self._host}:{self._port}): {e}"
            ) from e

    def disconnect(self):
        """Close all connections."""
        if self._quote_ctx is not None:
            self._quote_ctx.stop()
            self._quote_ctx = None
        for ctx in self._trade_ctxs.values():
            ctx.close()
        self._trade_ctxs.clear()

    def __del__(self):
        try:
            self.disconnect()
        except Exception:
            pass

    def _get_trade_ctx(self, symbol: str):
        """Get or create a trade context for the symbol's market.

        Returns None if connection fails (FutuOpenD not running).
        """
        market = _to_market(symbol)
        if market not in self._trade_ctxs:
            try:
                ctx = OpenSecTradeContext(
                    host=self._host,
                    port=self._port,
                    filter_trdmarket=market,
                )
                self._resolve_acc_id(ctx, market)
                self._trade_ctxs[market] = ctx
            except Exception as e:
                logger.warning("_get_trade_ctx 失败 [%s] market=%s: %s", symbol, market, e)
                return None
        return self._trade_ctxs[market]

    def _resolve_acc_id(self, ctx, market: int) -> None:
        """Resolve acc_id for the configured sim_acc_type.

        Calls get_acc_list(), filters for SIMULATE + sim_acc_type match,
        and stores the result for this market.
        Raises RuntimeError if no matching SIMULATE account found —
        we NEVER fall back to a real account.
        """
        if market in self._acc_ids:
            return
        try:
            ret, data = ctx.get_acc_list()
            if ret == 0 and not data.empty:
                mask = (data["trd_env"] == TrdEnv.SIMULATE) & (data["sim_acc_type"] == self._sim_acc_type)
                if mask.any():
                    self._acc_ids[market] = int(data[mask]["acc_id"].values[0])
                    return
                logger.warning("账户列表: %d 个账户, 无匹配 SIMULATE+%s", len(data), self._sim_acc_type)
        except Exception as e:
            logger.error("get_acc_list 失败: %s", e)
            raise RuntimeError(
                f"无法获取模拟账户列表 (FutuOpenD 是否已启动？): {e}"
            ) from e
        raise RuntimeError(
            f"未找到模拟账户 (sim_acc_type={self._sim_acc_type})，"
            f"请确认 FutuOpenD 已启动模拟盘功能"
        )

    def _acc_id_for(self, market: int) -> int:
        """Return the simulated account ID resolved for a trade market."""
        try:
            return self._acc_ids[market]
        except KeyError as exc:
            raise ConnectionError(
                f"Futu simulated account ID unavailable for market {market}"
            ) from exc

    # ------------------------------------------------------------------
    # Broker identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "futu"

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def warmup(self, symbols: List[str]) -> bool:
        """Pre-create trade contexts for the markets of given symbols.

        Call before get_account()/get_positions() to ensure trade contexts exist.
        Returns True only when every requested market context is available.
        """
        market_symbols = {}
        for symbol in symbols:
            market_symbols.setdefault(_to_market(symbol), symbol)
        if not market_symbols:
            return False

        failed_markets = [
            market for market, symbol in market_symbols.items()
            if self._get_trade_ctx(symbol) is None
        ]
        if failed_markets:
            logger.error("Futu warmup failed for markets: %s", failed_markets)
            return False
        return True

    def get_account(self) -> Account:
        """Query account info from Futu — aggregates across all markets."""
        if not self._trade_ctxs:
            raise ConnectionError("FutuBroker has no trade contexts; warmup failed or was skipped")

        equity_parts = []
        cash_parts = []
        frozen = 0.0

        for market, ctx in list(self._trade_ctxs.items()):
            if ctx is None:
                raise ConnectionError(f"Futu trade context missing for market {market}")
            try:
                ret, data = ctx.accinfo_query(
                    trd_env=TrdEnv.SIMULATE,
                    acc_id=self._acc_id_for(market),
                )
            except Exception as exc:
                logger.warning("Futu account query failed for market %s: %s", market, exc)
                raise ConnectionError(
                    f"Futu account query failed for market {market}"
                ) from exc
            if ret != 0 or data is None or data.empty:
                raise ConnectionError(
                    f"Futu account snapshot unavailable for market {market}: {data}"
                )
            row = data.iloc[0]
            equity_parts.append(float(row.get("total_assets", 0)))
            cash_parts.append(float(row.get("cash", 0)))
            frozen += float(row.get("frozen_cash", 0))

        total_equity = sum(equity_parts)
        available_cash = sum(cash_parts)
        total_pnl = total_equity - self._initial_cash

        return Account(
            total_equity=total_equity,
            available_cash=available_cash,
            frozen_margin=frozen,
            total_pnl=total_pnl,
            currency="USD",
        )

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_positions(self) -> List[Position]:
        """Query current positions from Futu."""
        if not self._trade_ctxs:
            raise ConnectionError("FutuBroker has no trade contexts; warmup failed or was skipped")

        results = []
        for market, ctx in list(self._trade_ctxs.items()):
            if ctx is None:
                raise ConnectionError(f"Futu trade context missing for market {market}")
            try:
                ret, data = ctx.position_list_query(
                    trd_env=TrdEnv.SIMULATE,
                    acc_id=self._acc_id_for(market),
                )
            except Exception as exc:
                logger.warning("Futu positions query failed for market %s: %s", market, exc)
                raise ConnectionError(
                    f"Futu positions query failed for market {market}"
                ) from exc
            if ret != 0 or data is None:
                raise ConnectionError(
                    f"Futu positions snapshot unavailable for market {market}: {data}"
                )
            for _, row in data.iterrows():
                futu_sym = row.get("code", "")
                sym = futu_sym.split(".")[-1]  # US.AAPL → AAPL
                qty = int(row.get("qty", 0))
                if qty == 0:
                    continue
                avg_price = float(row.get("cost_price", 0))
                last = self.last_prices.get(sym, avg_price)
                mv = qty * last
                upnl = (last - avg_price) * qty
                results.append(Position(
                    symbol=sym,
                    quantity=qty,
                    avg_price=avg_price,
                    market_value=mv,
                    unrealized_pnl=upnl,
                ))
        return results

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def submit_order(self, order: Order) -> Order:
        """Submit an order to Futu. Returns updated Order."""
        futu_sym = _to_futu_symbol(order.symbol)
        ctx = self._get_trade_ctx(order.symbol)
        if ctx is None:
            order.status = OrderStatus.REJECTED
            order.broker_data = {"error": "FutuOpenD not connected"}
            return order

        try:
            ret, data = ctx.place_order(
                price=order.price or 0.0,
                qty=order.quantity,
                code=futu_sym,
                trd_side=_to_futu_side(order.side),
                order_type=_to_futu_order_type(order.order_type),
                trd_env=TrdEnv.SIMULATE,
                acc_id=self._acc_id_for(_to_market(order.symbol)),
            )
            if ret == 0 and not data.empty:
                row = data.iloc[0]
                order.order_id = str(row.get("order_id", ""))
                order.status = _from_futu_status(row.get("order_status", FutuOrderStatus.SUBMITTED))
                order.filled_qty = int(row.get("dealt_qty", 0))
                order.avg_fill_price = float(row.get("dealt_avg_price", 0))
                order.updated_at = time.strftime("%Y-%m-%dT%H:%M:%S")
                order.broker_data = row.to_dict()
            else:
                order.status = OrderStatus.REJECTED
                order.broker_data = {"error": str(data) if data else "unknown"}
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.broker_data = {"error": str(e)}

        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        for market, ctx in list(self._trade_ctxs.items()):
            if ctx is None:
                continue
            try:
                ret, data = ctx.cancel_order(
                    order_id=order_id,
                    trd_env=TrdEnv.SIMULATE,
                    acc_id=self._acc_id_for(market),
                )
                if ret == 0:
                    return True
            except Exception:
                continue
        return False

    def get_order(self, order_id: str) -> Optional[Order]:
        """Query order status from Futu."""
        for market, ctx in list(self._trade_ctxs.items()):
            if ctx is None:
                continue
            try:
                ret, data = ctx.order_list_query(
                    order_id=order_id,
                    trd_env=TrdEnv.SIMULATE,
                    acc_id=self._acc_id_for(market),
                )
                if ret == 0 and not data.empty:
                    row = data.iloc[0]
                    futu_sym = row.get("code", "")
                    sym = futu_sym.split(".")[-1]
                    return Order(
                        symbol=sym,
                        side=_from_futu_side(row.get("trd_side", TrdSide.BUY)),
                        order_type=OrderType.MARKET,
                        quantity=int(row.get("qty", 0)),
                        order_id=order_id,
                        status=_from_futu_status(row.get("order_status", FutuOrderStatus.SUBMITTED)),
                        filled_qty=int(row.get("dealt_qty", 0)),
                        avg_fill_price=float(row.get("dealt_avg_price", 0)),
                        broker_data=row.to_dict(),
                    )
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def refresh_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Refresh requested quotes without exposing stale values on failure."""
        requested = {}
        stale_keys = set()
        for symbol in symbols:
            futu_symbol = _to_futu_symbol(symbol)
            requested.setdefault(futu_symbol, symbol.strip().upper())
            stale_keys.add(symbol.strip().upper())
            stale_keys.add(futu_symbol.split(".")[-1])
        for key in stale_keys:
            self.last_prices.pop(key, None)
        if not requested:
            return self.last_prices

        if self._quote_ctx is None:
            self.connect()
        if self._quote_ctx is None:
            return self.last_prices

        futu_symbols = list(requested)
        try:
            ret, data = self._quote_ctx.get_market_snapshot(futu_symbols)
            if ret != 0 or data is None or data.empty:
                logger.warning("Futu quote snapshot unavailable (ret=%s): %s", ret, data)
                return self.last_prices
            for _, row in data.iterrows():
                futu_sym = row.get("code", "")
                sym = futu_sym.split(".")[-1]
                price = float(row.get("last_price", 0))
                if price > 0:
                    self.last_prices[sym] = price
        except Exception as exc:
            logger.warning("Futu quote snapshot failed: %s", exc)
        return self.last_prices

    def get_historical_kline(self, symbol: str, start: str, end: str):
        """Fetch daily OHLCV kline from FutuOpenD. Returns empty DataFrame on failure."""
        import pandas as pd
        from futu import KLType, AuType

        if self._quote_ctx is None:
            self.connect()
        if self._quote_ctx is None:
            logger.warning("FutuOpenD not connected — cannot fetch kline")
            return pd.DataFrame()

        code = _to_futu_symbol(symbol)
        try:
            ret, data, _ = self._quote_ctx.request_history_kline(
                code, start=start, end=end,
                ktype=KLType.K_DAY, autype=AuType.QFQ, max_count=2000,
            )
            if ret == 0 and not data.empty:
                df = pd.DataFrame({
                    "date": pd.to_datetime(data["time_key"]),
                    "Open": data["open"], "High": data["high"],
                    "Low": data["low"], "Close": data["close"],
                    "Volume": data["volume"],
                }).set_index("date").sort_index()
                return df
        except Exception:
            logger.warning("Futu kline fetch failed for %s", symbol)
        return pd.DataFrame()
