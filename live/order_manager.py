"""OrderManager — order generation, submission, logging, slippage tracking."""

from typing import Dict, List

from broker import Order, OrderSide, OrderStatus, OrderType


class OrderManager:
    """Encapsulates all order-related logic previously embedded in LiveTrader.

    Parameters
    ----------
    broker : Broker
        Broker adapter for order submission and querying.
    cache : CacheManager
        SQLite cache manager for audit trail.
    execution_model : ExecutionModel
        Order execution config (timing, style, timeouts).
    notifier : Notifier
        Alert/notification sender.
    gate : SignalGate
        Signal gating for regime/risk filtering.
    risk_ctrl : RiskController
        Risk controller for position sizing and risk checks.
    dry_run : bool
        If True, orders are simulated rather than submitted.
    """

    def __init__(self, broker, cache, execution_model, notifier, gate, risk_ctrl, dry_run=False):
        self.broker = broker
        self.cache = cache
        self.execution_model = execution_model
        self.notifier = notifier
        self.gate = gate
        self.risk_ctrl = risk_ctrl
        self.dry_run = dry_run

    def generate_orders(
        self, signals: List[dict], positions: Dict[str, any], account
    ) -> List[Order]:
        """Generate long entries and position closes; short entries are research-only.

        Four-way signal/position matrix:

        ======= ============ ==========================
        Signal   Position      Action
        ======= ============ ==========================
         1       flat          BUY → open long
         1       short (<0)    BUY → cover short
        -1       flat          blocked (short research only)
        -1       long (>0)     SELL → close long
        ======= ============ ==========================

        Close orders (SELL for long exit / BUY for short cover) are
        generated first. Long openings are then allocated from cash.
        """
        orders = []

        symbol_signals: Dict[str, dict] = {}
        for s in signals:
            sym = s["symbol"]
            if sym not in symbol_signals:
                symbol_signals[sym] = s
            if s["signal"] != 0 and symbol_signals[sym]["signal"] == 0:
                symbol_signals[sym] = s

        # ---- Phase 1: close orders (do these first) -------------------------
        for sym, sig in symbol_signals.items():
            pos = positions.get(sym)
            if pos is None:
                continue
            qty = abs(pos.quantity) if hasattr(pos, 'quantity') else 0
            if qty <= 0:
                continue

            is_long = getattr(pos, 'quantity', 0) > 0
            is_short = getattr(pos, 'quantity', 0) < 0

            # Signal=-1 and long position → SELL to close
            if sig["signal"] == -1 and is_long:
                ok, reason = self.gate.allow_sell(sig)
                if not ok:
                    print(f"  ! {sym} {reason}，跳过")
                    self.cache.log_ops("gate_reject", symbol=sym, detail=reason, level="WARN")
                    continue
                plan = self.execution_model.make_plan(
                    symbol=sym, side=OrderSide.SELL,
                    quantity=qty, created_index=0,
                    reason=sig.get("strategy", "signal"),
                )
                orders.append(self.execution_model.to_broker_order(plan))

            # Signal=1 and short position → BUY to cover
            elif sig["signal"] == 1 and is_short:
                plan = self.execution_model.make_plan(
                    symbol=sym, side=OrderSide.BUY,
                    quantity=qty, created_index=0,
                    reason=sig.get("strategy", "cover"),
                )
                orders.append(self.execution_model.to_broker_order(plan))

        # ---- Phase 2: collect long entry candidates ----------------------------
        candidates_long = []
        for sym, sig in symbol_signals.items():
            if sig.get("orphan"):
                continue
            pos = positions.get(sym)
            has_position = pos is not None and abs(getattr(pos, 'quantity', 0)) > 0

            if sig["signal"] == 1 and not has_position:
                ok, reason = self.gate.allow_buy(sig, positions, account)
                if not ok:
                    print(f"  ! {sym} {reason}，跳过")
                    self.cache.log_ops("gate_reject", symbol=sym, detail=reason, level="WARN")
                    continue
                candidates_long.append((sym, sig))

            elif sig["signal"] == -1 and not has_position:
                # Short research signals must never open a broker position.
                # In particular, a long-only exit signal is not a short entry.
                self.cache.log_ops(
                    "gate_reject", symbol=sym,
                    detail="SHORT_ENTRY_RESEARCH_ONLY", level="INFO",
                )
                continue

        # ---- Phase 3: batch equal-risk allocation for all open orders --------
        total_candidates = len(candidates_long)
        if total_candidates > 0:
            avail = getattr(account, "available_cash", account.total_equity)
            capital_per = avail / total_candidates

            for sym, sig in candidates_long:
                sig["_qty"] = self.risk_ctrl.calc_position_size(
                    capital=capital_per, price=sig.get("price", 0),
                    atr=sig.get("atr", 0), last_price=sig.get("price", 0),
                    total_equity=account.total_equity,
                )
                if sig["_qty"] <= 0:
                    print(f"  ! {sym} 买入信号仓位为0，跳过")
                    continue
                qty = self.gate.vol_scaled_qty(sig["_qty"])
                if self.risk_ctrl.passes_risk(sig, qty, account):
                    plan = self.execution_model.make_plan(
                        symbol=sym, side=OrderSide.BUY,
                        quantity=qty, created_index=0,
                    )
                    orders.append(self.execution_model.to_broker_order(plan))
                else:
                    print(f"  ! {sym} 风控未通过，跳过")

        return orders

    def submit_and_wait(self, order: Order) -> Order:
        import time as _time

        result = self.broker.submit_order(order)
        if result.status in (OrderStatus.FILLED, OrderStatus.REJECTED, OrderStatus.CANCELLED):
            return result

        timeout_s = self.execution_model.timeout_seconds(order)
        poll_interval = 3
        elapsed = 0

        while elapsed < timeout_s:
            _time.sleep(poll_interval)
            elapsed += poll_interval
            updated = self.broker.get_order(result.order_id)
            if updated is None:
                continue
            result = updated
            if result.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED):
                return result

        from utils import get_logger
        logger = get_logger("live")
        if result.status not in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED):
            logger.warning("订单超时未成交，撤单: %s %s (%s)",
                           order.symbol, order.order_id, result.status.value)
            self.broker.cancel_order(result.order_id)
            result.status = OrderStatus.CANCELLED

        return result

    def record_slippage(self, order: Order, signal_price: float):
        if signal_price <= 0 or order.avg_fill_price <= 0:
            return
        slippage = (order.avg_fill_price - signal_price) / signal_price
        self.cache.log_ops("slippage", symbol=order.symbol,
                           detail=f"{slippage*100:+.2f}%", value=slippage*100)
        self.cache.init_schema()
        self.cache.conn.execute(
            "INSERT INTO slippage_log VALUES (?,?,?,?,?,?,?)",
            [order.order_id, order.symbol, order.side.value,
             signal_price, order.avg_fill_price, round(slippage * 100, 4),
             order.created_at],
        )
        self.cache._commit()

    def log_order(self, order: Order):
        self.cache.init_schema()
        self.cache.conn.execute(
            "INSERT INTO order_log VALUES (?,?,?,?,?,?,?)",
            [order.order_id, order.symbol, order.side.value,
             order.filled_qty, order.avg_fill_price,
             order.status.value, order.created_at],
        )
        self.cache._commit()

    @staticmethod
    def print_order(order: Order):
        print(f"  [DRY-RUN] {order.side.value} {order.symbol} "
              f"{order.quantity}股 "
              f"{'MARKET' if order.order_type == OrderType.MARKET else f'@ ${order.price:.2f}'}")
