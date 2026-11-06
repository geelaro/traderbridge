"""RiskController — risk checks, position sizing, circuit breaker, state persistence."""

import json
from datetime import date
from typing import Dict

from broker import Order, OrderSide
from utils import get_logger
from utils.sizing import calc_risk_budget_qty

logger = get_logger("live")


class RiskController:
    """Encapsulates all risk-related logic previously embedded in LiveTrader.

    Parameters
    ----------
    risk : RiskLimits
        Risk parameter dataclass instance.
    cache : CacheManager
        SQLite cache manager for persistence.
    broker : Broker
        Broker adapter (for last_prices in slippage checks).
    notifier : Notifier
        Alert/notification sender.
    """

    def __init__(self, risk, cache, broker, notifier):
        self.risk = risk
        self.cache = cache
        self.broker = broker
        self.notifier = notifier
        self.trading_paused = False
        self.pause_reason = ""
        self.alert_sent = False
        self.entry_prices: Dict[str, float] = {}
        self.restore_state()

    def init_risk(self, account):
        today = date.today().isoformat()
        if self.risk._date != today or self.risk._day_start_equity <= 0:
            self.risk._day_start_equity = account.total_equity
            self.risk._date = today
            self.persist_state()
        if account.total_equity > self.risk._peak_equity:
            self.risk._peak_equity = account.total_equity
            self.persist_state()

    def restore_state(self):
        """Restore persisted risk state.

        - ``peak_equity`` and ``consecutive_losses`` are time-invariant — they
          represent historical state and MUST be restored across days, otherwise
          the max_total_drawdown_pct circuit breaker silently disarms each
          morning (peak resets to today's open equity) and the consecutive-
          loss counter resets even though run_daemon's day-rollover only
          intends to reset daily counters.
        - ``day_start_equity`` and ``daily_trade_count`` are day-scoped and
          only restored when the stored date matches today.
        """
        stored_date = self.cache.load_risk_state("date")
        today = date.today().isoformat()

        # Always restore historical state
        cl = self.cache.load_risk_state("consecutive_losses")
        pe = self.cache.load_risk_state("peak_equity")
        self.risk._consecutive_losses = int(cl) if cl else 0
        self.risk._peak_equity = float(pe) if pe else 0.0

        # Day-scoped state only valid for today
        if stored_date == today:
            dt = self.cache.load_risk_state("daily_trade_count")
            dse = self.cache.load_risk_state("day_start_equity")
            self.risk._date = today
            self.risk._daily_trade_count = int(dt) if dt else 0
            self.risk._day_start_equity = float(dse) if dse else 0.0
            # The dashboard reads these keys to show the live pause state.
            # Keep them day-scoped so an old daily pause cannot leak into a
            # new trading day before the first daemon tick.
            self.trading_paused = self.cache.load_risk_state("trading_paused") == "True"
            self.pause_reason = self.cache.load_risk_state("pause_reason") or ""
        else:
            self.trading_paused = False
            self.pause_reason = ""

        self.entry_prices = {
            sym: price for sym, (price, _) in self.cache.load_all_entry_prices().items()
        }

    def persist_state(self):
        self.cache.save_risk_state("date", self.risk._date)
        self.cache.save_risk_state("day_start_equity", str(self.risk._day_start_equity))
        self.cache.save_risk_state("peak_equity", str(self.risk._peak_equity))
        self.cache.save_risk_state("consecutive_losses", str(self.risk._consecutive_losses))
        self.cache.save_risk_state("daily_trade_count", str(self.risk._daily_trade_count))
        self._persist_pause_state()

    def _persist_pause_state(self):
        """Persist the operational pause state used by dashboard readers."""
        self.cache.save_risk_state("trading_paused", str(bool(self.trading_paused)))
        self.cache.save_risk_state("pause_reason", self.pause_reason)

    def risk_snapshot(self, account=None, positions: dict | None = None) -> dict:
        """Return the current operational risk state for monitoring surfaces.

        This is deliberately separate from the analytical portfolio snapshot:
        it describes live guardrails and their current utilisation, so a
        dashboard can distinguish "portfolio VaR is high" from "the daemon is
        currently refusing new orders".
        """
        equity = float(getattr(account, "total_equity", 0.0) or 0.0)
        day_start = float(self.risk._day_start_equity or 0.0)
        peak = float(self.risk._peak_equity or 0.0)
        exposure = sum(
            float(getattr(position, "market_value", 0.0) or 0.0)
            for position in (positions or {}).values()
            if float(getattr(position, "market_value", 0.0) or 0.0) > 0
        )
        return {
            "as_of": date.today().isoformat(),
            "equity": round(equity, 2),
            "day_start_equity": round(day_start, 2),
            "peak_equity": round(peak, 2),
            "daily_pnl_pct": round((equity / day_start - 1) * 100, 2) if day_start > 0 else None,
            "drawdown_pct": round((equity / peak - 1) * 100, 2) if peak > 0 else None,
            "exposure_pct": round(exposure / equity * 100, 2) if equity > 0 else 0.0,
            "daily_trade_count": int(self.risk._daily_trade_count),
            "consecutive_losses": int(self.risk._consecutive_losses),
            "limits": {
                "max_daily_loss_pct": round(self.risk.max_daily_loss_pct * 100, 2),
                "max_total_drawdown_pct": round(self.risk.max_total_drawdown_pct * 100, 2),
                "max_total_exposure_pct": round(self.risk.max_total_exposure_pct * 100, 2),
                "max_daily_trades": int(self.risk.max_daily_trades),
                "max_consecutive_losses": int(self.risk.max_consecutive_losses),
            },
            "paused": bool(self.trading_paused),
            "pause_reason": self.pause_reason,
        }

    def check_global(self, account, positions: dict) -> bool:
        r = self.risk
        equity = account.total_equity

        today = date.today().isoformat()
        if r._date != today:
            self.trading_paused = False
            self.pause_reason = ""
            self.alert_sent = False

        paused = False
        reason = ""
        if r._day_start_equity > 0 and equity < r._day_start_equity * (1 - r.max_daily_loss_pct):
            loss_pct = (r._day_start_equity - equity) / r._day_start_equity * 100
            paused = True
            reason = f"日内亏损超限 ({loss_pct:.1f}% > {r.max_daily_loss_pct*100:.0f}%)"
        elif r._peak_equity > 0 and equity < r._peak_equity * (1 - r.max_total_drawdown_pct):
            dd_pct = (r._peak_equity - equity) / r._peak_equity * 100
            paused = True
            reason = f"历史峰值回撤超限 ({dd_pct:.1f}% > {r.max_total_drawdown_pct*100:.0f}%)"
        elif r._consecutive_losses >= r.max_consecutive_losses:
            paused = True
            reason = f"连续亏损熔断 ({r._consecutive_losses}/{r.max_consecutive_losses})"
        elif positions:
            total_exposure = sum(p.market_value for p in positions.values() if p.market_value > 0)
            exposure_pct = total_exposure / equity if equity > 0 else 0
            if exposure_pct > r.max_total_exposure_pct:
                paused = True
                reason = f"总敞口超限 ({exposure_pct*100:.1f}% > {r.max_total_exposure_pct*100:.0f}%)"

        # ── Apply pause state ──────────────────────────────────────────
        # Use local paused flag so that when conditions improve (e.g. a
        # sell drops exposure below the cap) the pause is *cleared* within
        # the same day.  Before this change trading_paused was a one-way
        # latch reset only at midnight.
        self.trading_paused = paused
        self.pause_reason = reason
        self._persist_pause_state()
        self.cache.save_risk_state(
            "risk_operational_snapshot",
            json.dumps(self.risk_snapshot(account, positions), ensure_ascii=False),
        )

        if self.trading_paused:
            print(f"\n  !! 交易暂停: {self.pause_reason}")
            if not self.alert_sent:
                self.notifier.error("交易暂停", self.pause_reason)
                self.alert_sent = True
            self.cache.log_ops("trading_paused", detail=self.pause_reason)
        else:
            self.alert_sent = False

        return not self.trading_paused

    def passes_risk(self, signal: dict, qty: int, account) -> bool:
        r = self.risk
        equity = account.total_equity

        if r._consecutive_losses >= r.max_consecutive_losses:
            print(f"  ! 连续亏损熔断 ({r._consecutive_losses}笔)，暂停交易")
            self.cache.log_ops("risk_reject", symbol=signal.get("symbol", ""),
                               detail="consecutive_losses", level="WARN")
            return False

        if r._daily_trade_count >= r.max_daily_trades:
            print(f"  ! 日内交易次数已达上限 ({r.max_daily_trades}笔)，暂停交易")
            self.cache.log_ops("risk_reject", symbol=signal.get("symbol", ""),
                               detail="daily_trade_cap", level="WARN")
            return False

        if equity < r._day_start_equity * (1 - r.max_daily_loss_pct):
            print(f"  ! 日内亏损超限 ({r.max_daily_loss_pct*100:.0f}%)，暂停交易")
            self.cache.log_ops("risk_reject", symbol=signal.get("symbol", ""),
                               detail="daily_loss", level="WARN")
            return False

        order_value = signal.get("price", 0) * qty
        if order_value < r.min_order_value:
            self.cache.log_ops("risk_reject", symbol=signal.get("symbol", ""),
                               detail="min_order_value", level="WARN")
            return False

        sym = signal.get("symbol", "")
        signal_price = signal.get("price", 0)
        if sym in self.broker.last_prices:
            last_price = self.broker.last_prices[sym]
            if last_price > 0 and signal_price > 0:
                slippage = abs(signal_price - last_price) / last_price
                if slippage > r.max_slippage_pct:
                    print(f"  ! {sym} 滑点超限 ({slippage*100:.2f}%)，拒绝")
                    self.cache.log_ops("slippage_rejected", symbol=sym,
                                       detail=f"{slippage*100:.2f}%", value=slippage*100)
                    return False

        return True

    def calc_position_size(self, capital: float, price: float, atr: float,
                            last_price: float, total_equity: float = 0) -> int:
        """Risk-budget position sizing.

        *capital* should be ``available_cash`` (can actually be spent).
        *total_equity* (optional) caps the position as a fraction of
        total portfolio value — prevents over-concentration.
        """
        if price <= 0:
            return 0

        r = self.risk
        raw_qty = calc_risk_budget_qty(
            capital, price, atr,
            risk_pct=r.base_risk_pct,
            stop_atr_mult=2.0,
            vol_sensitivity=r.vol_sensitivity,
            min_vol_scalar=r.min_vol_scalar,
        )

        ref = total_equity if total_equity > 0 else capital
        max_qty = int(ref * r.max_position_pct / price)
        return max(1, min(raw_qty, max_qty))

    def on_trade_filled(self, order: Order):
        r = self.risk
        sym = order.symbol
        fill_price = order.avg_fill_price
        r._daily_trade_count += 1

        if order.side == OrderSide.BUY:
            self.entry_prices[sym] = fill_price
            self.cache.save_entry_price(sym, fill_price, date.today().isoformat())
        elif order.side == OrderSide.SELL:
            entry = self.entry_prices.pop(sym, None)
            if entry is not None:
                self.cache.delete_entry_price(sym)
                self.cache.save_trade_pnl(
                    sym, "SELL", order.filled_qty, entry, fill_price,
                    date.today().isoformat(), order.order_id,
                )
                if fill_price < entry:
                    r._consecutive_losses += 1
                    logger.warning("连续亏损 %d/%d: %s  PnL=$%.2f",
                                   r._consecutive_losses, r.max_consecutive_losses,
                                   sym, (fill_price - entry) * order.filled_qty)
                else:
                    r._consecutive_losses = 0
                    logger.info("交易PnL: %s  $%.2f", sym, (fill_price - entry) * order.filled_qty)
        self.persist_state()

    def update_after_trade(self):
        self.persist_state()
