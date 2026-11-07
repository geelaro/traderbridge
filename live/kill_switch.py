"""Emergency portfolio liquidation — manual one-button kill switch.

Manual-only by design.  Empirical study (CBOE VIX history 1990+, see
project journal 2026-06-02) shows VIX > 50 historically marks
*bottoms*, not continuation lower — auto-liquidating at VIX > 50 would
have locked in losses right before SPY rallied ~45% over the next year.
So no automatic threshold triggers; the user clicks the button when
they decide.

On trigger
----------
1. Snapshot all current positions via broker.get_positions().
2. For each non-zero position, submit an opposing MARKET order
   (SELL for longs, BUY for shorts).
3. Set ``risk_ctrl.trading_paused = True`` so the daemon doesn't open
   new positions until the user explicitly resets.
4. Persist active flag + reason + timestamp via StateStore.
5. Append to alert_history (audit trail).
6. Push Feishu RED card via notifier.

Reset
-----
``reset()`` clears the active flag and unpauses trading.  Idempotent —
re-triggering while active is a no-op.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from broker import Broker
    from live.risk_controller import RiskController
    from data.cache import CacheManager
    from utils.notify import Notifier

logger = logging.getLogger(__name__)


_KEY_ACTIVE = "kill_switch:active"
_KEY_REASON = "kill_switch:reason"
_KEY_TRIGGERED_AT = "kill_switch:triggered_at"

# Composite-state JSON key for atomic write of {active, reason, triggered_at}.
# Three separate keys exist for backward-compat reads — new code writes both
# the JSON blob AND the legacy individual keys so old readers still see them.
_KEY_STATE = "kill_switch:state"


class KillSwitch:
    """Manual emergency liquidation.

    Construction
    ------------
    All four dependencies are injected so the class is testable without
    spinning up a real LiveTrader.
    """

    def __init__(
        self,
        broker: "Broker",
        risk_ctrl: "RiskController",
        notifier: "Notifier",
        cache: "CacheManager",
    ):
        self.broker = broker
        self.risk_ctrl = risk_ctrl
        self.notifier = notifier
        self.cache = cache

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        state = self._load_state()
        return bool(state.get("active"))

    def get_state(self) -> dict:
        """Snapshot of current kill switch state.

        Reads from the atomic ``_KEY_STATE`` JSON blob; falls back to the
        legacy three-key layout for installations that triggered before
        the atomic-state migration.
        """
        return self._load_state()

    def _load_state(self) -> dict:
        """Resolve {active, reason, triggered_at} preferring atomic JSON."""
        blob = self.cache.load_risk_state(_KEY_STATE)
        if blob:
            try:
                data = json.loads(blob)
                return {
                    "active": bool(data.get("active")),
                    "reason": str(data.get("reason") or ""),
                    "triggered_at": str(data.get("triggered_at") or ""),
                }
            except (ValueError, TypeError):
                logger.warning("KillSwitch: malformed state blob, "
                               "falling back to legacy keys")
        return {
            "active": self.cache.load_risk_state(_KEY_ACTIVE) == "1",
            "reason": self.cache.load_risk_state(_KEY_REASON) or "",
            "triggered_at": self.cache.load_risk_state(_KEY_TRIGGERED_AT) or "",
        }

    # ------------------------------------------------------------------
    # Trigger
    # ------------------------------------------------------------------

    def trigger(self, reason: str, dry_run: bool = False) -> dict:
        """Liquidate all open positions immediately.

        Parameters
        ----------
        reason : str
            Free-text reason (audit + notification body).  Required.
        dry_run : bool
            If True, returns intended orders without submitting them or
            changing trading state, audit history, metrics, or
            notifications.

        Returns
        -------
        dict::

            {
                "status": "triggered" | "preview" | "already_active" |
                           "no_positions",
                "reason": str,
                "n_positions": int,
                "orders": list[dict],   # one per attempted sell
                "errors": list[dict],   # per-symbol errors
                "dry_run": bool,
            }

        Idempotent: re-triggering while already active is a no-op and
        returns ``status="already_active"`` (no fresh orders submitted).
        """
        if not reason or not reason.strip():
            raise ValueError("Kill Switch requires a non-empty reason")

        if self.is_active:
            logger.warning("KillSwitch already active, skipping re-trigger")
            return {
                "status": "already_active",
                "reason": reason,
                "n_positions": 0,
                "orders": [],
                "errors": [],
                "dry_run": dry_run,
            }

        # Import here to avoid circular import at module load
        from broker import Order, OrderSide, OrderType

        orders: list = []
        errors: list = []

        # Reading positions must NEVER throw out of trigger(). A live trigger
        # still proceeds to pause and notify if the broker is unavailable.
        try:
            positions = list(self.broker.get_positions() or [])
        except Exception as exc:
            if dry_run:
                logger.warning(
                    "KillSwitch dry-run: broker.get_positions() failed",
                    exc_info=True,
                )
            else:
                logger.exception("KillSwitch: broker.get_positions() failed")
            errors.append({
                "symbol": "*",
                "error": f"get_positions failed: {type(exc).__name__}: {exc}",
            })
            positions = []

        for pos in positions:
            qty = int(getattr(pos, "quantity", 0) or 0)
            if qty == 0:
                continue
            side = OrderSide.SELL if qty > 0 else OrderSide.BUY
            sell_qty = abs(qty)
            try:
                order = Order(
                    symbol=pos.symbol,
                    side=side,
                    order_type=OrderType.MARKET,
                    quantity=sell_qty,
                )
                if dry_run:
                    orders.append({
                        "symbol": pos.symbol, "side": side.value,
                        "qty": sell_qty, "status": "DRY_RUN",
                        "order_id": "",
                    })
                else:
                    result = self.broker.submit_order(order)
                    orders.append({
                        "symbol": pos.symbol,
                        "side": result.side.value,
                        "qty": result.quantity,
                        "status": result.status.value,
                        "order_id": result.order_id,
                    })
            except Exception as exc:
                if dry_run:
                    logger.warning(
                        "KillSwitch dry-run: failed to prepare %s", pos.symbol,
                        exc_info=True,
                    )
                else:
                    # A live liquidation failure warrants Feishu attention
                    # via NotifyLogHandler. See AGENTS.md, rule 6.
                    logger.exception("KillSwitch: failed to liquidate %s", pos.symbol)
                errors.append({"symbol": pos.symbol, "error": str(exc)})

        ts = datetime.now().isoformat(timespec="seconds")
        payload = {
            "reason": reason,
            "n_positions": len(positions),
            "orders": orders,
            "errors": errors,
            "dry_run": dry_run,
            "triggered_at": ts,
        }
        if dry_run:
            status = "preview" if positions else "no_positions"
            return {"status": status, **payload}

        # Pause trading so the daemon doesn't open new positions
        try:
            self.risk_ctrl.trading_paused = True
            self.risk_ctrl.pause_reason = f"Kill Switch: {reason}"
            if hasattr(self.risk_ctrl, "persist_state"):
                self.risk_ctrl.persist_state()
        except Exception:
            logger.exception("KillSwitch: failed to pause risk_ctrl")

        # Persist active flag.  Atomic JSON write to _KEY_STATE so a crash
        # mid-save can't leave "active=1 with empty reason/timestamp".
        # Legacy individual keys are also written for backward-compat
        # readers; they're best-effort if the JSON write succeeded.
        state_blob = json.dumps({
            "active": True, "reason": reason, "triggered_at": ts,
        }, ensure_ascii=False)
        self.cache.save_risk_state(_KEY_STATE, state_blob)
        # Legacy mirror — failures here don't affect idempotency since
        # is_active reads from _KEY_STATE first.
        try:
            self.cache.save_risk_state(_KEY_ACTIVE, "1")
            self.cache.save_risk_state(_KEY_REASON, reason)
            self.cache.save_risk_state(_KEY_TRIGGERED_AT, ts)
        except Exception:
            logger.warning("KillSwitch: legacy state mirror failed", exc_info=True)

        # Audit trail
        try:
            self.cache.record_alert("kill_switch", payload)
        except Exception:
            logger.exception("KillSwitch: audit log write failed")

        # Also persist to the unified decision_history so the dashboard's
        # decision-review tab sees kill-switch events alongside trades.
        # Risk context: at trigger time, risk_light is by definition RED
        # in spirit (manual emergency) — we record what's known.
        try:
            risk_light = self.cache.load_risk_state("alert:last_risk_level")
            self.cache.record_decision(
                decision_type="kill_switch",
                symbol=None,
                reason=reason,
                context={
                    "risk_light": risk_light,
                    "num_positions": len(positions),
                },
                payload={
                    "n_orders": len(orders),
                    "n_errors": len(errors),
                    "dry_run": dry_run,
                },
                ts=ts,
            )
        except Exception:
            logger.exception("KillSwitch: decision log write failed")

        # Push Feishu card (best-effort)
        self._notify(reason, len(positions), orders, errors, dry_run)

        # Count live triggers for the ops dashboard.
        try:
            from utils import metrics_server
            metrics_server.incr(
                "kill_switch_triggered_total",
                {"dry_run": "true" if dry_run else "false"},
            )
        except Exception:
            pass

        status = "no_positions" if not positions else "triggered"
        return {"status": status, **payload}

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, reason: str = "Manual reset by user") -> None:
        """Clear the active flag and unpause trading.

        After triggering, the daemon stays paused until this is called.
        The user is expected to manually review positions, confirm safe
        state, then reset.
        """
        if not self.is_active:
            return
        # Atomic clear; mirror to legacy keys for backward-compat readers.
        cleared = json.dumps({"active": False, "reason": "",
                              "triggered_at": ""}, ensure_ascii=False)
        self.cache.save_risk_state(_KEY_STATE, cleared)
        try:
            self.cache.save_risk_state(_KEY_ACTIVE, "0")
        except Exception:
            logger.warning("KillSwitch: legacy reset mirror failed", exc_info=True)
        try:
            self.risk_ctrl.trading_paused = False
            self.risk_ctrl.pause_reason = ""
            if hasattr(self.risk_ctrl, "persist_state"):
                self.risk_ctrl.persist_state()
        except Exception:
            logger.exception("KillSwitch: failed to unpause risk_ctrl")
        try:
            self.cache.record_alert("kill_switch_reset", {
                "reason": reason,
                "reset_at": datetime.now().isoformat(timespec="seconds"),
            })
        except Exception:
            logger.exception("KillSwitch: reset audit log failed")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _notify(self, reason: str, n_positions: int, orders: list,
                errors: list, dry_run: bool) -> None:
        """Push a RED Feishu card describing the trigger."""
        if not getattr(self.notifier, "available", False):
            logger.warning("KillSwitch: notifier unavailable, alert dropped")
            return
        try:
            self.notifier.kill_switch_card(reason, n_positions, orders,
                                           errors, dry_run)
        except Exception:
            logger.exception("KillSwitch: failed to send Feishu card")
