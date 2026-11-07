"""Broker middleware — retry / circuit breaker wrappers.

Decorates a real :class:`Broker` to add cross-cutting resilience
behaviours without touching the underlying adapter.  The middleware
honours the same :class:`Broker` interface so callers can swap in /
out transparently.

When to use
-----------
- Production live trading with FutuOpenD — wrap to survive transient
  network blips, OpenD restarts, brief rate-limit windows.
- Anywhere broker calls cross a network boundary.

Not for
-------
- MockBroker (it's in-process, no flakiness to retry).
- Backtests (deterministic, retries would silently change behaviour).

Example
-------
    real = FutuBroker(...)
    broker = RetryingBroker(real, max_retries=3, base_delay=0.5)
    # use `broker` exactly like a FutuBroker

Design
------
- Wraps network-touching broker methods.  Order submission is never
  retried automatically because a timeout can leave its outcome ambiguous.
- Retries on requests.RequestException-like / TimeoutError /
  ConnectionError / OSError.
- Exponential backoff: ``base_delay * 2**attempt``, capped at ``max_delay``.
- Circuit breaker: after ``cb_threshold`` consecutive failures, opens for
  ``cb_cooldown`` seconds — fails fast instead of waiting on each call.
- All retries / opens emit metrics (broker_retry_total, broker_circuit_open)
  via utils.metrics_server.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import List, Optional

from broker.base import Account, Order, Position
# NOTE: RetryingBroker doesn't subclass Broker (avoids inheriting the
# parent's last_prices initialiser).  It's duck-typed and provides the
# same public interface via composition.

logger = logging.getLogger(__name__)


_RETRYABLE = (
    ConnectionError,
    TimeoutError,
    OSError,
)


def _maybe_metric_incr(name: str, labels: Optional[dict] = None) -> None:
    """Best-effort metrics call — never raises into the broker path."""
    try:
        from utils import metrics_server
        metrics_server.incr(name, labels)
    except Exception:
        pass


class CircuitOpenError(RuntimeError):
    """Raised when the circuit breaker is open and the call is rejected."""


class RetryingBroker:
    """Wrap any Broker with retry + exponential backoff + circuit breaker.

    Parameters
    ----------
    inner : Broker
        The real broker to delegate to.
    max_retries : int
        Number of retry attempts per call (in addition to the first call).
    base_delay : float
        Initial backoff seconds; doubles each retry.
    max_delay : float
        Cap on exponential backoff.
    cb_threshold : int
        Consecutive failures before opening the circuit breaker.
    cb_cooldown : float
        How many seconds the circuit stays open before retrying.
    """

    def __init__(
        self,
        inner,  # duck-typed Broker
        max_retries: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 8.0,
        cb_threshold: int = 5,
        cb_cooldown: float = 60.0,
    ):
        self._inner = inner
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._cb_threshold = cb_threshold
        self._cb_cooldown = cb_cooldown
        self._consecutive_failures = 0
        self._circuit_opened_at: Optional[float] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Broker interface — delegate with retry
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"retrying:{self._inner.name}"

    @property
    def last_prices(self):
        return self._inner.last_prices

    def get_account(self) -> Account:
        return self._call("get_account", self._inner.get_account)

    def get_positions(self) -> List[Position]:
        return self._call("get_positions", self._inner.get_positions)

    def submit_order(self, order: Order) -> Order:
        # A timeout may happen after the broker accepted the order.  Retrying
        # could place a duplicate, so only replay read/idempotent operations.
        return self._call(
            "submit_order", self._inner.submit_order, order, retry=False,
        )

    def cancel_order(self, order_id: str) -> bool:
        return self._call("cancel_order", self._inner.cancel_order, order_id)

    def get_order(self, order_id: str) -> Optional[Order]:
        return self._call("get_order", self._inner.get_order, order_id)

    def warmup(self, symbols: List[str]) -> bool:
        return self._call("warmup", self._inner.warmup, symbols)

    def connect(self):
        return self._inner.connect()

    def disconnect(self):
        return self._inner.disconnect()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_circuit(self) -> None:
        """Raise CircuitOpenError if circuit is currently open."""
        with self._lock:
            if self._circuit_opened_at is None:
                return
            elapsed = time.time() - self._circuit_opened_at
            if elapsed < self._cb_cooldown:
                _maybe_metric_incr("broker_circuit_rejected_total",
                                   {"broker": self._inner.name})
                raise CircuitOpenError(
                    f"Broker '{self._inner.name}' circuit open for "
                    f"{self._cb_cooldown - elapsed:.1f}s more"
                )
            # Cooldown elapsed — half-open: allow one trial call.
            self._circuit_opened_at = None
            self._consecutive_failures = 0

    def _record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0

    def _record_failure(self, op: str, exc: Exception) -> None:
        with self._lock:
            self._consecutive_failures += 1
            if (self._circuit_opened_at is None
                    and self._consecutive_failures >= self._cb_threshold):
                self._circuit_opened_at = time.time()
                logger.warning(
                    "Broker '%s' circuit OPENED after %d consecutive failures",
                    self._inner.name, self._consecutive_failures,
                )
                _maybe_metric_incr("broker_circuit_open_total",
                                   {"broker": self._inner.name})

    def _call(self, op: str, fn, *args, retry: bool = True, **kwargs):
        """Run ``fn(*args)`` with retry + circuit breaker semantics."""
        self._check_circuit()
        last_exc: Optional[Exception] = None
        max_attempts = self._max_retries + 1 if retry else 1
        for attempt in range(max_attempts):
            try:
                result = fn(*args, **kwargs)
                self._record_success()
                return result
            except _RETRYABLE as exc:
                last_exc = exc
                if not retry:
                    logger.warning(
                        "Broker '%s' %s failed; not retrying because the "
                        "operation outcome may be ambiguous: %s",
                        self._inner.name, op, exc,
                    )
                    break
                _maybe_metric_incr(
                    "broker_retry_total",
                    {"broker": self._inner.name, "op": op,
                     "attempt": str(attempt + 1)},
                )
                if attempt + 1 >= max_attempts:
                    break
                delay = min(self._base_delay * (2 ** attempt), self._max_delay)
                logger.debug(
                    "Broker '%s' %s attempt %d failed: %s — retrying in %.1fs",
                    self._inner.name, op, attempt + 1, exc, delay,
                )
                time.sleep(delay)
            except Exception:
                # Non-retryable: re-raise immediately.  Don't count as
                # circuit-breaker failure either — likely a logic bug
                # not a transient outage.
                raise
        # Out of retries — record failure and re-raise the last network error
        self._record_failure(op, last_exc)
        raise last_exc  # type: ignore[misc]
