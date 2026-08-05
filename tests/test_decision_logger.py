"""Tests for live.decision_logger + data.cache decision persistence."""

from __future__ import annotations

import os
import tempfile

import pytest

from data.cache import StateStore
from live.decision_logger import (
    DECISION_KILL_SWITCH,
    DECISION_TRADE_BUY,
    DECISION_TRADE_SELL,
    DecisionLogger,
)


@pytest.fixture
def store():
    """Fresh StateStore on a temp DB."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = StateStore(db_path=path)
    s.init_schema()
    yield s
    s.close()
    # init_schema enables WAL — clean up -wal/-shm sidecars too (Windows
    # leaves them behind after the main db is unlinked).
    for suffix in ("", "-wal", "-shm"):
        try:
            os.unlink(path + suffix)
        except OSError:
            pass


class TestRecordDecision:
    def test_inserts_row(self, store):
        rid = store.record_decision(
            decision_type="trade_buy",
            symbol="NVDA",
            reason="signal",
        )
        assert rid > 0

    def test_round_trip_basic_fields(self, store):
        store.record_decision(
            decision_type="trade_buy",
            symbol="NVDA",
            reason="weekly_macd long",
        )
        rows = store.query_decisions(days=30)
        assert len(rows) == 1
        r = rows[0]
        assert r["decision_type"] == "trade_buy"
        assert r["symbol"] == "NVDA"
        assert r["reason"] == "weekly_macd long"

    def test_persists_risk_context_columns(self, store):
        store.record_decision(
            decision_type="trade_buy",
            symbol="NVDA",
            context={
                "risk_light": "yellow",
                "vix": 22.4,
                "portfolio_value": 100_000,
                "num_positions": 6,
                "concentration_hhi": 0.18,
                "effective_bets": 4.2,
                "var_95": 0.023,
                "drawdown_pct": -3.5,
            },
        )
        r = store.query_decisions(days=30)[0]
        assert r["risk_light"] == "yellow"
        assert r["vix"] == 22.4
        assert r["portfolio_value"] == 100_000
        assert r["num_positions"] == 6
        assert r["concentration_hhi"] == 0.18
        assert r["effective_bets"] == 4.2
        assert r["var_95"] == 0.023
        assert r["drawdown_pct"] == -3.5

    def test_payload_round_trip(self, store):
        payload = {"order_id": "X123", "qty": 100, "price": 145.6}
        store.record_decision(
            decision_type="trade_buy",
            symbol="NVDA",
            payload=payload,
        )
        r = store.query_decisions(days=30)[0]
        assert r["payload"] == payload

    def test_unknown_context_keys_go_to_payload(self, store):
        store.record_decision(
            decision_type="trade_buy",
            symbol="NVDA",
            context={"risk_light": "green", "custom_metric": 42},
        )
        r = store.query_decisions(days=30)[0]
        assert r["risk_light"] == "green"
        assert r["payload"]["_extra_context"]["custom_metric"] == 42


class TestQueryFilters:
    def _seed(self, store):
        store.record_decision("trade_buy", "NVDA", context={"risk_light": "green"})
        store.record_decision("trade_buy", "TSLA", context={"risk_light": "red"})
        store.record_decision("trade_sell", "NVDA", context={"risk_light": "yellow"})
        store.record_decision("kill_switch", None, reason="manual",
                              context={"risk_light": "red"})

    def test_filter_by_decision_type(self, store):
        self._seed(store)
        buys = store.query_decisions(decision_type="trade_buy")
        assert len(buys) == 2
        assert all(r["decision_type"] == "trade_buy" for r in buys)

    def test_filter_by_symbol(self, store):
        self._seed(store)
        nvda = store.query_decisions(symbol="NVDA")
        assert len(nvda) == 2
        assert all(r["symbol"] == "NVDA" for r in nvda)

    def test_filter_by_risk_light(self, store):
        self._seed(store)
        reds = store.query_decisions(risk_light="red")
        assert len(reds) == 2
        assert all(r["risk_light"] == "red" for r in reds)

    def test_filters_compose(self, store):
        self._seed(store)
        result = store.query_decisions(
            decision_type="trade_buy", risk_light="red",
        )
        assert len(result) == 1
        assert result[0]["symbol"] == "TSLA"

    def test_newest_first_order(self, store):
        import time
        store.record_decision("trade_buy", "A")
        time.sleep(1.05)  # ts is iso-seconds; needs >=1s gap to differ
        store.record_decision("trade_buy", "B")
        rows = store.query_decisions()
        assert rows[0]["symbol"] == "B"
        assert rows[1]["symbol"] == "A"


class TestDecisionLogger:
    def test_log_persists_via_cache(self, store):
        dl = DecisionLogger(store)
        rid = dl.log(DECISION_TRADE_BUY, symbol="NVDA",
                     reason="signal", context={"risk_light": "green"})
        assert rid > 0
        r = store.query_decisions()[0]
        assert r["decision_type"] == DECISION_TRADE_BUY
        assert r["symbol"] == "NVDA"
        assert r["risk_light"] == "green"

    def test_resolver_provides_context_lazily(self, store):
        """Resolver called at log time, not construction time."""
        state = {"light": "green"}

        def resolver():
            return {"risk_light": state["light"]}

        dl = DecisionLogger(store, context_resolver=resolver)
        dl.log(DECISION_TRADE_BUY, symbol="A")
        # change underlying state — next log should reflect it
        state["light"] = "red"
        dl.log(DECISION_TRADE_SELL, symbol="A")

        rows = store.query_decisions()
        # Newest first
        assert rows[0]["risk_light"] == "red"
        assert rows[1]["risk_light"] == "green"

    def test_resolver_failure_does_not_break_logging(self, store):
        def bad_resolver():
            raise RuntimeError("data feed dead")

        dl = DecisionLogger(store, context_resolver=bad_resolver)
        rid = dl.log(DECISION_KILL_SWITCH, reason="manual")
        # Decision is still persisted, just without context
        assert rid > 0
        r = store.query_decisions()[0]
        assert r["decision_type"] == DECISION_KILL_SWITCH
        assert r["risk_light"] is None

    def test_explicit_context_overrides_resolver(self, store):
        def resolver():
            return {"risk_light": "green"}

        dl = DecisionLogger(store, context_resolver=resolver)
        dl.log(DECISION_TRADE_BUY, symbol="A",
               context={"risk_light": "red"})  # explicit wins
        r = store.query_decisions()[0]
        assert r["risk_light"] == "red"

    def test_log_failure_is_swallowed(self, store):
        """If cache raises, log returns 0 — never propagates."""

        class _Broken:
            def record_decision(self, **_kw):
                raise RuntimeError("disk full")

        dl = DecisionLogger(_Broken())
        # Should NOT raise
        rid = dl.log(DECISION_TRADE_BUY, symbol="X")
        assert rid == 0
