"""Tests for dashboard/decision_summary.py — pure-compute paths only.

render_decision_summary is Streamlit rendering and is intentionally not
covered here (see module docstring: "不需要 Streamlit").
"""

import json
from datetime import date, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from analysis.risk_monitor import RiskLevel
from data.protocol import DataSource
from dashboard.decision_summary import (
    _rejected_signals,
    _risk_delta,
    _verdict,
    build_decision_summary,
)
from utils.market_state import MarketRegime, Volatility


# ===================================================================
# _verdict — 5 branches
# ===================================================================


class TestVerdict:
    def test_paused_wins_over_everything(self):
        code, text = _verdict(RiskLevel.RED, True, "daily_loss", [{"signal": 1}], [])
        assert code == "paused"
        assert "daily_loss" in text

    def test_red_risk_blocks_when_not_paused(self):
        code, text = _verdict(RiskLevel.RED, False, "", [{"signal": 1}], [])
        assert code == "red"

    def test_no_signals(self):
        code, text = _verdict(RiskLevel.GREEN, False, "", [], [])
        assert code == "no_signal"

    def test_all_signals_blocked(self):
        signals = [{"signal": 1}, {"signal": 1}]
        rejected = [{"symbol": "A", "strategy": "s", "reason": "X"},
                    {"symbol": "B", "strategy": "s", "reason": "X"}]
        code, text = _verdict(RiskLevel.GREEN, False, "", signals, rejected)
        assert code == "all_blocked"

    def test_actionable_when_some_pass(self):
        signals = [{"signal": 1}, {"signal": 1}]
        rejected = [{"symbol": "A", "strategy": "s", "reason": "X"}]
        code, text = _verdict(RiskLevel.GREEN, False, "", signals, rejected)
        assert code == "actionable"
        assert "1" in text


# ===================================================================
# _rejected_signals — regime-block path (v1 scope)
# ===================================================================


class TestRejectedSignals:
    def test_ranging_blocks_trend_strategy(self):
        config = {"market_state": {"enabled": True}}
        ms = MagicMock()
        ms.regime = MarketRegime.RANGING
        ms.volatility = Volatility.NORMAL
        signals = [{"symbol": "AAPL", "strategy": "turtle_trading", "signal": 1}]

        rejected = _rejected_signals(config, ms, signals, errors=[])
        assert len(rejected) == 1
        assert rejected[0]["symbol"] == "AAPL"
        assert "RANGING_BLOCK" in rejected[0]["reason"]

    def test_trending_market_allows_trend_strategy(self):
        config = {"market_state": {"enabled": True}}
        ms = MagicMock()
        ms.regime = MarketRegime.TRENDING_UP
        ms.volatility = Volatility.NORMAL
        signals = [{"symbol": "AAPL", "strategy": "turtle_trading", "signal": 1}]

        rejected = _rejected_signals(config, ms, signals, errors=[])
        assert rejected == []

    def test_sell_signals_never_blocked(self):
        config = {"market_state": {"enabled": True}}
        ms = MagicMock()
        ms.regime = MarketRegime.RANGING
        ms.volatility = Volatility.NORMAL
        signals = [{"symbol": "AAPL", "strategy": "turtle_trading", "signal": -1}]

        rejected = _rejected_signals(config, ms, signals, errors=[])
        assert rejected == []

    def test_empty_signals_short_circuits(self):
        assert _rejected_signals({}, None, [], errors=[]) == []

    def test_market_state_disabled_never_blocks(self):
        config = {"market_state": {"enabled": False}}
        ms = MagicMock()
        ms.regime = MarketRegime.RANGING
        ms.volatility = Volatility.NORMAL
        signals = [{"symbol": "AAPL", "strategy": "turtle_trading", "signal": 1}]

        rejected = _rejected_signals(config, ms, signals, errors=[])
        assert rejected == []


# ===================================================================
# _risk_delta — snapshot round-trip via risk_state key/value store
# ===================================================================


class _MockPriceSource(DataSource):
    """Deterministic multi-symbol OHLCV so correlation/VaR math is stable."""

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def supports(self, symbol: str) -> bool:
        return True

    def fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        dates = pd.bdate_range("2023-01-01", periods=280)
        rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
        close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(dates))))
        df = pd.DataFrame({
            "Open": close * 0.99, "High": close * 1.02,
            "Low": close * 0.98, "Close": close,
            "Volume": rng.integers(1_000_000, 10_000_000, len(dates)),
        }, index=dates)
        df.index.name = "date"
        return df


@pytest.fixture
def price_provider(temp_cache):
    """DataProvider wired to synthetic data only.

    data/provider.py._fetch_from_sources looks sources up *by name*
    against the hardcoded SOURCE_PRIORITY list (data/protocol.py) — it
    does not just use whatever's in ``sources``. So the mock must be
    registered under the real source names ("sina_us" for US equities,
    "cboe" for ^VIX, "tencent" as the default fallback) or lookups miss
    it entirely and silently fall through to empty data / real network
    calls.
    """
    from data import DataProvider
    sources = [_MockPriceSource(n) for n in ("sina_us", "cboe", "tencent")]
    return DataProvider(cache=temp_cache, sources=sources)


@pytest.fixture
def two_symbol_config():
    # active="no_such_strategy" makes compute_hypothetical_positions skip
    # both symbols deterministically (see analysis/hypothetical_positions.py:
    # `if strat_name not in STRATEGY_MAP: continue`), so _build_weights
    # reliably falls back to equal-weight watchlist instead of depending
    # on whether the synthetic random walk happens to have an open
    # Chandelier-stop position on a given symbol.
    return {
        "scanner": {"lookback_years": 1},
        "watchlist": [
            {"symbol": "AAA", "name": "AAA", "active": "no_such_strategy", "monitor": []},
            {"symbol": "BBB", "name": "BBB", "active": "no_such_strategy", "monitor": []},
        ],
    }


class TestRiskDelta:
    def test_no_prior_snapshot_returns_none_delta(self, temp_cache, price_provider, two_symbol_config):
        target = date(2023, 10, 2)
        snapshot, delta = _risk_delta(two_symbol_config, target, price_provider, temp_cache, errors=[])
        assert snapshot is not None
        assert delta is None

    def test_writes_snapshot_for_diffing_tomorrow(self, temp_cache, price_provider, two_symbol_config):
        target = date(2023, 10, 2)
        _risk_delta(two_symbol_config, target, price_provider, temp_cache, errors=[])

        raw = temp_cache.load_risk_state(f"risk_snapshot:{target.isoformat()}")
        assert raw is not None
        stored = json.loads(raw)
        assert "var_pct" in stored
        assert "hhi" in stored

    def test_delta_computed_against_yesterday(self, temp_cache, price_provider, two_symbol_config):
        yesterday = date(2023, 10, 1)
        today = date(2023, 10, 2)

        # Seed yesterday's snapshot directly (round-trip: write via the
        # same key format _risk_delta reads, independent of its own write).
        temp_cache.save_risk_state(
            f"risk_snapshot:{yesterday.isoformat()}",
            json.dumps({"var_pct": 1.0, "hhi": 5000.0, "sector_hhi": 6000.0, "effective_n": 2.0}),
        )

        snapshot, delta = _risk_delta(two_symbol_config, today, price_provider, temp_cache, errors=[])
        assert snapshot is not None
        assert delta is not None
        assert delta["hhi"] == pytest.approx(snapshot["hhi"] - 5000.0)
        assert delta["var_pct"] == pytest.approx(snapshot["var_pct"] - 1.0)

    def test_insufficient_symbols_returns_none_without_raising(self, temp_cache, price_provider):
        config = {
            "scanner": {"lookback_years": 1},
            "watchlist": [{"symbol": "AAA", "name": "AAA", "active": "weekly_macd", "monitor": []}],
        }
        errors = []
        snapshot, delta = _risk_delta(config, date(2023, 10, 2), price_provider, temp_cache, errors)
        assert snapshot is None
        assert delta is None
        assert errors  # recorded, not raised


# ===================================================================
# build_decision_summary — end-to-end smoke test (no crash, well-formed shape)
# ===================================================================


class TestBuildDecisionSummarySmoke:
    def test_returns_well_formed_dict_without_raising(self, temp_cache, price_provider, two_symbol_config):
        summary = build_decision_summary(
            two_symbol_config, date(2023, 10, 2), price_provider, temp_cache,
        )
        for key in (
            "verdict_code", "verdict_text", "risk_level", "risk_reasons",
            "regime", "volatility", "signals", "signals_total",
            "rejected", "rejected_count", "risk_snapshot", "risk_delta",
            "trading_paused", "pause_reason", "errors",
        ):
            assert key in summary
        assert isinstance(summary["verdict_text"], str) and summary["verdict_text"]

    def test_empty_watchlist_does_not_raise(self, temp_cache, price_provider):
        summary = build_decision_summary(
            {"watchlist": []}, date(2023, 10, 2), price_provider, temp_cache,
        )
        assert summary["verdict_code"] == "no_signal"
