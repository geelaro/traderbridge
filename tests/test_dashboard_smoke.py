"""Dashboard smoke tests — v1 scope, see ROADMAP.md 阶段八 CI-1.

Two layers:
1. Import every dashboard submodule — catches module-level syntax/
   dependency errors for zero cost across the whole package.
2. Bare-mode render smoke for the 3 highest-traffic functions
   (decision summary card, ops health, risk light) — calling Streamlit
   `st.*` functions outside a real ScriptRunContext only warns, it
   doesn't raise (confirmed manually during D-1/Ops-1 development), so
   this is a real regression net even without a browser.

NOT covered (explicitly out of scope, see ROADMAP.md): single_backtest,
portfolio_backtest, factor_attribution, brinson_attribution, kill_switch,
and the other tab renderers each need their own broker/backtest-year/
selected-symbol fixtures — that's follow-up work, not this pass.

All fixtures here are offline (mock DataSource registered under real
SOURCE_PRIORITY names — see data/protocol.py — no real network calls),
consistent with this repo's "1043 tests, all offline" convention.
"""

import importlib
import zlib

import numpy as np
import pandas as pd
import pytest

from data import DataProvider
from data.protocol import DataSource


DASHBOARD_MODULES = [
    "dashboard.brinson_attribution",
    "dashboard.config_editor",
    "dashboard.decision_review",
    "dashboard.decision_summary",
    "dashboard.factor_attribution",
    "dashboard.historical_analog",
    "dashboard.kill_switch",
    "dashboard.main",
    "dashboard.ops",
    "dashboard.pnl_breakdown",
    "dashboard.portfolio_backtest",
    "dashboard.risk_analytics",
    "dashboard.risk_report",
    "dashboard.signal_effectiveness",
    "dashboard.signal_history",
    "dashboard.signals",
    "dashboard.single_backtest",
]


@pytest.mark.parametrize("module_name", DASHBOARD_MODULES)
def test_module_imports_without_error(module_name):
    importlib.import_module(module_name)


# ===================================================================
# Bare-mode render smoke — offline mock provider
# ===================================================================


class _MockPriceSource(DataSource):
    """Registered under real SOURCE_PRIORITY names — see data/protocol.py.
    DataProvider._fetch_from_sources looks sources up *by name*, so a
    mock named e.g. "mock" would silently never be selected."""

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def supports(self, symbol: str) -> bool:
        return True

    def fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        dates = pd.bdate_range("2023-01-01", periods=280)
        # crc32 (not hash()) so the series is reproducible across runs —
        # str.__hash__ is randomized per-process (PYTHONHASHSEED).
        rng = np.random.default_rng(zlib.crc32(symbol.encode("utf-8")))
        close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(dates))))
        df = pd.DataFrame({
            "Open": close * 0.99, "High": close * 1.02,
            "Low": close * 0.98, "Close": close,
            "Volume": rng.integers(1_000_000, 10_000_000, len(dates)),
        }, index=dates)
        df.index.name = "date"
        return df


@pytest.fixture
def smoke_provider(temp_cache):
    sources = [_MockPriceSource(n) for n in ("sina_us", "cboe", "tencent")]
    return DataProvider(cache=temp_cache, sources=sources)


@pytest.fixture
def smoke_config():
    return {
        "scanner": {"lookback_years": 1},
        "watchlist": [
            {"symbol": "AAA", "name": "AAA", "active": "no_such_strategy", "monitor": []},
            {"symbol": "BBB", "name": "BBB", "active": "no_such_strategy", "monitor": []},
        ],
    }


class TestRenderSmoke:
    def test_render_decision_summary(self, smoke_config, smoke_provider, temp_cache):
        from dashboard.decision_summary import build_decision_summary, render_decision_summary
        summary = build_decision_summary(smoke_config, pd.Timestamp("2023-10-02").date(),
                                          smoke_provider, temp_cache)
        render_decision_summary(summary)  # no assertion beyond "doesn't raise"

    def test_render_ops(self, smoke_config, smoke_provider, temp_cache):
        from dashboard.ops import render_ops
        render_ops(smoke_config, temp_cache, smoke_provider)

    def test_render_risk_light(self, smoke_config, smoke_provider):
        from dashboard.signals import render_risk_light
        render_risk_light(smoke_config, pd.Timestamp("2023-10-02").date(), smoke_provider)

    def test_render_signal_history(self, smoke_config, temp_cache):
        from dashboard.signal_history import render_signal_history

        # Same (symbol, strategy, bar_date, signal) persisted on 3 different
        # scan_dates — mirrors a weekly signal being re-confirmed daily until
        # the bar rolls over. Exercises the dedup/groupby path, not just the
        # empty-table path.
        for scan_date in ("2023-10-01", "2023-10-02", "2023-10-03"):
            temp_cache.save_signal(
                scan_date=scan_date, symbol="AAA", strategy="weekly_macd",
                bar_date="2023-09-29", signal=-1, price=101.5, atr=2.1,
                indicators='{"MACD": -0.3}',
            )
        temp_cache.save_signal(
            scan_date="2023-10-03", symbol="BBB", strategy="weekly_macd",
            bar_date="2023-10-03", signal=1, price=50.0, atr=1.0,
            indicators='{"MACD": 0.2}',
        )
        # CN symbol — exercises the ¥ price formatting branch
        temp_cache.save_signal(
            scan_date="2023-10-03", symbol="510300", strategy="weekly_macd_kdj",
            bar_date="2023-10-03", signal=1, price=4.6, atr=0.1,
            indicators="{}",
        )
        render_signal_history(temp_cache, smoke_config)
