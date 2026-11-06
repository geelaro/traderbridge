"""Tests for analysis/risk_report.py — weekly report aggregator."""

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from analysis.risk_report import RiskReport, Section


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _synthetic_prices(symbols, n_days=500, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    data = {}
    for i, sym in enumerate(symbols):
        rets = rng.normal(0.0005, 0.015, n_days)
        data[sym] = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def mock_provider():
    """Provider that returns deterministic synthetic OHLCV for any symbol."""
    p = MagicMock()
    # Cache by (symbol, start, end) for determinism across calls
    _cache = {}

    def get_daily(symbol, start=None, end=None):
        if symbol in _cache:
            return _cache[symbol]
        # Generate 500 days of plausible OHLCV
        df = _synthetic_prices([symbol]).rename(columns={symbol: "Close"})
        df["Open"] = df["Close"] * 0.999
        df["High"] = df["Close"] * 1.005
        df["Low"] = df["Close"] * 0.995
        df["Volume"] = 1_000_000
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        if end:
            df = df[df.index <= pd.Timestamp(end)]
        _cache[symbol] = df
        return df

    p.get_daily = MagicMock(side_effect=get_daily)
    return p


@pytest.fixture
def config():
    return {
        "watchlist": [
            {"symbol": "AAPL", "active": "weekly_macd_kdj"},
            {"symbol": "MSFT", "active": "weekly_macd_kdj"},
            {"symbol": "NVDA", "active": "weekly_macd_kdj"},
        ],
        "strategy": {"weekly_macd_kdj": {}},
    }


@pytest.fixture
def report(config, mock_provider, temp_cache):
    return RiskReport(config, mock_provider, temp_cache,
                      target_date=date(2025, 6, 15))


# ===================================================================
# Section dataclass
# ===================================================================


class TestSection:
    def test_defaults(self):
        s = Section(title="t")
        assert s.title == "t"
        assert s.summary == ""
        assert s.metrics == {}
        assert s.warnings == []
        assert s.tables == []


# ===================================================================
# RiskReport — top-level build
# ===================================================================


class TestRiskReportBuild:
    def test_build_returns_structure(self, report):
        result = report.build()
        assert result["as_of"] == "2025-06-15"
        assert result["watchlist_size"] == 3
        assert isinstance(result["sections"], list)
        # Should have ~9 sections — exact count may vary if any silently
        # skip; require at least 5 non-empty ones.
        assert len(result["sections"]) >= 5

    def test_individual_section_failures_dont_kill_build(
            self, config, mock_provider, temp_cache):
        """If one section raises, the rest still build."""
        r = RiskReport(config, mock_provider, temp_cache,
                       target_date=date(2025, 6, 15))

        def _boom():
            raise RuntimeError("boom")

        _boom.__name__ = "_build_var"
        r._build_var = _boom
        result = r.build()
        # Failure surfaces in the section's warnings with exception type + msg
        failed = [s for s in result["sections"]
                  if s.warnings and "RuntimeError" in s.warnings[0]
                  and "boom" in s.warnings[0]]
        assert len(failed) == 1

    def test_empty_watchlist(self, mock_provider, temp_cache):
        r = RiskReport({"watchlist": []}, mock_provider, temp_cache,
                       target_date=date(2025, 6, 15))
        result = r.build()
        assert result["watchlist_size"] == 0
        # Should still build (most sections will warn but not crash)
        assert isinstance(result["sections"], list)


# ===================================================================
# Section content
# ===================================================================


class TestSectionContent:
    def test_risk_light_present(self, report):
        result = report.build()
        rl = next((s for s in result["sections"] if "风险灯" in s.title), None)
        assert rl is not None
        # Has SPY / VIX metrics or a warning
        assert rl.metrics or rl.warnings
        if rl.metrics:
            assert "RSI14" in rl.metrics
            assert "顶部风险" in rl.metrics

    def test_concentration_metrics(self, report):
        result = report.build()
        conc = next((s for s in result["sections"] if "集中度" in s.title), None)
        assert conc is not None
        assert "持仓数" in conc.metrics or conc.warnings

    def test_pnl_section_always_renders(self, report):
        """PnL works even with empty trade_pnl + no positions."""
        result = report.build()
        pnl = next((s for s in result["sections"] if "盈亏" in s.title), None)
        assert pnl is not None
        # Should have totals even if zero
        assert "Realized 笔数" in pnl.metrics


# ===================================================================
# Output formats
# ===================================================================


class TestOutputFormats:
    def test_markdown_renders(self, report):
        md = report.to_markdown()
        assert "# Traderbridge 风险报告 — 2025-06-15" in md
        assert "## " in md  # has section headers
        # Should be a multi-line string
        assert md.count("\n") > 10

    def test_markdown_includes_metrics(self, report):
        md = report.to_markdown()
        # All sections that produced metrics should appear as bullet points
        assert "**" in md  # bold markdown for metric labels

    def test_feishu_card_shape(self, report):
        card = report.to_feishu_card()
        assert "header" in card
        assert card["header"]["template"] == "blue"
        assert "2025-06-15" in card["header"]["title"]["content"]
        assert isinstance(card["elements"], list)
        # Each section produces a div + hr
        divs = [e for e in card["elements"] if e.get("tag") == "div"]
        assert len(divs) >= 5

    def test_feishu_card_no_empty_content(self, report):
        """Every div has non-empty content."""
        card = report.to_feishu_card()
        for e in card["elements"]:
            if e.get("tag") == "div":
                assert e["text"]["content"].strip()
