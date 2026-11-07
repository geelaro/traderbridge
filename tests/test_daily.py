"""Tests for daily.py — config loading, scan_day, signal history."""

import json
import os
import tempfile
from datetime import date
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from daily import load_config, scan_day, show_history, print_summary
from data.cache import CacheManager
from data.protocol import DataSource, OHLCV_COLUMNS


# ===================================================================
# Mock data source for controlled testing
# ===================================================================


class _MockDailySource(DataSource):
    """Returns a known OHLCV dataframe so scan_day is deterministic."""

    def __init__(self, data=None):
        self._name = "mock"
        self._data = data or self._make_data()

    @property
    def name(self) -> str:
        return self._name

    def supports(self, symbol: str) -> bool:
        return True

    def fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        return self._data

    @staticmethod
    def _make_data() -> pd.DataFrame:
        dates = pd.bdate_range("2025-01-01", periods=120)
        import numpy as np
        rng = np.random.default_rng(42)
        close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(dates))))
        df = pd.DataFrame({
            "Open": close * 0.99, "High": close * 1.02,
            "Low": close * 0.98, "Close": close,
            "Volume": rng.integers(1_000_000, 10_000_000, len(dates)),
        }, index=dates)
        df.index.name = "date"
        return df


# ===================================================================
# load_config
# ===================================================================


class TestLoadConfig:
    def test_loads_valid_toml(self):
        cfg = load_config("watchlist.toml")
        assert "watchlist" in cfg
        assert "scanner" in cfg
        assert len(cfg["watchlist"]) >= 1

    def test_watchlist_has_active_field(self):
        cfg = load_config("watchlist.toml")
        for item in cfg["watchlist"]:
            assert "active" in item or "strategies" in item


# ===================================================================
# scan_day
# ===================================================================


class TestScanDay:
    def test_returns_empty_for_empty_watchlist(self, temp_cache):
        config = {"watchlist": []}
        results = scan_day(config, target_date="2025-01-15", cache=temp_cache)
        assert results == []

    def test_returns_signals_for_mock_data(self, temp_cache):
        mock_src = _MockDailySource()
        from data import DataProvider
        provider = DataProvider(cache=temp_cache, sources=[mock_src])

        config = {
            "watchlist": [{
                "symbol": "TEST",
                "name": "TestStock",
                "active": "weekly_macd",
                "monitor": [],
            }],
            "strategy": {"weekly_macd": {}},
        }

        results = scan_day(config, target_date="2025-06-01",
                           provider=provider, cache=temp_cache)
        assert isinstance(results, list)
        # weekly_macd resamples, so we may get 0 or 1 result
        for r in results:
            assert "symbol" in r
            assert "strategy" in r
            assert "signal" in r
            assert "symbol_sentiment" in r
            assert r["symbol"] == "TEST"

    def test_respects_active_monitor_split(self, temp_cache):
        """active + monitor strategies should both be scanned without error."""
        mock_src = _MockDailySource()
        from data import DataProvider
        provider = DataProvider(cache=temp_cache, sources=[mock_src])

        config = {
            "watchlist": [{
                "symbol": "TEST",
                "name": "TestStock",
                "active": "trend_follower",
                "monitor": ["turtle_trading"],
            }],
            "strategy": {"trend_follower": {}, "turtle_trading": {}},
        }
        results = scan_day(config, target_date="2025-06-01",
                           provider=provider, cache=temp_cache)
        # scan should complete without error for both active and monitor
        assert isinstance(results, list)

    def test_skips_unknown_strategy(self, temp_cache):
        mock_src = _MockDailySource()
        from data import DataProvider
        provider = DataProvider(cache=temp_cache, sources=[mock_src])

        config = {
            "watchlist": [{
                "symbol": "TEST",
                "name": "TestStock",
                "active": "no_such_strategy",
                "monitor": [],
            }],
        }
        results = scan_day(config, target_date="2025-06-01",
                           provider=provider, cache=temp_cache)
        assert results == []


# ===================================================================
# show_history
# ===================================================================


class TestShowHistory:
    def test_empty_cache(self, temp_cache, capsys):
        show_history(cache=temp_cache, days=7)
        captured = capsys.readouterr()
        assert "无扫描记录" in captured.out

    def test_with_data(self, temp_cache, capsys):
        temp_cache.save_signal("2025-01-15", "AAPL", "weekly_macd",
                               "2025-01-15", 1, 195.0, 5.0, "")
        show_history(cache=temp_cache, days=30)
        captured = capsys.readouterr()
        # Assert UTF-8 content without relying on Chinese encoding
        assert "AAPL" in captured.out or "30" in captured.out


# ===================================================================
# print_summary
# ===================================================================


class TestPrintSummary:
    def test_empty_signals(self, capsys):
        print_summary([])
        assert "无买入/卖出信号" in capsys.readouterr().out

    def test_with_buy_signal(self, capsys):
        results = [{
            "symbol": "AAPL", "name": "Apple", "strategy": "weekly_macd",
            "signal": 1, "price": 195.0, "atr": 5.0,
            "bar_date": "2025-01-15", "indicators": {},
        }]
        print_summary(results)
        out = capsys.readouterr().out
        assert "AAPL" in out
        assert "1 个标的" in out or "买入" in out

    def test_with_sell_signal(self, capsys):
        results = [{
            "symbol": "AAPL", "name": "Apple", "strategy": "weekly_macd",
            "signal": -1, "price": 200.0, "atr": 5.0,
            "bar_date": "2025-01-15", "indicators": {},
        }]
        print_summary(results)
        out = capsys.readouterr().out
        assert "AAPL" in out
