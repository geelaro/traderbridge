"""Tests for utils/signal_scanner.py — SignalScanner + enrich_scan_items."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from tests.conftest import make_ohlcv
from utils.signal_scanner import SignalScanner, enrich_scan_items


@pytest.fixture
def minimal_config():
    return {
        "watchlist": [
            {"symbol": "AAPL", "name": "Apple", "active": "weekly_macd", "monitor": []},
        ],
        "strategy": {
            "weekly_macd": {"macd_fast": 8, "macd_slow": 17, "macd_signal": 9},
        },
    }


@pytest.fixture
def multi_config():
    return {
        "watchlist": [
            {"symbol": "AAPL", "name": "Apple", "active": "weekly_macd", "monitor": ["turtle_trading"]},
            {"symbol": "NVDA", "name": "NVIDIA", "active": "weekly_macd", "monitor": []},
        ],
        "strategy": {
            "weekly_macd": {"macd_fast": 8, "macd_slow": 17, "macd_signal": 9},
        },
    }


@pytest.fixture
def empty_config():
    return {"watchlist": []}


@pytest.fixture
def synthetic_ohlcv():
    return make_ohlcv(300)


@pytest.fixture
def mock_provider(synthetic_ohlcv):
    provider = MagicMock()
    provider.get_daily.return_value = synthetic_ohlcv
    return provider


@pytest.fixture
def scanner(mock_provider):
    return SignalScanner(provider=mock_provider, lookback_years=1)


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------


class TestSignalScannerScan:
    def test_scan_with_valid_config_returns_signals(self, scanner, minimal_config):
        results = scanner.scan(minimal_config, target_date="2021-01-04")
        assert len(results) > 0
        for r in results:
            assert "symbol" in r
            assert "strategy" in r
            assert "signal" in r
            assert r["symbol"] == "AAPL"
            assert r["strategy"] == "weekly_macd"

    def test_scan_with_empty_watchlist_returns_empty(self, scanner, empty_config):
        results = scanner.scan(empty_config, target_date="2021-01-04")
        assert results == []

    def test_orphan_positions_are_scanned(self, scanner, minimal_config):
        orphans = [
            {"symbol": "MSFT", "name": "Microsoft", "strategy": "weekly_macd"},
        ]
        results = scanner.scan(minimal_config, target_date="2021-01-04", orphan_positions=orphans)
        msft_results = [r for r in results if r["symbol"] == "MSFT"]
        assert len(msft_results) > 0
        assert msft_results[0]["orphan"] is True

    def test_duplicate_orphan_not_rescanned(self, scanner, minimal_config):
        orphans = [
            {"symbol": "AAPL", "name": "Apple", "strategy": "weekly_macd"},
        ]
        results = scanner.scan(minimal_config, target_date="2021-01-04", orphan_positions=orphans)
        apple_results = [r for r in results if r["symbol"] == "AAPL"]
        # AAPL is in watchlist, orchans should not duplicate it
        assert all(r["orphan"] is False for r in apple_results)

    def test_data_fetch_failure_returns_empty(self, scanner, minimal_config):
        scanner.provider.get_daily.return_value = None
        results = scanner.scan(minimal_config, target_date="2021-01-04")
        assert results == []

    def test_data_fetch_empty_df_returns_empty(self, scanner, minimal_config):
        scanner.provider.get_daily.return_value = pd.DataFrame()
        results = scanner.scan(minimal_config, target_date="2021-01-04")
        assert results == []

    def test_signal_persistence_when_cache_provided(self, synthetic_ohlcv, minimal_config, temp_cache):
        provider = MagicMock()
        provider.get_daily.return_value = synthetic_ohlcv
        scanner = SignalScanner(provider=provider, cache=temp_cache, lookback_years=1)
        results = scanner.scan(minimal_config, target_date="2021-01-04")
        assert len(results) > 0

    def test_strategy_type_error_is_not_retried_without_weekly_data(
        self, scanner, minimal_config, monkeypatch,
    ):
        import utils.signal_scanner as scanner_module

        calls = []

        class BrokenStrategy:
            def __init__(self, **kwargs):
                pass

            def calculate_indicators(self, df, df_weekly=None):
                calls.append(df_weekly)
                raise TypeError("strategy implementation bug")

        monkeypatch.setitem(scanner_module.STRATEGY_MAP, "weekly_macd", BrokenStrategy)

        assert scanner.scan(minimal_config, target_date="2021-01-04") == []
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# Missing target date — falls back to latest bar, never skips
# ---------------------------------------------------------------------------


class TestSignalScannerMissingData:
    """SignalScanner._scan_symbol — target_date not in cached data."""

    def test_missing_target_date_uses_latest_bar(self, scanner, minimal_config):
        """target_date beyond synthetic data range → fall back to last bar."""
        results = scanner.scan(minimal_config, target_date="2025-01-01")
        assert len(results) > 0
        for r in results:
            assert r["symbol"] == "AAPL"
            assert "bar_date" in r
            # bar_date should be the last available date, not 2025-01-01
            assert r["bar_date"] != "2025-01-01"

    def test_missing_target_date_multiple_strategies(self, scanner, multi_config):
        results = scanner.scan(multi_config, target_date="2025-01-01")
        assert len(results) == 3  # AAPL: weekly_macd+turtle, NVDA: weekly_macd
        for r in results:
            assert r["bar_date"] != "2025-01-01"

    def test_missing_target_date_still_generates_valid_signals(self, scanner, minimal_config):
        results = scanner.scan(minimal_config, target_date="2025-01-01")
        for r in results:
            assert "signal" in r
            assert r["signal"] in (0, 1)
            assert r["price"] > 0
            assert r["atr"] >= 0


class TestSignalScannerMulti:
    def test_multi_symbol_scan(self, scanner, multi_config):
        results = scanner.scan(multi_config, target_date="2021-01-04")
        symbols = {r["symbol"] for r in results}
        assert "AAPL" in symbols
        assert "NVDA" in symbols

    def test_monitor_strategies_included(self, scanner, multi_config):
        results = scanner.scan(multi_config, target_date="2021-01-04")
        aapl_strategies = {r["strategy"] for r in results if r["symbol"] == "AAPL"}
        assert "weekly_macd" in aapl_strategies
        assert "turtle_trading" in aapl_strategies

    def test_orphan_scanned_when_active_in_params(self, scanner, multi_config):
        orphans = [
            {"symbol": "TSLA", "name": "Tesla", "strategy": "weekly_macd"},
        ]
        results = scanner.scan(multi_config, target_date="2021-01-04", orphan_positions=orphans)
        tsla_results = [r for r in results if r["symbol"] == "TSLA"]
        assert len(tsla_results) > 0
        assert all(r["orphan"] is True for r in tsla_results)


# ---------------------------------------------------------------------------
# enrich_scan_items
# ---------------------------------------------------------------------------


class TestEnrichScanItems:
    def test_bakes_strategy_params_into_items(self, minimal_config):
        items = enrich_scan_items(minimal_config)
        assert len(items) == 1
        assert items[0]["params"] == {"macd_fast": 8, "macd_slow": 17, "macd_signal": 9}

    def test_empty_watchlist_returns_empty(self, empty_config):
        items = enrich_scan_items(empty_config)
        assert items == []

    def test_missing_strategy_params_defaults_to_empty_dict(self):
        config = {
            "watchlist": [
                {"symbol": "SPY", "active": "turtle_trading", "monitor": []},
            ],
        }
        items = enrich_scan_items(config)
        assert items[0]["params"] == {}

    def test_preserves_original_item_fields(self):
        config = {
            "watchlist": [
                {"symbol": "QQQ", "name": "Nasdaq ETF", "active": "weekly_macd", "monitor": ["turtle_trading"]},
            ],
            "strategy": {
                "weekly_macd": {"macd_fast": 10},
            },
        }
        items = enrich_scan_items(config)
        assert items[0]["symbol"] == "QQQ"
        assert items[0]["name"] == "Nasdaq ETF"
        assert items[0]["active"] == "weekly_macd"
        assert items[0]["monitor"] == ["turtle_trading"]
        assert items[0]["params"] == {"macd_fast": 10}
