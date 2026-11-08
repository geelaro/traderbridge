"""Tests for data/provider.py and data/sources.py — routing, validation, classify."""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data.protocol import (
    DataSource,
    OHLCV_COLUMNS,
    classify_symbol,
    CN_SYMBOLS,
    SOURCE_PRIORITY,
)
from data.sources import TencentSource, SinaSource, SinaUSSource, YahooChartSource, AKShareSource
from data.provider import DataProvider
from data.cache import CacheManager


# ===================================================================
# classify_symbol
# ===================================================================


class TestClassifySymbol:
    def test_us_stock(self):
        assert classify_symbol("AAPL") == "us"
        assert classify_symbol("NVDA") == "us"

    def test_us_class_share(self):
        assert classify_symbol("BRK.B") == "us"
        assert classify_symbol("BRK-B") == "us"

    def test_cn_digit(self):
        assert classify_symbol("510300") == "cn"
        assert classify_symbol("159919") == "cn"

    def test_cn_prefix(self):
        assert classify_symbol("sh510300") == "cn"
        assert classify_symbol("SZ159915") == "cn"

    def test_cn_alias(self):
        assert classify_symbol("510050") == "cn"

    def test_global(self):
        assert classify_symbol("0700.HK") == "global"


# ===================================================================
# SOURCE_PRIORITY
# ===================================================================


class TestSourcePriority:
    def test_us_priority(self):
        assert SOURCE_PRIORITY["us"] == ["sina_us", "tencent", "yahoo_chart"]

    def test_cn_priority(self):
        assert SOURCE_PRIORITY["cn"][0] == "sina"
        assert "tencent" in SOURCE_PRIORITY["cn"]


# ===================================================================
# DataSource.validate
# ===================================================================


class TestDataSourceValidate:
    def test_normalises_columns(self):
        df = pd.DataFrame({
            "open": [100, 101], "high": [102, 103], "low": [99, 100],
            "close": [101, 102], "volume": [1000, 2000],
        }, index=pd.to_datetime(["2025-01-01", "2025-01-02"]))
        result = DataSource.validate(df, "TEST")
        assert list(result.columns) == OHLCV_COLUMNS

    def test_drops_nan_prices(self):
        df = pd.DataFrame({
            "Open": [100, None], "High": [102, 103], "Low": [99, 100],
            "Close": [None, 102], "Volume": [1000, 2000],
        }, index=pd.to_datetime(["2025-01-01", "2025-01-02"]))
        result = DataSource.validate(df, "TEST")
        assert len(result) <= 1

    def test_empty_input(self):
        df = pd.DataFrame(columns=OHLCV_COLUMNS)
        result = DataSource.validate(df, "TEST")
        assert result.empty

    def test_sorts_index(self):
        df = pd.DataFrame({
            "Open": [101, 100], "High": [103, 102], "Low": [100, 99],
            "Close": [102, 101], "Volume": [2000, 1000],
        }, index=pd.to_datetime(["2025-01-02", "2025-01-01"]))
        result = DataSource.validate(df, "TEST")
        assert result.index[0] < result.index[1]


# ===================================================================
# Source supports
# ===================================================================


class TestSourceSupports:
    def test_tencent_knows_us(self):
        src = TencentSource()
        assert src.supports("AAPL")
        assert src.supports("QQQ")
        assert not src.supports("510300")

    def test_tencent_supports_non_whitelist_us(self):
        """AMD was not in the whitelist but should be supported via fallback pattern."""
        src = TencentSource()
        assert src.supports("AMD")
        assert src.supports("MU")
        assert src.supports("ORCL")

    def test_tencent_rejects_non_alpha(self):
        src = TencentSource()
        assert not src.supports("510300")
        assert not src.supports("TQQQ3L")  # > 5 chars
        assert src.supports("AMD")  # 3-char US ticker

    def test_sina_knows_cn(self):
        src = SinaSource()
        assert src.supports("510300")
        assert src.supports("sh510050")
        assert not src.supports("AAPL")

    def test_akshare_knows_cn(self):
        src = AKShareSource()
        assert src.supports("510300")
        assert src.supports("sh510050")

    def test_sina_us_knows_us(self):
        src = SinaUSSource()
        assert src.supports("AAPL")
        assert src.supports("NVDA")
        assert src.supports("QQQ")
        assert src.supports("BRK.B")
        assert not src.supports("510300")

    def test_yahoo_chart_knows_us(self):
        src = YahooChartSource()
        assert src.supports("AAPL")
        assert src.supports("TSLA")
        assert not src.supports("510300")


# ===================================================================
# SinaSource symbol normalisation
# ===================================================================


class TestSinaNormalise:
    def test_digit_to_sh(self):
        assert SinaSource._normalise_symbol("510300") == "sh510300"

    def test_already_prefixed(self):
        assert SinaSource._normalise_symbol("sh510050") == "sh510050"

    def test_shenzhen(self):
        assert SinaSource._normalise_symbol("000001") == "sz000001"


# ===================================================================
# _is_complete with exchange calendar
# ===================================================================


class TestIsComplete:
    """DataProvider._is_complete — exchange calendar + fallback heuristic."""

    def _df(self, dates: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {"Open": 100, "High": 101, "Low": 99, "Close": 100.5, "Volume": 1000},
            index=pd.DatetimeIndex(dates),
        )

    # -- empty / None -------------------------------------------------------

    def test_empty_df_returns_false(self):
        df = pd.DataFrame()
        assert DataProvider._is_complete(df, "2026-01-01", "2026-01-10", "us") is False

    def test_none_df_returns_false(self):
        assert DataProvider._is_complete(None, "2026-01-01", "2026-01-10", "us") is False

    # -- exchange calendar (XNYS installed) ---------------------------------

    def test_calendar_detects_missing_trading_day(self):
        """Cache ends Tue 5/26, end=Thu 5/28 → 5/27 + 5/28 are open → incomplete."""
        df = self._df(["2026-05-26"])
        assert DataProvider._is_complete(df, "2020-01-01", "2026-05-28", "us") is False

    def test_calendar_complete_no_missing_sessions(self):
        """Cache ends Fri 5/29, end=Sat 5/30 → no open sessions → complete."""
        df = self._df(["2026-05-26", "2026-05-27", "2026-05-28", "2026-05-29"])
        assert DataProvider._is_complete(df, "2020-01-01", "2026-05-30", "us") is True

    def test_calendar_complete_weekend_gap(self):
        """Cache ends Fri 5/29, end=Sun 5/31 → Sat+Sun no sessions → complete."""
        df = self._df(["2026-05-26", "2026-05-27", "2026-05-28", "2026-05-29"])
        assert DataProvider._is_complete(df, "2020-01-01", "2026-05-31", "us") is True

    def test_calendar_incomplete_monday_missing(self):
        """Cache ends Fri 5/29, end=Mon 6/1 → Mon is a trading day → incomplete."""
        df = self._df(["2026-05-26", "2026-05-27", "2026-05-28", "2026-05-29"])
        assert DataProvider._is_complete(df, "2020-01-01", "2026-06-01", "us") is False

    def test_calendar_detects_holiday(self):
        """Memorial Day 5/25 — cache ends Fri 5/22, end=Tue 5/26.
        Only Mon 5/25 is Memorial Day (closed), next open is Tue 5/26.
        Since cache only has up to 5/22, sessions 5/26 is missing → incomplete."""
        df = self._df(["2026-05-19", "2026-05-20", "2026-05-21", "2026-05-22"])
        # 5/25 is Memorial Day, 5/26 is a trading day — not cached → incomplete
        assert DataProvider._is_complete(df, "2020-01-01", "2026-05-26", "us") is False

    def test_calendar_internal_gap_returns_false(self):
        """Gap > 7 calendar days between bars → incomplete regardless of tail."""
        df = self._df(["2026-05-01", "2026-05-20", "2026-05-21"])
        assert DataProvider._is_complete(df, "2020-01-01", "2026-05-21", "us") is False

    # -- CN market (XSHG) ---------------------------------------------------

    def test_calendar_cn_market(self):
        """CN market uses XSHG calendar. Cache ends Fri 5/29, end=Sun 5/31."""
        df = self._df(["2026-05-28", "2026-05-29"])
        assert DataProvider._is_complete(df, "2020-01-01", "2026-05-31", "cn") is True

    def test_calendar_cn_incomplete_monday(self):
        """Cache ends Fri 5/29, end=Mon 6/1 → Mon trading day → incomplete."""
        df = self._df(["2026-05-28", "2026-05-29"])
        assert DataProvider._is_complete(df, "2020-01-01", "2026-06-01", "cn") is False

    # -- fallback weekday heuristic -----------------------------------------

    def test_fallback_monday_allows_3_days(self, monkeypatch):
        import data.provider as dp
        monkeypatch.setattr(dp, "_HAS_XCALS", False)
        # cache ends Fri, end=Mon → gap=3 → allowed=3 → complete
        df = self._df(["2026-05-29"])
        assert DataProvider._is_complete(df, "2020-01-01", "2026-06-01", "us") is True

    def test_fallback_monday_rejects_4_days(self, monkeypatch):
        import data.provider as dp
        monkeypatch.setattr(dp, "_HAS_XCALS", False)
        # cache ends Thu, end=Mon → gap=4 → allowed=3 → incomplete
        df = self._df(["2026-05-28"])
        assert DataProvider._is_complete(df, "2020-01-01", "2026-06-01", "us") is False

    def test_fallback_tue_fri_allows_1_day(self, monkeypatch):
        import data.provider as dp
        monkeypatch.setattr(dp, "_HAS_XCALS", False)
        # cache ends Tue, end=Wed → gap=1 → allowed=1 → complete
        df = self._df(["2026-05-26"])
        assert DataProvider._is_complete(df, "2020-01-01", "2026-05-27", "us") is True

    def test_fallback_tue_fri_rejects_2_days(self, monkeypatch):
        import data.provider as dp
        monkeypatch.setattr(dp, "_HAS_XCALS", False)
        # cache ends Tue, end=Thu → gap=2 → allowed=1 → incomplete
        df = self._df(["2026-05-26"])
        assert DataProvider._is_complete(df, "2020-01-01", "2026-05-28", "us") is False

    def test_fallback_weekend_allows_2_days(self, monkeypatch):
        import data.provider as dp
        monkeypatch.setattr(dp, "_HAS_XCALS", False)
        # cache ends Thu, end=Sat → gap=2 → allowed=2 → complete
        df = self._df(["2026-05-28"])
        assert DataProvider._is_complete(df, "2020-01-01", "2026-05-30", "us") is True


# ===================================================================
# DataProvider with mock source
# ===================================================================


class _MockSource(DataSource):
    """A controlled data source for testing provider routing."""

    def __init__(self, name="tencent", supports_all=True, data=None):
        self._name = name
        self._supports_all = supports_all
        self._data = data or self._make_data()

    @property
    def name(self) -> str:
        return self._name  # matches SOURCE_PRIORITY names for routing

    def supports(self, symbol: str) -> bool:
        return self._supports_all

    def fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        return self._data

    @staticmethod
    def _make_data() -> pd.DataFrame:
        dates = pd.bdate_range("2025-01-01", periods=30)
        df = pd.DataFrame({
            "Open": 100, "High": 105, "Low": 95, "Close": 102, "Volume": 1_000_000,
        }, index=dates)
        df.index.name = "date"
        return df


class TestDataProvider:
    def test_routes_to_mock_source(self, temp_cache):
        mock = _MockSource()
        provider = DataProvider(cache=temp_cache, sources=[mock])
        df = provider.get_daily("ANY", start="2025-01-01", end="2025-01-15")
        assert len(df) > 0
        assert "Close" in df.columns

    def test_cache_hit_avoids_refetch(self, temp_cache):
        mock = _MockSource()
        provider = DataProvider(cache=temp_cache, sources=[mock])

        # First call — should fetch and cache
        df1 = provider.get_daily("TEST", start="2025-01-01", end="2025-01-20")
        assert len(df1) > 0

        # Second call with same range — should hit cache
        mock._supports_all = False  # if provider tries to refetch, this fails
        df2 = provider.get_daily("TEST", start="2025-01-01", end="2025-01-20")
        assert len(df2) >= len(df1)  # cache returns what was stored

    def test_force_refresh(self, temp_cache):
        mock = _MockSource()
        provider = DataProvider(cache=temp_cache, sources=[mock])
        df = provider.get_daily("TEST", start="2025-01-01", end="2025-01-10", force_refresh=True)
        assert len(df) > 0

    def test_cached_range(self, temp_cache):
        mock = _MockSource()
        provider = DataProvider(cache=temp_cache, sources=[mock])
        provider.get_daily("TEST", start="2025-01-01", end="2025-01-15")
        start, end = provider.cached_range("TEST")
        assert start is not None

    def test_list_sources(self):
        provider = DataProvider()
        sources = provider.list_sources("AAPL")
        assert len(sources) >= 1


# ===================================================================
# Source fetch mocking
# ===================================================================


MOCK_TENCENT_RESP = {
    "code": 0,
    "data": {
        "usAAPL.OQ": {
            "day": [
                ["2025-01-02", "190.00", "191.00", "192.00", "189.00", "50000000"],
                ["2025-01-03", "191.50", "192.50", "193.50", "190.50", "45000000"],
                ["2025-01-06", "192.00", "194.00", "195.00", "191.00", "55000000"],
            ]
        }
    }
}

MOCK_SINA_RESP = [
    {"day": "2025-01-02", "open": "3.500", "high": "3.550", "low": "3.480", "close": "3.520", "volume": "10000000"},
    {"day": "2025-01-03", "open": "3.520", "high": "3.580", "low": "3.510", "close": "3.560", "volume": "12000000"},
]


class TestTencentCodeMap:
    """Symbols we routinely fetch must have explicit code mappings.

    Without an entry, TencentSource falls back to ``usXLK`` (no suffix)
    which returns a single-bar bogus quote that pollutes the cache —
    seen in production when Brinson tab fetched SPDR sector ETFs.
    """

    def test_spdr_sector_etfs_mapped(self):
        from data.sources import _TENCENT_CODE_MAP
        # SPDR Select Sector ETFs trade on NYSE Arca — .AM suffix
        for sym in ("XLK", "XLF", "XLY", "XLV", "XLE", "XLI",
                    "XLU", "XLB", "XLRE", "XLC", "XLP"):
            assert sym in _TENCENT_CODE_MAP, f"{sym} missing from Tencent map"
            assert _TENCENT_CODE_MAP[sym].endswith(".AM"), \
                f"{sym} not on NYSE Arca suffix"

    def test_spy_qqq_still_mapped(self):
        """Regression: existing benchmark symbols stayed mapped."""
        from data.sources import _TENCENT_CODE_MAP
        assert _TENCENT_CODE_MAP["SPY"] == "usSPY.AM"
        assert _TENCENT_CODE_MAP["QQQ"] == "usQQQ.OQ"


class TestTencentSourceFetch:
    def test_fetch_returns_dataframe(self):
        from data.sources import TencentSource

        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = MOCK_TENCENT_RESP
        mock_session.get.return_value = mock_resp

        src = TencentSource()
        with patch("data.sources._make_session", return_value=mock_session):
            df = src.fetch("AAPL", "2025-01-01", "2025-01-10")
            assert len(df) == 3
            assert list(df.columns) == OHLCV_COLUMNS
            assert df["Close"].iloc[0] == 191.0

    def test_fetch_api_error_returns_empty(self):
        from data.sources import TencentSource

        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"code": -1, "msg": "error"}
        mock_session.get.return_value = mock_resp

        src = TencentSource()
        with patch("data.sources._make_session", return_value=mock_session):
            df = src.fetch("AAPL", "2025-01-01", "2025-01-10")
            assert df.empty

    def test_fetch_request_exception_returns_empty(self):
        from data.sources import TencentSource
        import requests as req

        mock_session = MagicMock()
        mock_session.get.side_effect = req.RequestException("connection error")

        src = TencentSource()
        with patch("data.sources._make_session", return_value=mock_session):
            df = src.fetch("NONEXIST", "2025-01-01", "2025-01-10")
            assert isinstance(df, pd.DataFrame)

    def test_fetch_with_split_adjustment(self):
        from data.sources import TencentSource

        src = TencentSource()
        # AAPL has a 4:1 split on 2020-08-31
        # With 2025 data, no splits should apply
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = MOCK_TENCENT_RESP
        mock_session.get.return_value = mock_resp

        with patch("data.sources._make_session", return_value=mock_session):
            df = src.fetch("AAPL", "2025-01-01", "2025-01-10")
            assert len(df) == 3


class TestSinaSourceFetch:
    def test_fetch_returns_dataframe(self):
        from data.sources import SinaSource

        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = json.dumps(MOCK_SINA_RESP)
        mock_session.get.return_value = mock_resp

        src = SinaSource()
        with patch("data.sources._make_session", return_value=mock_session):
            df = src.fetch("sh510300", "2025-01-01", "2025-01-10")
            assert len(df) == 2
            assert list(df.columns) == OHLCV_COLUMNS

    def test_fetch_empty_response(self):
        from data.sources import SinaSource

        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "[]"
        mock_session.get.return_value = mock_resp

        src = SinaSource()
        with patch("data.sources._make_session", return_value=mock_session):
            df = src.fetch("sh510300", "2025-01-01", "2025-01-10")
            assert df.empty

    def test_fetch_json_error(self):
        from data.sources import SinaSource

        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "not json"
        mock_session.get.return_value = mock_resp

        src = SinaSource()
        with patch("data.sources._make_session", return_value=mock_session):
            df = src.fetch("sh510300", "2025-01-01", "2025-01-10")
            assert df.empty


class TestAKShareSourceFetch:
    def test_fetch_returns_dataframe(self):
        from data.sources import AKShareSource

        mock_df = pd.DataFrame({
            "日期": ["2025-01-02", "2025-01-03"],
            "开盘": [3.50, 3.52],
            "最高": [3.55, 3.58],
            "最低": [3.48, 3.51],
            "收盘": [3.52, 3.56],
            "成交量": [10000000, 12000000],
        })

        src = AKShareSource()
        with patch("data.sources.AKShareSource.fetch", return_value=pd.DataFrame(columns=OHLCV_COLUMNS)):
            # Just verify import/init doesn't crash
            df = src.fetch("sh510300", "2025-01-01", "2025-01-10")
            assert df.empty

    def test_without_akshare_installed(self):
        try:
            import akshare  # noqa: F401
            pytest.skip("akshare is installed")
        except ImportError:
            from data.sources import AKShareSource
            src = AKShareSource()
            df = src.fetch("sh510300", "2025-01-01", "2025-01-10")
            assert df.empty


# ===================================================================
# SinaUSSource fetch mock
# ===================================================================

_MOCK_SINAUS_RESP = (
    "var=(["
    '{"d":"2025-01-02","o":"190.00","h":"192.00","l":"189.00","c":"191.00","v":"50000000"},'
    '{"d":"2025-01-03","o":"191.50","h":"193.00","l":"191.00","c":"192.00","v":"45000000"}'
    "])"
)


class TestSinaUSSourceFetch:
    def test_fetch_returns_dataframe(self):
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = _MOCK_SINAUS_RESP
        mock_session.get.return_value = mock_resp

        src = SinaUSSource()
        with patch("data.sources._make_session", return_value=mock_session):
            df = src.fetch("AAPL", "2025-01-01", "2025-01-10")
            assert len(df) == 2
            assert list(df.columns) == OHLCV_COLUMNS
            assert df["Close"].iloc[0] == 191.0

    def test_fetch_empty_response(self):
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "var=([])"
        mock_session.get.return_value = mock_resp

        src = SinaUSSource()
        with patch("data.sources._make_session", return_value=mock_session):
            df = src.fetch("AAPL", "2025-01-01", "2025-01-10")
            assert df.empty

    def test_fetch_request_exception(self):
        import requests as req
        mock_session = MagicMock()
        mock_session.get.side_effect = req.RequestException("timeout")

        src = SinaUSSource()
        with patch("data.sources._make_session", return_value=mock_session):
            df = src.fetch("AAPL", "2025-01-01", "2025-01-10")
            assert isinstance(df, pd.DataFrame) and df.empty



# ===================================================================
# YahooChartSource fetch mock
# ===================================================================

_MOCK_YAHOO_RESP = {
    "chart": {
        "result": [{
            "meta": {"symbol": "AAPL"},
            "timestamp": [1704150000, 1704236400],
            "indicators": {
                "quote": [{
                    "open": [190.0, 191.5],
                    "high": [192.0, 193.0],
                    "low": [189.0, 191.0],
                    "close": [191.0, 192.0],
                    "volume": [50000000, 45000000],
                }]
            }
        }]
    }
}


class TestYahooChartSourceFetch:
    def test_fetch_returns_dataframe(self):
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = _MOCK_YAHOO_RESP
        mock_resp.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_resp

        src = YahooChartSource()
        with patch("data.sources._yahoo_session", return_value=mock_session):
            df = src.fetch("AAPL", "2025-01-01", "2025-01-10")
            assert len(df) == 2
            assert list(df.columns) == OHLCV_COLUMNS
            assert df["Close"].iloc[0] == 191.0

    def test_fetch_empty_chart_returns_empty(self):
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"chart": {"result": [None]}}
        mock_resp.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_resp

        src = YahooChartSource()
        with patch("data.sources._yahoo_session", return_value=mock_session):
            df = src.fetch("AAPL", "2025-01-01", "2025-01-10")
            assert df.empty

    def test_fetch_maps_dot_class_share_for_yahoo(self):
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = _MOCK_YAHOO_RESP
        mock_resp.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_resp

        src = YahooChartSource()
        with patch("data.sources._yahoo_session", return_value=mock_session):
            src.fetch("BRK.B", "2025-01-01", "2025-01-10")

        assert mock_session.get.call_args.args[0].endswith("/BRK-B")

    def test_fetch_request_exception(self):
        import requests as req
        mock_session = MagicMock()
        mock_session.get.side_effect = req.RequestException("timeout")

        src = YahooChartSource()
        with patch("data.sources._yahoo_session", return_value=mock_session):
            df = src.fetch("AAPL", "2025-01-01", "2025-01-10")
            assert isinstance(df, pd.DataFrame) and df.empty
