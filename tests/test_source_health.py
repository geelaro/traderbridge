"""Tests for data/source_health.py and its wiring into DataProvider."""

import json
import os
import tempfile
from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

from data import provider as provider_module
from data import DataProvider
from data.protocol import DataSource
from data.source_health import (
    check_splits_changed,
    is_in_cooldown,
    record_fetch_result,
    source_health_report,
)


# ===================================================================
# record_fetch_result / is_in_cooldown
# ===================================================================


class TestRecordAndCooldown:
    def test_not_in_cooldown_before_any_failure(self, temp_cache):
        assert is_in_cooldown(temp_cache, "sina_us") is False

    def test_in_cooldown_immediately_after_failure(self, temp_cache):
        record_fetch_result(temp_cache, "sina_us", success=False, detail="timeout")
        assert is_in_cooldown(temp_cache, "sina_us") is True

    def test_cooldown_window_of_zero_minutes_never_blocks(self, temp_cache):
        record_fetch_result(temp_cache, "sina_us", success=False, detail="timeout")
        assert is_in_cooldown(temp_cache, "sina_us", cooldown_minutes=0) is False

    def test_success_does_not_clear_prior_cooldown(self, temp_cache):
        """Cooldown is time-based, not cleared by the next success — a
        source that failed 1 second ago and then succeeded once is still
        within its cooldown window (see data/source_health.py docstring:
        record_fetch_result on success never wipes last_failure_ts)."""
        record_fetch_result(temp_cache, "sina_us", success=False, detail="timeout")
        record_fetch_result(temp_cache, "sina_us", success=True)
        assert is_in_cooldown(temp_cache, "sina_us") is True

    def test_success_only_never_triggers_cooldown(self, temp_cache):
        record_fetch_result(temp_cache, "sina_us", success=True)
        assert is_in_cooldown(temp_cache, "sina_us") is False

    def test_unknown_source_not_in_cooldown(self, temp_cache):
        assert is_in_cooldown(temp_cache, "nonexistent_source") is False


# ===================================================================
# source_health_report
# ===================================================================


class TestSourceHealthReport:
    def test_empty_when_no_records(self, temp_cache):
        report = source_health_report(temp_cache)
        assert report["sources"] == []

    def test_aggregates_success_and_failure(self, temp_cache):
        record_fetch_result(temp_cache, "sina_us", success=True)
        record_fetch_result(temp_cache, "sina_us", success=True)
        record_fetch_result(temp_cache, "sina_us", success=False, detail="429")

        report = source_health_report(temp_cache)
        row = next(s for s in report["sources"] if s["source"] == "sina_us")
        assert row["success_count"] == 2
        assert row["failure_count"] == 1
        assert row["success_rate"] == pytest.approx(2 / 3)
        assert row["last_failure_detail"] == "429"
        assert row["in_cooldown"] is True

    def test_multiple_sources_kept_separate(self, temp_cache):
        record_fetch_result(temp_cache, "sina_us", success=True)
        record_fetch_result(temp_cache, "tencent", success=False, detail="timeout")

        report = source_health_report(temp_cache)
        by_source = {s["source"]: s for s in report["sources"]}
        assert by_source["sina_us"]["in_cooldown"] is False
        assert by_source["tencent"]["in_cooldown"] is True


# ===================================================================
# check_splits_changed
# ===================================================================


@pytest.fixture
def splits_file():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump({"AAPL": 1}, f)
        path = f.name
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


class TestCheckSplitsChanged:
    def test_first_check_is_silent_baseline(self, temp_cache, splits_file):
        assert check_splits_changed(temp_cache, splits_file) is None

    def test_unchanged_stays_silent(self, temp_cache, splits_file):
        check_splits_changed(temp_cache, splits_file)
        assert check_splits_changed(temp_cache, splits_file) is None

    def test_change_triggers_one_time_message(self, temp_cache, splits_file):
        check_splits_changed(temp_cache, splits_file)  # baseline
        with open(splits_file, "w") as f:
            json.dump({"AAPL": 2}, f)

        msg = check_splits_changed(temp_cache, splits_file)
        assert msg is not None
        assert "force_refresh" in msg

        # Second call after the change is acknowledged — silent again.
        assert check_splits_changed(temp_cache, splits_file) is None

    def test_missing_file_returns_none(self, temp_cache):
        assert check_splits_changed(temp_cache, "no/such/file.json") is None


# ===================================================================
# DataProvider integration — cooldown skip + best-effort isolation
# ===================================================================


class _FlakySource(DataSource):
    """Always returns empty — used to seed a real cooldown."""

    def __init__(self, name):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def supports(self, symbol: str) -> bool:
        return True

    def fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        return pd.DataFrame()


class _GoodSource(DataSource):
    def __init__(self, name):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def supports(self, symbol: str) -> bool:
        return True

    def fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        dates = pd.bdate_range("2025-01-01", periods=30)
        df = pd.DataFrame({
            "Open": 100, "High": 105, "Low": 95, "Close": 102, "Volume": 1_000_000,
        }, index=dates)
        df.index.name = "date"
        return df


class TestProviderCooldownIntegration:
    def test_source_in_cooldown_is_skipped(self, temp_cache):
        """Pre-seed a recent failure for 'sina_us' (first in the 'us'
        priority list) — the provider must skip straight past it rather
        than retrying, and fall through to the next priority source."""
        record_fetch_result(temp_cache, "sina_us", success=False, detail="pre-seeded")
        good = _GoodSource("tencent")
        provider = DataProvider(cache=temp_cache, sources=[_FlakySource("sina_us"), good])

        df = provider.get_daily("AAPL", start="2025-01-01", end="2025-01-15")
        assert not df.empty  # got data from tencent, sina_us was skipped

    def test_cooldown_persists_across_provider_instances(self, temp_cache):
        """Unlike the old in-memory _failed_sources set, this must survive
        a brand-new DataProvider() — the whole point of persisting it."""
        record_fetch_result(temp_cache, "sina_us", success=False, detail="pre-seeded")

        flaky_calls = []
        flaky = _FlakySource("sina_us")
        original_fetch = flaky.fetch
        flaky.fetch = lambda *a, **kw: flaky_calls.append(1) or original_fetch(*a, **kw)

        provider = DataProvider(cache=temp_cache, sources=[flaky, _GoodSource("tencent")])
        provider.get_daily("AAPL", start="2025-01-01", end="2025-01-15")
        assert flaky_calls == []  # never even attempted — skipped via cooldown

    def test_force_refresh_bypasses_cooldown(self, temp_cache):
        """A deliberate force_refresh must not be silently swallowed by an
        unrelated recent failure's cooldown window — the whole point of
        asking for a forced refresh is to retry right now regardless."""
        record_fetch_result(temp_cache, "sina_us", success=False, detail="pre-seeded")
        good = _GoodSource("sina_us")
        provider = DataProvider(cache=temp_cache, sources=[good])

        df = provider.get_daily("AAPL", start="2025-01-01", end="2025-01-15",
                                 force_refresh=True)
        assert not df.empty  # sina_us was actually retried despite cooldown


class TestProviderHealthTrackingBestEffort:
    def test_cache_write_failure_does_not_break_fetch(self, temp_cache, monkeypatch):
        """The critical safety property from the plan: a bug in health
        recording must never take down the actual data-fetch path."""
        def _boom(*args, **kwargs):
            raise RuntimeError("cache is on fire")

        monkeypatch.setattr(provider_module, "record_fetch_result", _boom)
        monkeypatch.setattr(provider_module, "is_in_cooldown", _boom)

        good = _GoodSource("tencent")
        provider = DataProvider(cache=temp_cache, sources=[good])
        df = provider.get_daily("AAPL", start="2025-01-01", end="2025-01-15")
        assert not df.empty

    def test_successful_fetch_is_recorded(self, temp_cache):
        good = _GoodSource("tencent")
        provider = DataProvider(cache=temp_cache, sources=[good])
        provider.get_daily("AAPL", start="2025-01-01", end="2025-01-15")

        report = source_health_report(temp_cache)
        row = next(s for s in report["sources"] if s["source"] == "tencent")
        assert row["success_count"] >= 1

    def test_empty_result_is_recorded_as_failure(self, temp_cache):
        flaky = _FlakySource("tencent")
        provider = DataProvider(cache=temp_cache, sources=[flaky])
        provider.get_daily("AAPL", start="2025-01-01", end="2025-01-15")

        report = source_health_report(temp_cache)
        row = next(s for s in report["sources"] if s["source"] == "tencent")
        assert row["failure_count"] >= 1
