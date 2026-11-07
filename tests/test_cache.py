"""Tests for data/cache.py — CacheManager."""

import sqlite3

import pytest
from data.cache import CacheManager


class TestCacheSchema:
    def test_init_creates_tables(self, temp_cache):
        temp_cache.init_schema()
        tables = temp_cache.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = {r[0] for r in tables}
        assert "ohlcv_daily" in names
        assert "signal_history" in names

    def test_init_is_idempotent(self, temp_cache):
        temp_cache.init_schema()
        temp_cache.init_schema()  # should not raise
        tables = temp_cache.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        assert len([r for r in tables if r[0] == "ohlcv_daily"]) == 1


class TestSaveLoadRoundtrip:
    def test_save_and_load(self, temp_cache, ohlcv):
        temp_cache.save("TEST", ohlcv, source="test")
        loaded = temp_cache.load("TEST")
        assert len(loaded) == len(ohlcv)
        assert list(loaded.columns) == ["Open", "High", "Low", "Close", "Volume"]
        # Values should match within float tolerance
        assert abs(loaded.iloc[0]["Close"] - ohlcv.iloc[0]["Close"]) < 0.01

    def test_load_with_date_filter(self, temp_cache, ohlcv):
        temp_cache.save("TEST", ohlcv, source="test")
        start = ohlcv.index[50].strftime("%Y-%m-%d")
        end = ohlcv.index[99].strftime("%Y-%m-%d")
        subset = temp_cache.load("TEST", start=start, end=end)
        assert len(subset) == 50
        assert subset.index[0] >= pd.Timestamp(start)
        assert subset.index[-1] <= pd.Timestamp(end)

    def test_load_empty_symbol(self, temp_cache):
        df = temp_cache.load("NOEXIST")
        assert df.empty
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_save_upsert(self, temp_cache, ohlcv):
        """Saving same symbol twice should not duplicate rows."""
        temp_cache.save("TEST", ohlcv.iloc[:100], source="a")
        temp_cache.save("TEST", ohlcv.iloc[:100], source="b")
        loaded = temp_cache.load("TEST")
        assert len(loaded) == 100  # not 200


class TestDateRange:
    def test_empty_cache(self, temp_cache):
        s, e = temp_cache.date_range("NOEXIST")
        assert s is None
        assert e is None

    def test_has_data(self, temp_cache, ohlcv):
        temp_cache.save("TEST", ohlcv)
        s, e = temp_cache.date_range("TEST")
        assert s is not None
        assert e is not None


class TestMissingRanges:
    def test_all_missing(self, temp_cache):
        gaps = temp_cache.missing_ranges("X", "2020-01-01", "2020-06-01")
        assert len(gaps) == 1
        assert gaps[0] == ("2020-01-01", "2020-06-01")

    def test_partial_gap(self, temp_cache, ohlcv):
        temp_cache.save("TEST", ohlcv)  # approx 2020-01-01 → 2021-03
        # Request earlier data
        gaps = temp_cache.missing_ranges("TEST", "2019-06-01", "2021-06-01")
        # Should have a gap before and after cached range
        assert len(gaps) >= 1

    def test_internal_gap_detected(self, temp_cache, ohlcv):
        df = ohlcv.iloc[:20].drop(ohlcv.index[5:10])
        temp_cache.save("TEST", df)
        gaps = temp_cache.missing_ranges("TEST", str(ohlcv.index[0].date()), str(ohlcv.index[19].date()))
        assert any(start <= str(ohlcv.index[5].date()) <= end for start, end in gaps)


class TestSignalHistory:
    def test_save_signal(self, temp_cache):
        temp_cache.save_signal("2025-01-15", "AAPL", "weekly_macd",
                               "2025-01-15", 1, 195.0, 5.0, '{"MACD":1.5}')
        rows = temp_cache.query_signals()
        assert len(rows) == 1
        assert rows[0]["symbol"] == "AAPL"
        assert rows[0]["signal"] == 1

    def test_save_signal_upsert(self, temp_cache):
        """Second save for same scan_date+symbol+strategy replaces first."""
        temp_cache.save_signal("2025-01-15", "AAPL", "w", "2025-01-15", 1, 1, 1, "")
        temp_cache.save_signal("2025-01-15", "AAPL", "w", "2025-01-15", -1, 2, 2, "")
        rows = temp_cache.query_signals()
        assert len(rows) == 1
        assert rows[0]["signal"] == -1

    def test_query_by_date(self, temp_cache):
        temp_cache.save_signal("2025-01-10", "A", "s", "2025-01-10", 1, 1, 1, "")
        temp_cache.save_signal("2025-01-15", "B", "s", "2025-01-15", -1, 2, 2, "")
        rows = temp_cache.query_signals(scan_date="2025-01-13")
        assert len(rows) == 1
        assert rows[0]["symbol"] == "B"

    def test_query_by_symbol(self, temp_cache):
        temp_cache.save_signal("2025-01-10", "AAPL", "s", "2025-01-10", 1, 1, 1, "")
        temp_cache.save_signal("2025-01-10", "NVDA", "s", "2025-01-10", -1, 2, 2, "")
        rows = temp_cache.query_signals(symbol="AAPL")
        assert len(rows) == 1
        assert rows[0]["symbol"] == "AAPL"


class TestRiskState:
    def test_save_and_load(self, temp_cache):
        temp_cache.save_risk_state("consecutive_losses", "2")
        val = temp_cache.load_risk_state("consecutive_losses")
        assert val == "2"

    def test_overwrite(self, temp_cache):
        temp_cache.save_risk_state("daily_trade_count", "3")
        temp_cache.save_risk_state("daily_trade_count", "5")
        val = temp_cache.load_risk_state("daily_trade_count")
        assert val == "5"

    def test_nonexistent_key(self, temp_cache):
        val = temp_cache.load_risk_state("no_such_key")
        assert val is None

    def test_multiple_keys(self, temp_cache):
        temp_cache.save_risk_state("date", "2025-06-01")
        temp_cache.save_risk_state("consecutive_losses", "1")
        temp_cache.save_risk_state("daily_trade_count", "4")
        assert temp_cache.load_risk_state("date") == "2025-06-01"
        assert temp_cache.load_risk_state("consecutive_losses") == "1"
        assert temp_cache.load_risk_state("daily_trade_count") == "4"

    def test_save_many_is_atomic_on_failure(self, temp_cache):
        temp_cache.init_schema()
        temp_cache.conn.execute(
            """CREATE TRIGGER reject_risk_state_insert
               BEFORE INSERT ON risk_state WHEN NEW.key = 'reject'
               BEGIN SELECT RAISE(ABORT, 'forced failure'); END"""
        )

        with pytest.raises(sqlite3.IntegrityError):
            temp_cache.save_risk_state_many({"first": "1", "reject": "x"})

        assert temp_cache.load_risk_state("first") is None

    def test_save_many_roundtrip(self, temp_cache):
        temp_cache.save_risk_state_many({"date": "2026-11-07", "count": "3"})

        assert temp_cache.load_risk_state("date") == "2026-11-07"
        assert temp_cache.load_risk_state("count") == "3"


class TestAlertHistory:
    def test_record_and_load_single(self, temp_cache):
        temp_cache.record_alert(
            "risk_light",
            {"level": "red", "reasons": ["VIX > 30"]},
        )
        rows = temp_cache.load_alert_history(days=1)
        assert len(rows) == 1
        assert rows[0]["alert_type"] == "risk_light"
        assert rows[0]["payload"]["level"] == "red"
        assert rows[0]["payload"]["reasons"] == ["VIX > 30"]

    def test_newest_first_ordering(self, temp_cache):
        temp_cache.record_alert("a", {"i": 1}, ts="2026-05-01T10:00:00")
        temp_cache.record_alert("b", {"i": 2}, ts="2026-05-31T10:00:00")
        temp_cache.record_alert("c", {"i": 3}, ts="2026-05-15T10:00:00")
        rows = temp_cache.load_alert_history(days=365)
        # DESC by ts: 5-31, 5-15, 5-01
        assert [r["payload"]["i"] for r in rows] == [2, 3, 1]

    def test_filter_by_alert_type(self, temp_cache):
        temp_cache.record_alert("risk_light", {"x": 1})
        temp_cache.record_alert("vix_spike", {"x": 2})
        temp_cache.record_alert("risk_light", {"x": 3})

        only_rl = temp_cache.load_alert_history(days=1, alert_type="risk_light")
        assert len(only_rl) == 2
        assert all(r["alert_type"] == "risk_light" for r in only_rl)

    def test_days_filter_excludes_old(self, temp_cache):
        from datetime import datetime, timedelta
        old_ts = (datetime.now() - timedelta(days=60)).isoformat(timespec="seconds")
        recent_ts = (datetime.now() - timedelta(days=2)).isoformat(timespec="seconds")
        temp_cache.record_alert("a", {"i": "old"}, ts=old_ts)
        temp_cache.record_alert("a", {"i": "recent"}, ts=recent_ts)

        rows = temp_cache.load_alert_history(days=30)
        assert len(rows) == 1
        assert rows[0]["payload"]["i"] == "recent"

    def test_empty_history_returns_empty_list(self, temp_cache):
        assert temp_cache.load_alert_history(days=30) == []

    def test_unicode_payload_round_trip(self, temp_cache):
        temp_cache.record_alert("vix_spike", {"原因": "极端恐慌", "value": 35.0})
        rows = temp_cache.load_alert_history(days=1)
        assert rows[0]["payload"]["原因"] == "极端恐慌"
        assert rows[0]["payload"]["value"] == 35.0


class TestEntryPrices:
    def test_save_and_load_single(self, temp_cache):
        temp_cache.save_entry_price("AAPL", 195.0, "2025-06-01")
        result = temp_cache.load_entry_price("AAPL")
        assert result is not None
        assert result[0] == 195.0
        assert result[1] == "2025-06-01"

    def test_load_all(self, temp_cache):
        temp_cache.save_entry_price("AAPL", 195.0, "2025-06-01")
        temp_cache.save_entry_price("NVDA", 850.0, "2025-06-02")
        temp_cache.save_entry_price("TSLA", 240.0, "2025-06-01")
        all_prices = temp_cache.load_all_entry_prices()
        assert len(all_prices) == 3
        assert all_prices["AAPL"] == (195.0, "2025-06-01")
        assert all_prices["NVDA"] == (850.0, "2025-06-02")

    def test_delete(self, temp_cache):
        temp_cache.save_entry_price("AAPL", 195.0, "2025-06-01")
        temp_cache.delete_entry_price("AAPL")
        result = temp_cache.load_entry_price("AAPL")
        assert result is None

    def test_nonexistent(self, temp_cache):
        result = temp_cache.load_entry_price("NOEXIST")
        assert result is None

    def test_load_all_empty(self, temp_cache):
        all_prices = temp_cache.load_all_entry_prices()
        assert all_prices == {}

    def test_upsert(self, temp_cache):
        temp_cache.save_entry_price("AAPL", 195.0, "2025-06-01")
        temp_cache.save_entry_price("AAPL", 200.0, "2025-06-05")
        result = temp_cache.load_entry_price("AAPL")
        assert result[0] == 200.0
        assert result[1] == "2025-06-05"
        # Only one row in DB
        all_prices = temp_cache.load_all_entry_prices()
        assert len(all_prices) == 1


class TestSchemaNewTables:
    def test_risk_state_table_exists(self, temp_cache):
        temp_cache.init_schema()
        tables = temp_cache.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = {r[0] for r in tables}
        assert "risk_state" in names

    def test_entry_prices_table_exists(self, temp_cache):
        temp_cache.init_schema()
        tables = temp_cache.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = {r[0] for r in tables}
        assert "entry_prices" in names


class TestBatchMode:
    def test_batch_defers_commit(self, temp_cache):
        temp_cache.enable_batch()
        temp_cache.save_risk_state("test_key", "42")
        # In batch mode, data should not be visible to another connection
        import sqlite3
        c2 = sqlite3.connect(str(temp_cache.db_path))
        val = c2.execute("SELECT value FROM risk_state WHERE key='test_key'").fetchone()
        c2.close()
        assert val is None  # not committed yet

    def test_commit_batch_flushes(self, temp_cache):
        temp_cache.enable_batch()
        temp_cache.save_risk_state("test_key", "43")
        temp_cache.commit_batch()
        # After commit, data visible to another connection
        import sqlite3
        c2 = sqlite3.connect(str(temp_cache.db_path))
        val = c2.execute("SELECT value FROM risk_state WHERE key='test_key'").fetchone()
        c2.close()
        assert val == ("43",)

    def test_normal_mode_commits_immediately(self, temp_cache):
        temp_cache.save_risk_state("test_key", "44")
        import sqlite3
        c2 = sqlite3.connect(str(temp_cache.db_path))
        val = c2.execute("SELECT value FROM risk_state WHERE key='test_key'").fetchone()
        c2.close()
        assert val == ("44",)


class TestOpsLog:
    def test_log_ops_writes(self, temp_cache):
        temp_cache.log_ops("trading_paused", detail="test pause")
        rows = temp_cache.conn.execute(
            "SELECT event, detail FROM ops_log WHERE event='trading_paused'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][1] == "test pause"

    def test_log_ops_all_event_types(self, temp_cache):
        events = ["trading_paused", "slippage", "slippage_rejected",
                   "gate_reject", "risk_reject"]
        for e in events:
            temp_cache.log_ops(e, symbol="TEST", detail=e, value=1.0)
        row_count = temp_cache.conn.execute("SELECT COUNT(*) FROM ops_log").fetchone()[0]
        assert row_count == len(events)
        # Verify each event type
        for e in events:
            r = temp_cache.conn.execute(
                "SELECT symbol, detail, value FROM ops_log WHERE event=?", [e]
            ).fetchone()
            assert r is not None
            assert r[0] == "TEST"

    def test_log_ops_idempotent(self, temp_cache):
        """Calling log_ops twice should insert two rows (no PK conflict)."""
        temp_cache.log_ops("test_event", symbol="A")
        temp_cache.log_ops("test_event", symbol="B")
        rows = temp_cache.conn.execute(
            "SELECT symbol FROM ops_log WHERE event='test_event' ORDER BY ts"
        ).fetchall()
        assert len(rows) == 2


# Need pd for asserts
import pandas as pd
