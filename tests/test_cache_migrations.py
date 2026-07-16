"""Tests for data/cache.py's versioned migration system (_run_migrations).

Never tested before this pass — the most valuable scenario is "upgrade an
existing production DB", which is exactly what migrations exist for but
nobody had verified end-to-end.
"""

import sqlite3
import tempfile
import os

import pytest

from data.cache import CacheManager, _SCHEMA_DDL


def _table_exists(conn, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", [name]
    ).fetchone()
    return row is not None


def _column_exists(conn, table: str, column: str) -> bool:
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    return column in cols


def _schema_version(conn) -> int:
    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    return row[0] if row and row[0] is not None else 0


@pytest.fixture
def db_path():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


class TestFreshDatabase:
    def test_reaches_latest_schema_version(self, db_path):
        cache = CacheManager(db_path=db_path)
        cache.init_schema()
        assert _schema_version(cache.conn) == 4
        cache.close()

    def test_all_migrated_tables_exist(self, db_path):
        cache = CacheManager(db_path=db_path)
        cache.init_schema()
        for table in ("decision_history", "source_health"):
            assert _table_exists(cache.conn, table), f"{table} missing after fresh init"
        for col in ("source", "level"):
            assert _column_exists(cache.conn, "ops_log", col)
        cache.close()

    def test_init_schema_is_idempotent(self, db_path):
        cache = CacheManager(db_path=db_path)
        cache.init_schema()
        v1 = _schema_version(cache.conn)
        cache.init_schema()  # second call must not error or double-apply
        v2 = _schema_version(cache.conn)
        assert v1 == v2 == 4
        cache.close()


class TestUpgradeFromPreMigrationSchema:
    """Simulates a real old production DB: only the base _SCHEMA_DDL tables
    exist (no schema_version row, no migrations ever run), with real data
    already in it. Opening it through CacheManager must upgrade it in
    place without losing that data."""

    def _create_pre_migration_db(self, path):
        conn = sqlite3.connect(path)
        for stmt in _SCHEMA_DDL:
            conn.execute(stmt)
        # Seed data that must survive the upgrade.
        conn.execute(
            "INSERT INTO ohlcv_daily (symbol, date, open, high, low, close, volume, source) "
            "VALUES ('AAPL', '2025-01-02', 100, 101, 99, 100.5, 1000000, 'legacy')"
        )
        conn.commit()
        conn.close()

    def test_upgrade_adds_new_tables_and_columns(self, db_path):
        self._create_pre_migration_db(db_path)

        cache = CacheManager(db_path=db_path)
        cache.init_schema()

        assert _schema_version(cache.conn) == 4
        assert _table_exists(cache.conn, "decision_history")
        assert _table_exists(cache.conn, "source_health")
        assert _column_exists(cache.conn, "ops_log", "source")
        assert _column_exists(cache.conn, "ops_log", "level")
        cache.close()

    def test_upgrade_preserves_existing_data(self, db_path):
        self._create_pre_migration_db(db_path)

        cache = CacheManager(db_path=db_path)
        cache.init_schema()

        row = cache.conn.execute(
            "SELECT symbol, close, source FROM ohlcv_daily WHERE symbol='AAPL'"
        ).fetchone()
        assert row == ("AAPL", 100.5, "legacy")
        cache.close()

    def test_upgraded_db_supports_new_methods(self, db_path):
        """Not just schema shape — the new StateStore methods (added for
        Ops-1) must actually work against an upgraded (not fresh) DB."""
        self._create_pre_migration_db(db_path)

        cache = CacheManager(db_path=db_path)
        cache.record_source_health("tencent", success=True)
        report = cache.get_source_health(days=7)
        assert any(r["source"] == "tencent" for r in report)
        cache.close()
