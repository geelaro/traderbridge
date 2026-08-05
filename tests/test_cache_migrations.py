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

# Pre-v1 shapes: ohlcv_daily had no `source` column and ops_log had no
# `source`/`level` columns — migration v1's ALTER TABLE statements exist
# precisely to add them. Building a legacy DB from the *current* _SCHEMA_DDL
# would already contain those columns, making the ALTERs silent no-ops and
# the upgrade test vacuous.
_LEGACY_OHLCV = """CREATE TABLE ohlcv_daily (
    symbol   TEXT    NOT NULL,
    date     TEXT    NOT NULL,
    open     REAL,
    high     REAL,
    low      REAL,
    close    REAL,
    volume   INTEGER,
    PRIMARY KEY (symbol, date)
)"""

_LEGACY_OPS_LOG = """CREATE TABLE ops_log (
    ts         TEXT DEFAULT (datetime('now','localtime')),
    event      TEXT,
    symbol     TEXT,
    detail     TEXT,
    value      REAL
)"""


def _legacy_schema() -> list:
    """Current DDL but with ohlcv_daily/ops_log replaced by their pre-v1
    column shapes — the rest of the base tables are unchanged."""
    out = []
    for stmt in _SCHEMA_DDL:
        if stmt.startswith("CREATE TABLE IF NOT EXISTS ohlcv_daily"):
            out.append(_LEGACY_OHLCV)
        elif stmt.startswith("CREATE TABLE IF NOT EXISTS ops_log"):
            out.append(_LEGACY_OPS_LOG)
        else:
            out.append(stmt)
    return out


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
    # init_schema enables WAL — clean up the -wal/-shm sidecars too,
    # otherwise they leak on Windows after the main db is unlinked.
    for suffix in ("", "-wal", "-shm"):
        try:
            os.unlink(path + suffix)
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


class TestUpgradeFromTruePreV1Schema:
    """The real upgrade path the v1 migration exists for: a DB whose
    ohlcv_daily/ops_log predate the `source`/`level` columns, so the
    ALTER TABLE statements actually execute (unlike TestUpgradeFrom
    PreMigrationSchema, where current-shape DDL makes them no-ops)."""

    def _create_legacy_db(self, path):
        conn = sqlite3.connect(path)
        for stmt in _legacy_schema():
            conn.execute(stmt)
        conn.execute(
            "INSERT INTO ohlcv_daily (symbol, date, open, high, low, close, volume) "
            "VALUES ('AAPL', '2025-01-02', 100, 101, 99, 100.5, 1000000)"
        )
        conn.execute(
            "INSERT INTO ops_log (event, symbol, detail, value) "
            "VALUES ('pre-v1-event', 'AAA', 'legacy', 1)"
        )
        conn.commit()
        conn.close()

    def test_alter_migrations_actually_run(self, db_path):
        self._create_legacy_db(db_path)

        cache = CacheManager(db_path=db_path)
        cache.init_schema()

        assert _schema_version(cache.conn) == 4
        assert _column_exists(cache.conn, "ohlcv_daily", "source")
        assert _column_exists(cache.conn, "ops_log", "source")
        assert _column_exists(cache.conn, "ops_log", "level")
        cache.close()

    def test_legacy_rows_survive_with_alter_defaults(self, db_path):
        """Existing rows get the ALTER's DEFAULT value — a real old DB's
        history must not come back with NULL source/level."""
        self._create_legacy_db(db_path)

        cache = CacheManager(db_path=db_path)
        cache.init_schema()

        row = cache.conn.execute(
            "SELECT source, level FROM ops_log"
        ).fetchone()
        assert row == ("live_trader", "INFO")
        ohlcv = cache.conn.execute(
            "SELECT symbol, close, source FROM ohlcv_daily"
        ).fetchone()
        assert ohlcv == ("AAPL", 100.5, "")
        cache.close()
