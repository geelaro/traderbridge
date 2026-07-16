"""SQLite-based local storage — OHLCV cache, state persistence, ops logging.

Classes
-------
- ``OhlcvCache`` — OHLCV load / save / date_range / missing_ranges
- ``StateStore`` — risk_state, entry_prices, trade_pnl persistence
- ``OpsLogger`` — ops_log, order_log, slippage_log writes
- ``CacheManager`` — full backward-compat facade (all three + signal_history)
"""

from __future__ import annotations

import os
import sqlite3
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from pandas import Timestamp

from .protocol import OHLCV_COLUMNS

# --- ops_log source constants -----------------------------------------
OPS_SRC_LIVE = "live_trader"
OPS_SRC_DASHBOARD = "dashboard"
OPS_SRC_BROKER = "broker"


# ======================================================================
# Schema DDL — individual statements for safe execution
# ======================================================================

_SCHEMA_DDL = [
    """CREATE TABLE IF NOT EXISTS ohlcv_daily (
        symbol   TEXT    NOT NULL,
        date     TEXT    NOT NULL,
        open     REAL,
        high     REAL,
        low      REAL,
        close    REAL,
        volume   INTEGER,
        source   TEXT,
        PRIMARY KEY (symbol, date)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol ON ohlcv_daily(symbol)",
    "CREATE INDEX IF NOT EXISTS idx_ohlcv_date  ON ohlcv_daily(date)",
    """CREATE TABLE IF NOT EXISTS signal_history (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        scan_date TEXT    NOT NULL,
        symbol    TEXT    NOT NULL,
        strategy  TEXT    NOT NULL,
        bar_date  TEXT    NOT NULL,
        signal    INTEGER NOT NULL,
        price     REAL,
        atr       REAL,
        indicators TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS idx_signal_scan ON signal_history(scan_date)",
    "CREATE INDEX IF NOT EXISTS idx_signal_sym  ON signal_history(symbol)",
    """CREATE TABLE IF NOT EXISTS risk_state (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS entry_prices (
        symbol     TEXT PRIMARY KEY,
        price      REAL NOT NULL,
        entry_date TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS order_log (
        order_id   TEXT,
        symbol     TEXT,
        side       TEXT,
        qty        INTEGER,
        price      REAL,
        status     TEXT,
        created_at TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS slippage_log (
        order_id      TEXT,
        symbol        TEXT,
        side          TEXT,
        signal_price  REAL,
        fill_price    REAL,
        slippage_pct  REAL,
        created_at    TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS trade_pnl (
        symbol      TEXT,
        side        TEXT,
        qty         INTEGER,
        entry_price REAL,
        exit_price  REAL,
        pnl         REAL,
        pnl_pct     REAL,
        exit_date   TEXT,
        order_id    TEXT PRIMARY KEY
    )""",
    """CREATE TABLE IF NOT EXISTS ops_log (
        ts         TEXT DEFAULT (datetime('now','localtime')),
        source     TEXT DEFAULT 'live_trader',
        level      TEXT DEFAULT 'INFO',
        event      TEXT,
        symbol     TEXT,
        detail     TEXT,
        value      REAL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_ops_event    ON ops_log(event)",
    "CREATE INDEX IF NOT EXISTS idx_ops_ts       ON ops_log(ts)",
    """CREATE TABLE IF NOT EXISTS alert_history (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        ts         TEXT NOT NULL,
        alert_type TEXT NOT NULL,
        payload    TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_alert_ts   ON alert_history(ts)",
    "CREATE INDEX IF NOT EXISTS idx_alert_type ON alert_history(alert_type)",
    """CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY
    )""",
]


# ======================================================================
# Shared base
# ======================================================================


class _CacheBase:
    """Shared connection lifecycle, schema init, commit control.

    All non-read methods are guarded by a ``threading.RLock`` so a single
    SQLite connection is safe for multi-threaded access.
    """

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            # Resolution order: TRADERBRIDGE_DB (new) → MYTRADER_DB (legacy
            # for pre-rename .env files / docker-compose) → default file.
            db_path = (
                os.environ.get("TRADERBRIDGE_DB")
                or os.environ.get("MYTRADER_DB")
                or "trading_data.db"
            )
        self.db_path = Path(db_path).resolve()
        self._conn: Optional[sqlite3.Connection] = None
        self._batch_mode = False
        self._lock = threading.RLock()

    # -- connection ---------------------------------------------------------

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        return self._conn

    def close(self):
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def _commit(self):
        if not self._batch_mode:
            self.conn.commit()

    def enable_batch(self):
        """Defer all commits — caller must call commit_batch() once."""
        with self._lock:
            self._batch_mode = True

    def commit_batch(self):
        """Flush all deferred writes in one commit."""
        with self._lock:
            self.conn.commit()
            self._batch_mode = False

    # -- schema -------------------------------------------------------------

    def init_schema(self):
        """Create all tables, indexes, and run migrations."""
        with self._lock:
            # PRAGMAs must run before any table DDL
            try:
                self.conn.execute("PRAGMA journal_mode = WAL")
            except sqlite3.OperationalError:
                pass
            try:
                self.conn.execute("PRAGMA synchronous = NORMAL")
            except sqlite3.OperationalError:
                pass
            for stmt in _SCHEMA_DDL:
                self.conn.execute(stmt)
        self._run_migrations()
        self._commit()

    def _run_migrations(self):
        """Versioned schema migrations. Called from init_schema (lock held)."""
        cur = self.conn.execute(
            "SELECT MAX(version) FROM (SELECT 0 AS version UNION ALL SELECT MAX(version) FROM schema_version)"
        )
        row = cur.fetchone()
        cur.close()
        current = row[0] if row else 0

        migrations = [
            (1, [
                "ALTER TABLE ohlcv_daily ADD COLUMN source TEXT DEFAULT ''",
                "ALTER TABLE ops_log ADD COLUMN source TEXT DEFAULT 'live_trader'",
                "ALTER TABLE ops_log ADD COLUMN level TEXT DEFAULT 'INFO'",
            ]),
            (2, [
                "CREATE INDEX IF NOT EXISTS idx_ops_src_ts ON ops_log(source, ts)",
            ]),
            (3, [
                """CREATE TABLE IF NOT EXISTS decision_history (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts                TEXT NOT NULL,
                    decision_type     TEXT NOT NULL,
                    symbol            TEXT,
                    reason            TEXT,
                    risk_light        TEXT,
                    vix               REAL,
                    portfolio_value   REAL,
                    num_positions     INTEGER,
                    concentration_hhi REAL,
                    effective_bets    REAL,
                    var_95            REAL,
                    drawdown_pct      REAL,
                    payload           TEXT
                )""",
                "CREATE INDEX IF NOT EXISTS idx_decision_ts   ON decision_history(ts)",
                "CREATE INDEX IF NOT EXISTS idx_decision_type ON decision_history(decision_type)",
                "CREATE INDEX IF NOT EXISTS idx_decision_sym  ON decision_history(symbol)",
            ]),
            (4, [
                """CREATE TABLE IF NOT EXISTS source_health (
                    source              TEXT NOT NULL,
                    date                TEXT NOT NULL,
                    success_count       INTEGER DEFAULT 0,
                    failure_count       INTEGER DEFAULT 0,
                    last_failure_ts     TEXT,
                    last_failure_detail TEXT,
                    PRIMARY KEY (source, date)
                )""",
                "CREATE INDEX IF NOT EXISTS idx_source_health_date ON source_health(date)",
            ]),
        ]

        for version, statements in migrations:
            if version <= current:
                continue
            for stmt in statements:
                try:
                    self.conn.execute(stmt)
                except sqlite3.OperationalError:
                    pass
            self.conn.execute(
                "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                [version],
            )



# ======================================================================
# OhlcvCache
# ======================================================================


class OhlcvCache(_CacheBase):
    """OHLCV daily cache with incremental-update helpers."""

    # -- load ---------------------------------------------------------------

    def load(
        self, symbol: str, start: Optional[str] = None, end: Optional[str] = None
    ) -> pd.DataFrame:
        """Return cached bars for *symbol*, or empty DataFrame."""
        with self._lock:
            self.init_schema()
            query = "SELECT date, open, high, low, close, volume FROM ohlcv_daily WHERE symbol = ?"
            params = [symbol.upper()]

            if start:
                query += " AND date >= ?"
                params.append(str(pd.Timestamp(start).date()))
            if end:
                query += " AND date <= ?"
                params.append(str(pd.Timestamp(end).date()))

            query += " ORDER BY date ASC"
            df = pd.read_sql_query(
                query, self.conn, params=params, index_col="date", parse_dates=["date"]
            )
        if df.empty:
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        }, inplace=True)

        for col in OHLCV_COLUMNS:
            if col not in df.columns:
                df[col] = 0
        return df[OHLCV_COLUMNS]

    def date_range(self, symbol: str) -> Tuple[Optional[str], Optional[str]]:
        """Return (earliest_date, latest_date) cached for *symbol*."""
        with self._lock:
            self.init_schema()
            cur = self.conn.execute(
                "SELECT MIN(date), MAX(date) FROM ohlcv_daily WHERE symbol = ?",
                [symbol.upper()],
            )
            row = cur.fetchone()
            cur.close()
        if row is None or row[0] is None:
            return None, None
        return row[0], row[1]

    # -- save ---------------------------------------------------------------

    def save(self, symbol: str, df: pd.DataFrame, source: str = "") -> int:
        """Insert or replace bars. Returns number of rows written."""
        with self._lock:
            if df is None or df.empty:
                return 0
            return self._save(symbol, df, source)

    def _save(self, symbol: str, df: pd.DataFrame, source: str = "") -> int:
        self.init_schema()
        df = df.copy()

        if df.index.name != "date" and "date" not in df.columns:
            raise ValueError("DataFrame must have a 'date' index or column")
        if "date" not in df.columns:
            df["date"] = df.index.map(lambda x: str(pd.Timestamp(x).date()))

        df["symbol"] = symbol.upper()
        df["source"] = source
        df["date"] = df["date"].map(lambda x: str(pd.Timestamp(x).date()))

        col_map = {
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        }
        db_df = pd.DataFrame()
        db_df["symbol"] = df["symbol"]
        db_df["date"] = df["date"]
        for cap, low in col_map.items():
            db_df[low] = df[cap] if cap in df.columns else 0
        db_df["source"] = df["source"]

        dates = [str(d) for d in db_df["date"].unique()]
        if not dates:
            return 0
        try:
            if self._batch_mode:
                self.conn.execute("SAVEPOINT save_ohlcv")
            else:
                self.conn.execute("BEGIN IMMEDIATE")
            self.conn.execute(
                f"DELETE FROM ohlcv_daily WHERE symbol = ? AND date IN ({','.join('?' * len(dates))})",
                [symbol.upper()] + dates,
            )
            # SQLite caps a single statement at SQLITE_MAX_VARIABLE_NUMBER
            # (default 999). With 8 columns, chunksize 100 → 800 vars/INSERT,
            # comfortably under the limit and fast enough for 10k-row writes.
            db_df.to_sql(
                "ohlcv_daily", self.conn, if_exists="append", index=False,
                method="multi", chunksize=100,
            )
            if self._batch_mode:
                self.conn.execute("RELEASE SAVEPOINT save_ohlcv")
            self._commit()
        except Exception:
            if self._batch_mode:
                self.conn.execute("ROLLBACK TO SAVEPOINT save_ohlcv")
            else:
                self.conn.rollback()
            raise
        return len(db_df)

    # -- incremental helpers ------------------------------------------------

    def missing_ranges(
        self, symbol: str, start: str, end: str
    ) -> List[Tuple[str, str]]:
        """Return merged (start_date, end_date) tuples that need fetching.

        Compares the requested interval against what's already cached.
        Adjacent gaps (separated by ≤ 7 calendar days) are merged into a
        single larger range to minimise round-trips.
        """
        req_start_ts = pd.Timestamp(start).normalize()
        req_end_ts = pd.Timestamp(end).normalize()
        req_start = str(req_start_ts.date())
        req_end = str(req_end_ts.date())

        with self._lock:
            self.init_schema()
            rows = self.conn.execute(
                """
                SELECT date FROM ohlcv_daily
                WHERE symbol = ? AND date >= ? AND date <= ?
                ORDER BY date ASC
                """,
                [symbol.upper(), req_start, req_end],
            ).fetchall()
        cached_dates = [pd.Timestamp(row[0]).normalize() for row in rows]

        if not cached_dates:
            return [(req_start, req_end)]

        gaps: List[Tuple[str, str]] = []
        first = cached_dates[0]
        if req_start_ts < first:
            gap_end = first - pd.Timedelta(days=1)
            if req_start_ts <= gap_end:
                gaps.append((str(req_start_ts.date()), str(gap_end.date())))

        prev = first
        for current in cached_dates[1:]:
            # 7-day threshold tolerates long-weekend / holiday clusters
            # (Fri → following Wed = 5 days, Christmas-NYE gap = 6 days).
            # Anything beyond that is a real missing window worth re-fetching.
            if (current - prev).days > 7:
                gap_start = prev + pd.Timedelta(days=1)
                gap_end = current - pd.Timedelta(days=1)
                if gap_start <= gap_end:
                    gaps.append((str(gap_start.date()), str(gap_end.date())))
            prev = current

        last = cached_dates[-1]
        if req_end_ts > last:
            gap_start = last + pd.Timedelta(days=1)
            if gap_start <= req_end_ts:
                gaps.append((str(gap_start.date()), str(req_end_ts.date())))

        if len(gaps) <= 1:
            return gaps

        merged: List[Tuple[str, str]] = []
        cur_start, cur_end = gaps[0]
        for gs, ge in gaps[1:]:
            gs_ts = pd.Timestamp(gs)
            cur_end_ts = pd.Timestamp(cur_end)
            if (gs_ts - cur_end_ts).days <= 8:
                cur_end = ge
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = gs, ge
        merged.append((cur_start, cur_end))
        return merged


# ======================================================================
# StateStore
# ======================================================================


class StateStore(_CacheBase):
    """Risk state, entry prices, and trade PnL persistence."""

    # -- risk state ---------------------------------------------------------

    def save_risk_state(self, key: str, value: str):
        with self._lock:
            self._save_risk_state(key, value)

    def _save_risk_state(self, key: str, value: str):
        self.init_schema()
        self.conn.execute(
            "INSERT OR REPLACE INTO risk_state (key, value) VALUES (?, ?)",
            [key, value],
        )
        self._commit()

    def load_risk_state(self, key: str) -> Optional[str]:
        with self._lock:
            self.init_schema()
            row = self.conn.execute(
                "SELECT value FROM risk_state WHERE key = ?", [key]
            ).fetchone()
        return row[0] if row else None

    # -- source health --------------------------------------------------------

    def record_source_health(self, source: str, success: bool, detail: str = "") -> None:
        """Upsert today's (source, date) row — accumulate success/failure
        counts. On failure, stamps last_failure_ts/detail; a success never
        clears a prior failure record (COALESCE keeps the old value)."""
        with self._lock:
            self.init_schema()
            today = pd.Timestamp.now().strftime("%Y-%m-%d")
            fail_ts = None if success else pd.Timestamp.now().isoformat()
            self.conn.execute(
                """INSERT INTO source_health
                       (source, date, success_count, failure_count, last_failure_ts, last_failure_detail)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(source, date) DO UPDATE SET
                       success_count = success_count + excluded.success_count,
                       failure_count = failure_count + excluded.failure_count,
                       last_failure_ts = COALESCE(excluded.last_failure_ts, last_failure_ts),
                       last_failure_detail = COALESCE(excluded.last_failure_detail, last_failure_detail)
                """,
                [
                    source, today,
                    1 if success else 0,
                    0 if success else 1,
                    fail_ts,
                    None if success else detail,
                ],
            )
            self._commit()

    def get_last_source_failure(self, source: str) -> Optional[str]:
        """Most recent failure timestamp (ISO string) for *source*, across
        all history — or None if it has never failed / has no record."""
        with self._lock:
            self.init_schema()
            row = self.conn.execute(
                "SELECT MAX(last_failure_ts) FROM source_health WHERE source = ?",
                [source],
            ).fetchone()
        return row[0] if row and row[0] else None

    def get_source_health(self, days: int = 7) -> list:
        """Per-source success/failure counts over the last *days* days, plus
        each source's most recent failure (unbounded by the window — a
        cooldown-relevant failure from outside the window still matters)."""
        with self._lock:
            self.init_schema()
            since = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
            rows = self.conn.execute(
                """SELECT sh.source,
                          SUM(sh.success_count) AS success_count,
                          SUM(sh.failure_count) AS failure_count,
                          (SELECT MAX(sh2.last_failure_ts) FROM source_health sh2
                               WHERE sh2.source = sh.source) AS last_failure_ts,
                          (SELECT sh3.last_failure_detail FROM source_health sh3
                               WHERE sh3.source = sh.source AND sh3.last_failure_ts = (
                                   SELECT MAX(sh4.last_failure_ts) FROM source_health sh4
                                       WHERE sh4.source = sh.source
                               ) LIMIT 1) AS last_failure_detail
                   FROM source_health sh
                   WHERE sh.date >= ?
                   GROUP BY sh.source
                   ORDER BY sh.source
                """,
                [since],
            ).fetchall()
        return [
            {
                "source": r[0], "success_count": r[1] or 0, "failure_count": r[2] or 0,
                "last_failure_ts": r[3], "last_failure_detail": r[4],
            }
            for r in rows
        ]

    # -- alert history ------------------------------------------------------

    def record_alert(self, alert_type: str, payload: dict,
                     ts: Optional[str] = None) -> None:
        """Append one alert event to the history log.

        Parameters
        ----------
        alert_type : str
            Caller-defined category ("risk_light" / "vix_spike" / "position_stop").
        payload : dict
            Structured detail (reasons, indicators, symbol …); JSON-encoded.
        ts : str, optional
            ISO timestamp.  Defaults to ``datetime.now().isoformat()``.
        """
        import json as _json
        if ts is None:
            ts = datetime.now().isoformat(timespec="seconds")
        with self._lock:
            self.init_schema()
            self.conn.execute(
                "INSERT INTO alert_history (ts, alert_type, payload) VALUES (?, ?, ?)",
                [ts, alert_type, _json.dumps(payload, ensure_ascii=False)],
            )
            self._commit()

    def load_alert_history(
        self,
        days: int = 30,
        alert_type: Optional[str] = None,
    ) -> list:
        """Return alerts within the last ``days`` (newest first).

        Each row is a dict ``{"ts": str, "alert_type": str, "payload": dict}``.
        Pass ``alert_type`` to restrict to one category.
        """
        import json as _json
        cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
        sql = "SELECT ts, alert_type, payload FROM alert_history WHERE ts >= ?"
        args = [cutoff]
        if alert_type:
            sql += " AND alert_type = ?"
            args.append(alert_type)
        sql += " ORDER BY ts DESC"
        with self._lock:
            self.init_schema()
            rows = self.conn.execute(sql, args).fetchall()
        out = []
        for ts, atype, payload_str in rows:
            try:
                payload = _json.loads(payload_str)
            except (ValueError, TypeError):
                payload = {"_raw": payload_str}
            out.append({"ts": ts, "alert_type": atype, "payload": payload})
        return out

    # -- decision history ---------------------------------------------------

    def record_decision(
        self,
        decision_type: str,
        symbol: Optional[str] = None,
        reason: Optional[str] = None,
        context: Optional[dict] = None,
        payload: Optional[dict] = None,
        ts: Optional[str] = None,
    ) -> int:
        """Persist one decision moment with current risk snapshot.

        Parameters
        ----------
        decision_type : str
            Caller category. Suggested: 'trade_buy' / 'trade_sell' /
            'kill_switch' / 'rebalance' / 'manual_override' /
            'signal_ignored'.
        symbol : str, optional
            Affected symbol. None for portfolio-wide events.
        reason : str, optional
            Free-text explanation persisted alongside the row.
        context : dict, optional
            Risk snapshot — known keys are columns; unknown keys go into
            payload. Recognised: ``risk_light, vix, portfolio_value,
            num_positions, concentration_hhi, effective_bets, var_95,
            drawdown_pct``.
        payload : dict, optional
            Action-specific structured data (order details, what-if
            params, etc.) — JSON-encoded into the payload column.
        ts : str, optional
            ISO timestamp. Defaults to now.

        Returns
        -------
        int : the rowid of the inserted record (lets caller link a later
        outcome update back to this decision).
        """
        import json as _json
        if ts is None:
            ts = datetime.now().isoformat(timespec="seconds")
        ctx = dict(context or {})
        # Pull recognised fields out of ctx; any leftovers merge into payload.
        cols = {
            "risk_light":        ctx.pop("risk_light", None),
            "vix":               ctx.pop("vix", None),
            "portfolio_value":   ctx.pop("portfolio_value", None),
            "num_positions":     ctx.pop("num_positions", None),
            "concentration_hhi": ctx.pop("concentration_hhi", None),
            "effective_bets":    ctx.pop("effective_bets", None),
            "var_95":            ctx.pop("var_95", None),
            "drawdown_pct":      ctx.pop("drawdown_pct", None),
        }
        merged_payload = dict(payload or {})
        if ctx:  # leftovers from context go into the payload JSON
            merged_payload["_extra_context"] = ctx

        with self._lock:
            self.init_schema()
            cur = self.conn.execute(
                """INSERT INTO decision_history (
                    ts, decision_type, symbol, reason,
                    risk_light, vix, portfolio_value, num_positions,
                    concentration_hhi, effective_bets, var_95, drawdown_pct,
                    payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    ts, decision_type, symbol, reason,
                    cols["risk_light"], cols["vix"],
                    cols["portfolio_value"], cols["num_positions"],
                    cols["concentration_hhi"], cols["effective_bets"],
                    cols["var_95"], cols["drawdown_pct"],
                    _json.dumps(merged_payload, ensure_ascii=False)
                    if merged_payload else None,
                ],
            )
            row_id = cur.lastrowid
            self._commit()
        return int(row_id) if row_id else 0

    def query_decisions(
        self,
        days: int = 90,
        decision_type: Optional[str] = None,
        symbol: Optional[str] = None,
        risk_light: Optional[str] = None,
        limit: int = 1000,
    ) -> list:
        """Return decisions within the last ``days`` (newest first).

        Each row is a dict with all columns + parsed ``payload``.
        Filters compose with AND; pass None to skip a filter.
        """
        import json as _json
        cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
        sql = """SELECT id, ts, decision_type, symbol, reason,
                        risk_light, vix, portfolio_value, num_positions,
                        concentration_hhi, effective_bets, var_95, drawdown_pct,
                        payload
                 FROM decision_history WHERE ts >= ?"""
        args = [cutoff]
        if decision_type:
            sql += " AND decision_type = ?"
            args.append(decision_type)
        if symbol:
            sql += " AND symbol = ?"
            args.append(symbol.upper())
        if risk_light:
            sql += " AND risk_light = ?"
            args.append(risk_light)
        sql += " ORDER BY ts DESC LIMIT ?"
        args.append(int(limit))

        with self._lock:
            self.init_schema()
            rows = self.conn.execute(sql, args).fetchall()

        cols = ("id", "ts", "decision_type", "symbol", "reason",
                "risk_light", "vix", "portfolio_value", "num_positions",
                "concentration_hhi", "effective_bets", "var_95", "drawdown_pct",
                "payload")
        out = []
        for row in rows:
            d = dict(zip(cols, row))
            if d.get("payload"):
                try:
                    d["payload"] = _json.loads(d["payload"])
                except (ValueError, TypeError):
                    d["payload"] = {"_raw": d["payload"]}
            else:
                d["payload"] = {}
            out.append(d)
        return out

    # -- entry prices -------------------------------------------------------

    def save_entry_price(self, symbol: str, price: float, entry_date: str):
        with self._lock:
            self._save_entry_price(symbol, price, entry_date)

    def _save_entry_price(self, symbol: str, price: float, entry_date: str):
        self.init_schema()
        self.conn.execute(
            "INSERT OR REPLACE INTO entry_prices (symbol, price, entry_date) VALUES (?, ?, ?)",
            [symbol.upper(), price, entry_date],
        )
        self._commit()

    def load_entry_price(self, symbol: str) -> Optional[tuple]:
        with self._lock:
            self.init_schema()
            row = self.conn.execute(
                "SELECT price, entry_date FROM entry_prices WHERE symbol = ?",
                [symbol.upper()],
            ).fetchone()
        return (row[0], row[1]) if row else None

    def load_all_entry_prices(self) -> dict:
        with self._lock:
            self.init_schema()
            rows = self.conn.execute(
                "SELECT symbol, price, entry_date FROM entry_prices"
            ).fetchall()
        return {row[0]: (row[1], row[2]) for row in rows}

    def delete_entry_price(self, symbol: str):
        with self._lock:
            self._delete_entry_price(symbol)

    def _delete_entry_price(self, symbol: str):
        self.init_schema()
        self.conn.execute(
            "DELETE FROM entry_prices WHERE symbol = ?", [symbol.upper()]
        )
        self._commit()

    # -- trade PnL ----------------------------------------------------------

    def save_trade_pnl(self, symbol: str, side: str, qty: int,
                       entry_price: float, exit_price: float,
                       exit_date: str, order_id: str):
        """Record a completed round-trip trade PnL."""
        with self._lock:
            self._save_trade_pnl(symbol, side, qty, entry_price, exit_price, exit_date, order_id)

    def _save_trade_pnl(self, symbol: str, side: str, qty: int,
                        entry_price: float, exit_price: float,
                        exit_date: str, order_id: str):
        pnl = (exit_price - entry_price) * qty
        pnl_pct = (exit_price / entry_price - 1) * 100
        self.init_schema()
        self.conn.execute(
            "INSERT OR REPLACE INTO trade_pnl VALUES (?,?,?,?,?,?,?,?,?)",
            [symbol.upper(), side, qty, entry_price, exit_price,
             round(pnl, 2), round(pnl_pct, 2), exit_date, order_id],
        )
        self._commit()

    def query_trade_pnl(self, symbol: str = None, limit: int = 50) -> list[dict]:
        """Return recent trade PnL records."""
        query = "SELECT * FROM trade_pnl"
        params = []
        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol.upper())
        query += " ORDER BY exit_date DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            self.init_schema()
            rows = self.conn.execute(query, params).fetchall()
        return [dict(zip(["symbol","side","qty","entry_price","exit_price",
                         "pnl","pnl_pct","exit_date","order_id"], row)) for row in rows]


# ======================================================================
# OpsLogger
# ======================================================================


class OpsLogger(_CacheBase):
    """Operational event logging — ops_log, order_log, slippage_log."""

    def log_ops(self, event: str, symbol: str = "", detail: str = "",
                value: float = 0, level: str = "INFO", source: str = OPS_SRC_LIVE):
        """Record an operational event (pause, rejection, slippage, etc.)."""
        with self._lock:
            self.init_schema()
            self.conn.execute(
                "INSERT INTO ops_log (source, level, event, symbol, detail, value) VALUES (?,?,?,?,?,?)",
                [source, level, event, symbol, detail, value],
            )
            self._commit()


# ======================================================================
# CacheManager — backward-compat facade
# ======================================================================


class CacheManager(OhlcvCache, StateStore, OpsLogger):
    """Full combined cache — all three specialized stores + signal history.

    This is the backward-compatible class. New code can reach for the
    specialised classes directly if only a subset is needed.

    ── Signal history (only present in the combined facade) ───────────
    """

    # -- signal history -----------------------------------------------------

    def save_signal(
        self, scan_date: str, symbol: str, strategy: str,
        bar_date: str, signal: int, price: float, atr: float,
        indicators: str = "",
    ):
        with self._lock:
            self._save_signal(scan_date, symbol, strategy, bar_date, signal, price, atr, indicators)

    def _save_signal(self, scan_date, symbol, strategy, bar_date, signal, price, atr, indicators=""):
        self.init_schema()
        self.conn.execute(
            "DELETE FROM signal_history WHERE scan_date = ? AND symbol = ? AND strategy = ?",
            [scan_date, symbol.upper(), strategy],
        )
        self.conn.execute(
            "INSERT INTO signal_history (scan_date, symbol, strategy, bar_date, signal, price, atr, indicators) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [scan_date, symbol.upper(), strategy, bar_date, signal, price, atr, indicators],
        )
        self._commit()

    def query_signals(self, scan_date: str | None = None, symbol: str | None = None) -> list[dict]:
        query = "SELECT * FROM signal_history WHERE 1=1"
        params = []
        if scan_date:
            query += " AND scan_date >= ?"
            params.append(scan_date)
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol.upper())
        query += " ORDER BY scan_date DESC, symbol ASC"
        with self._lock:
            self.init_schema()
            rows = self.conn.execute(query, params).fetchall()
        return [dict(zip(["id", "scan_date", "symbol", "strategy", "bar_date",
                          "signal", "price", "atr", "indicators"], row)) for row in rows]
