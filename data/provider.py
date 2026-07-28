"""Unified DataProvider — gateway for all market-data requests.

Responsibilities
----------------
1. Route a symbol to the correct data source(s).
2. Check local cache; identify gaps.
3. Fetch only missing date ranges from external sources (fallback chain).
4. Merge + cache + return a complete DataFrame.
"""

import logging
from datetime import date, datetime
from typing import List, Optional, Tuple

import pandas as pd

from .cache import CacheManager
from .protocol import (
    OHLCV_COLUMNS,
    SOURCE_PRIORITY,
    DataSource,
    classify_symbol,
    CN_SYMBOLS,
)
from .sources import (
    TencentSource,
    SinaSource,
    SinaUSSource,
    YahooChartSource,
    AKShareSource,
    CBOEVixSource,
)
from .source_health import is_in_cooldown, record_fetch_result

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trading-calendar helpers (used for gap detection)
# ---------------------------------------------------------------------------

try:
    import exchange_calendars as xcals
    _HAS_XCALS = True
except ImportError:
    xcals = None  # type: ignore[assignment]
    _HAS_XCALS = False

_CALENDARS: dict = {}

def _get_calendar(market: str):
    """Lazy-load exchange calendar by market tag ('us'|'cn'|'hk').

    Falls back to None quietly — callers use weekday heuristic instead.
    """
    if not _HAS_XCALS:
        return None
    if market not in _CALENDARS:
        code = {"us": "XNYS", "cn": "XSHG", "hk": "XHKG"}.get(market, "XNYS")
        try:
            _CALENDARS[market] = xcals.get_calendar(code)
        except Exception:
            logger.debug("Could not load calendar %s for market %s", code, market)
            _CALENDARS[market] = None
    return _CALENDARS[market]

# ---------------------------------------------------------------------------
# DataProvider
# ---------------------------------------------------------------------------


class DataProvider:
    """Unified market-data access point.

    .. code-block:: python

        from data import DataProvider

        dp = DataProvider()
        df = dp.get_daily("AAPL",  start="2022-01-01", end="2024-12-31")
        df = dp.get_daily("510300", start="2022-01-01", end="2024-12-31")
    """

    def __init__(
        self,
        cache: Optional[CacheManager] = None,
        sources: Optional[List[DataSource]] = None,
    ):
        self.cache = cache or CacheManager()
        self._sources: List[DataSource] = sources or [
            CBOEVixSource(),  # ^VIX / VIX only
            SinaUSSource(),
            TencentSource(),
            SinaSource(),
            YahooChartSource(),
            AKShareSource(),
        ]
        self._fetch_failures: set = set()  # symbols that failed this session entirely
        self._tail_attempted: set = set()  # (symbol, end) pairs already attempted
        # Deduplicate cross-source drift warnings — only warn once per
        # (symbol, source) pair per session.
        self._cross_source_warned: set = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_daily(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Return daily OHLCV for *symbol*.

        On first call, fetches from external sources and caches locally.
        Subsequent calls return cached data (incremental updates).

        Parameters
        ----------
        symbol : str
            Ticker (e.g. "AAPL", "510300", "sh510300").
        start : str | None
        end : str | None
        force_refresh : bool
            If True, skip cache and re-fetch everything.
        """
        sym = symbol.upper().strip()
        # Resolve CN aliases
        if sym in CN_SYMBOLS:
            sym = CN_SYMBOLS[sym].upper()
        market = classify_symbol(sym)

        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        # If no start, try to return whatever the cache has
        if start is None:
            cached_start, _ = self.cache.date_range(sym)
            start = cached_start or "2015-01-01"

        if not force_refresh:
            df = self._load_from_cache(sym, start, end)
            if self._is_complete(df, start, end, market):
                return df
            # Tail-incomplete but no internal gaps → incremental fetch only
            if not df.empty and not self._has_internal_gaps(df):
                tail_start = (df.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                dedup_key = (sym, end)
                if tail_start <= end and dedup_key not in self._tail_attempted:
                    self._tail_attempted.add(dedup_key)
                    fetched, src = self._fetch_from_sources(sym, tail_start, end)
                    if fetched is not None and not fetched.empty:
                        self._check_cross_source_drift(sym, fetched, src)
                        self.cache.save(sym, fetched, source=src or "unknown")
                return self._load_from_cache(sym, start, end)

        # Full gap scan (force_refresh, internal gaps, or first fetch)
        if force_refresh and sym in self._fetch_failures:
            self._fetch_failures.discard(sym)
        if sym in self._fetch_failures:
            return self._load_from_cache(sym, start, end)

        gaps = self._find_gaps(sym, start, end, force_refresh)
        any_fetched = False
        for gap_start, gap_end in gaps:
            fetched, actual_source = self._fetch_from_sources(
                sym, gap_start, gap_end, bypass_cooldown=force_refresh,
            )
            if fetched is not None and not fetched.empty:
                self._check_cross_source_drift(sym, fetched, actual_source)
                self.cache.save(sym, fetched, source=actual_source or "unknown")
                any_fetched = True
        if gaps and not any_fetched:
            self._fetch_failures.add(sym)

        return self._load_from_cache(sym, start, end)

    def get_data(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        freqs: Optional[List[str]] = None,
        force_refresh: bool = False,
    ) -> dict:
        """Return OHLCV for *symbol* at requested frequencies.

        Parameters
        ----------
        freqs : list
            e.g. ``["D"]`` or ``["D", "W"]``.  Defaults to ``["D"]``.

        Returns
        -------
        dict[str, pd.DataFrame]
            ``{"D": df_daily, "W": df_weekly}`` — weekly is resampled from
            daily (Friday close, cached internally).
        """
        if freqs is None:
            freqs = ["D"]
        result = {}
        if "D" in freqs or "W" in freqs:
            df_daily = self.get_daily(symbol, start=start, end=end, force_refresh=force_refresh)
            result["D"] = df_daily
        if "W" in freqs:
            result["W"] = self._resample_weekly(df_daily) if "D" in result and not result["D"].empty else pd.DataFrame()
        return result

    @staticmethod
    def _resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
        """Resample daily OHLCV to weekly (Friday close)."""
        if df is None or df.empty:
            return pd.DataFrame()
        return (
            df.resample("W-FRI")
            .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
            .dropna()
        )

    def cached_range(self, symbol: str) -> Tuple[Optional[str], Optional[str]]:
        """Return (earliest, latest) cached dates for *symbol*."""
        return self.cache.date_range(symbol.upper())

    def list_sources(self, symbol: str) -> List[str]:
        """Return ordered list of source names that can serve *symbol*."""
        market = classify_symbol(symbol)
        sources = SOURCE_PRIORITY.get(market, SOURCE_PRIORITY["default"])
        return sources

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_from_cache(
        self, symbol: str, start: str, end: str
    ) -> pd.DataFrame:
        return self.cache.load(symbol, start, end)

    def _find_gaps(
        self, symbol: str, start: str, end: str, force: bool
    ) -> List[Tuple[str, str]]:
        """Determine date ranges that need fetching."""
        if force:
            return [(start, end)]
        return self.cache.missing_ranges(symbol, start, end)

    def _fetch_from_sources(
        self, symbol: str, start: str, end: str, bypass_cooldown: bool = False,
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """Try sources in priority order; return (df, actual_source_name).

        Sources that fail are recorded in the persisted source_health table
        (see data/source_health.py) and skipped while in cooldown — a
        transient failure degrades fetches for ~15 minutes, not for the
        rest of the process's life. (Previously this used an in-memory
        _failed_sources set with no expiry; combined with dashboard/main.py's
        @st.cache_resource singleton DataProvider, one empty result for any
        symbol could permanently blacklist a source for the whole Streamlit
        session. See ROADMAP.md.)

        ``bypass_cooldown`` is set only for an explicit caller-requested
        ``force_refresh`` — a deliberate manual retry should not be silently
        swallowed by an unrelated recent failure's cooldown window. Ordinary
        gap-fill/incremental fetches still respect cooldown normally.
        """
        market = classify_symbol(symbol)
        priorities = SOURCE_PRIORITY.get(market, SOURCE_PRIORITY["default"])

        for source_name in priorities:
            if not bypass_cooldown and self._is_source_cooling_down(source_name):
                continue
            src = self._find_source_by_name(source_name)
            if src is None or not src.supports(symbol):
                continue
            logger.info("Fetching %s from %s (%s → %s)", symbol, source_name, start, end)
            try:
                df = src.fetch(symbol, start, end)
                if df is not None and not df.empty:
                    logger.info("  → got %d bars from %s", len(df), source_name)
                    self._record_health(source_name, success=True)
                    return df, source_name
                self._record_health(source_name, success=False, detail="empty result")
            except (ConnectionError, TimeoutError, OSError, ValueError, KeyError) as exc:
                logger.warning("Source %s failed for %s", source_name, symbol)
                self._record_health(source_name, success=False, detail=str(exc)[:200])
            except Exception as exc:
                logger.exception("Source %s unexpected error for %s", source_name, symbol)
                self._record_health(source_name, success=False, detail=str(exc)[:200])
        return pd.DataFrame(columns=OHLCV_COLUMNS), None

    def _is_source_cooling_down(self, source_name: str) -> bool:
        """Persisted, time-window cooldown check — see data/source_health.py.

        Best-effort: a health-store read failure must never block an actual
        data fetch, so any exception here is treated as "not cooling down".
        """
        try:
            return is_in_cooldown(self.cache, source_name)
        except Exception:
            logger.debug("source_health cooldown check failed for %s", source_name, exc_info=True)
            return False

    def _record_health(self, source_name: str, success: bool, detail: str = "") -> None:
        """Best-effort health record — must never break the fetch path."""
        try:
            record_fetch_result(self.cache, source_name, success=success, detail=detail)
        except Exception:
            logger.debug("source_health record failed for %s", source_name, exc_info=True)

    def _find_source_by_name(self, name: str) -> Optional[DataSource]:
        for src in self._sources:
            if src.name == name:
                return src
        return None

    def _check_cross_source_drift(
        self, symbol: str, fetched: pd.DataFrame, source: Optional[str],
        tolerance: float = 0.01,
    ) -> None:
        """Warn when *fetched* close prices disagree with the existing cache.

        Detects silent provider switches that change splits / adjustments
        midstream. Compares overlapping dates and the cached bar immediately
        adjacent to the fetched range (boundary drift catches split-adjust
        differences between sources). Logs a warning at most once per
        (symbol, source) per process — never blocks the write.
        """
        if fetched is None or fetched.empty or source is None:
            return
        key = (symbol.upper(), source)
        if key in self._cross_source_warned:
            return

        try:
            f_start = str(pd.Timestamp(fetched.index[0]).date())
            f_end = str(pd.Timestamp(fetched.index[-1]).date())
            # Pull cache covering 5 bars before/after the fetched window
            pad_start = (pd.Timestamp(f_start) - pd.Timedelta(days=14)).date().isoformat()
            pad_end = (pd.Timestamp(f_end) + pd.Timedelta(days=14)).date().isoformat()
            cached = self.cache.load(symbol, pad_start, pad_end)
        except Exception:
            return
        if cached is None or cached.empty:
            return

        # 1) Same-date overlap
        common = fetched.index.normalize().intersection(cached.index.normalize())
        max_overlap_drift = 0.0
        for d in common[:5]:  # sample up to 5 overlapping bars
            try:
                fc = float(fetched.loc[fetched.index.normalize() == d, "Close"].iloc[0])
                cc = float(cached.loc[cached.index.normalize() == d, "Close"].iloc[0])
                if cc > 0:
                    drift = abs(fc - cc) / cc
                    if drift > max_overlap_drift:
                        max_overlap_drift = drift
            except (IndexError, KeyError):
                continue

        # 2) Boundary check — last cached bar before fetch vs first fetched bar
        boundary_drift = 0.0
        before = cached[cached.index < pd.Timestamp(f_start)]
        if not before.empty:
            try:
                prev_close = float(before["Close"].iloc[-1])
                first_close = float(fetched["Close"].iloc[0])
                if prev_close > 0:
                    # Compare relative move; >20% single-bar jump is suspicious
                    boundary_drift = abs(first_close - prev_close) / prev_close
            except (IndexError, KeyError):
                pass

        if max_overlap_drift > tolerance:
            logger.warning(
                "cross-source drift: %s from %s differs %.2f%% on %d overlapping bars; "
                "possible split/adjustment divergence",
                symbol, source, max_overlap_drift * 100, len(common),
            )
            self._cross_source_warned.add(key)
        elif boundary_drift > 0.20:
            logger.warning(
                "cross-source boundary jump: %s from %s — %.1f%% move vs prior cached bar; "
                "verify split/adjustment alignment",
                symbol, source, boundary_drift * 100,
            )
            self._cross_source_warned.add(key)

    @staticmethod
    def _has_internal_gaps(df: pd.DataFrame) -> bool:
        """Check for gaps > 7 days between consecutive cached bars."""
        if df is None or df.empty or len(df) < 2:
            return False
        dates = pd.DatetimeIndex(df.index).sort_values().normalize()
        max_gap = dates.to_series().diff().dt.days.max()
        return pd.notna(max_gap) and max_gap > 7

    @staticmethod
    def _is_complete(df: pd.DataFrame, start: str, end: str, market: str = "us") -> bool:
        """Check if cached data is complete up to *end*.

        Uses real exchange calendar when available (NYSE for US, SSE for CN):
          - counts un-cached trading sessions between last bar and *end*
          - if any → incomplete, triggers tail fetch

        Falls back to weekday heuristic when ``exchange-calendars`` is not installed:
          - Monday:      3-day buffer (covers weekends)
          - Sat/Sun:     2-day buffer
          - Tue–Fri:     1-day buffer

        Internal holes > 7 calendar days also trigger re-fetch.
        """
        if df is None or df.empty:
            return False

        last = pd.Timestamp(df.index[-1]).normalize()
        expected_last = pd.Timestamp(end).normalize()

        # ── Real trading calendar ─────────────────────────────────────
        cal = _get_calendar(market)
        if cal is not None:
            sessions = cal.sessions_in_range(last + pd.Timedelta(days=1), expected_last)
            if len(sessions) > 0:
                return False
        else:
            # ── Fallback: weekday heuristic ───────────────────────────
            gap = (expected_last - last).days
            weekday = expected_last.weekday()
            if weekday == 0:          # Monday
                allowed = 3
            elif weekday in (5, 6):   # Sat / Sun
                allowed = 2
            else:                     # Tue – Fri
                allowed = 1
            if gap > allowed:
                return False

        # Internal data quality check — suspicious gaps > 7 days
        dates = pd.DatetimeIndex(df.index).sort_values().normalize()
        if len(dates) > 1:
            max_gap_days = dates.to_series().diff().dt.days.max()
            if pd.notna(max_gap_days) and max_gap_days > 7:
                return False
        return True
