"""Data-source health tracking and splits.json change detection.

ROADMAP.md 阶段八 Ops-1 的落地工具. All persistence goes through
``CacheManager``/``StateStore`` (``source_health`` table +
``risk_state`` key/value) — this module holds the business logic
(cooldown math, hash comparison, report shaping), not raw SQL.

Why this replaces ``DataProvider._failed_sources``
----------------------------------------------------
The prior in-memory blacklist (``data/provider.py``) is permanent-until
-process-restart and resets on every new ``DataProvider()`` — i.e. every
Streamlit rerun. It looks like a cooldown but isn't one. ``is_in_cooldown``
here is a real time-window check backed by a persisted last-failure
timestamp, so it survives reruns and actually expires.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import pandas as pd

_SPLITS_HASH_KEY = "splits_hash"


def record_fetch_result(cache, source: str, success: bool, detail: str = "") -> None:
    """Record one fetch attempt's outcome for *source*. Never raises —
    callers (the hot fetch path in data/provider.py) must not break on a
    health-tracking failure; wrap call sites in try/except regardless."""
    cache.record_source_health(source, success=success, detail=detail)


def is_in_cooldown(cache, source: str, cooldown_minutes: int = 15) -> bool:
    """True if *source* failed within the last *cooldown_minutes* minutes."""
    last_failure = cache.get_last_source_failure(source)
    if not last_failure:
        return False
    elapsed = pd.Timestamp.now() - pd.Timestamp(last_failure)
    return elapsed < pd.Timedelta(minutes=cooldown_minutes)


def source_health_report(cache, days: int = 7, cooldown_minutes: int = 15) -> dict:
    """Per-source health summary for the last *days* days.

    Returns ``{"sources": [{source, success_count, failure_count,
    success_rate, last_failure_ts, last_failure_detail, in_cooldown}]}``.
    """
    rows = cache.get_source_health(days=days)
    sources = []
    for r in rows:
        total = r["success_count"] + r["failure_count"]
        success_rate = r["success_count"] / total if total > 0 else None
        sources.append({
            **r,
            "success_rate": success_rate,
            "in_cooldown": is_in_cooldown(cache, r["source"], cooldown_minutes),
        })
    return {"sources": sources, "days": days}


def check_splits_changed(cache, splits_path: str = "data/splits.json") -> Optional[str]:
    """Detect whether *splits_path* changed since the last check.

    On a detected change, immediately persists the new hash (one-time
    notice, not a repeating nag) and returns a human-readable reminder
    string. Returns None if unchanged, missing, or unreadable.

    Known limitation — does NOT reload data.sources._US_SPLITS (that's a
    module-level constant bound once at import time); the returned message
    says so explicitly. This only tells you the file changed, it does not
    make the change take effect.
    """
    path = Path(splits_path)
    if not path.exists():
        return None
    try:
        content = path.read_bytes()
    except OSError:
        return None
    current_hash = hashlib.sha256(content).hexdigest()

    stored_hash = cache.load_risk_state(_SPLITS_HASH_KEY)
    if stored_hash == current_hash:
        return None

    cache.save_risk_state(_SPLITS_HASH_KEY, current_hash)
    if stored_hash is None:
        return None  # first time ever checking — nothing to compare against, not a "change"

    return (
        f"{splits_path} 已变更 — 已缓存的标的仍是旧调整比例, "
        "需要重启进程并对相关标的执行 force_refresh (或清缓存) 才能生效。"
    )
