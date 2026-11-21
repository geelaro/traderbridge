"""Realized + Unrealized PnL aggregation.

Realized PnL
------------
Comes from the ``trade_pnl`` table — every fully-closed round-trip is
persisted by :class:`OrderManager` on the closing fill.  We aggregate by
symbol, by month, and over the requested time window.

Unrealized PnL
--------------
Computed from current open positions: ``(current_price − entry_price) × qty``.
Positions come from either:
  - broker.get_positions() — actual holdings when wired to a live broker
  - compute_hypothetical_positions() — strategy-derived "would-be" positions
    when no broker is connected (dry-run / paper).

Together they answer:
  Total PnL = Realized (already crystalised) + Unrealized (still open).
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Iterable, Mapping, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Time-window helpers
# ---------------------------------------------------------------------------


_PERIOD_DAYS = {
    "7d": 7,
    "30d": 30,
    "90d": 90,
    "1y": 365,
}


def resolve_period(period: str, now: Optional[datetime] = None) -> tuple[Optional[str], Optional[str]]:
    """Map a period token to (since, until) iso-date strings.

    Tokens: 7d / 30d / 90d / 1y / ytd / all.  Returns (None, None) for
    "all" — caller skips filtering.
    """
    if not period or period == "all":
        return None, None
    now = now or datetime.now()
    until = now.strftime("%Y-%m-%d")
    if period == "ytd":
        since = f"{now.year}-01-01"
    elif period in _PERIOD_DAYS:
        since = (now - timedelta(days=_PERIOD_DAYS[period])).strftime("%Y-%m-%d")
    else:
        return None, None
    return since, until


# ---------------------------------------------------------------------------
# Realized PnL
# ---------------------------------------------------------------------------


def realized_pnl_summary(
    cache,
    since: Optional[str] = None,
    until: Optional[str] = None,
    limit: int = 10000,
) -> dict:
    """Aggregate realized PnL from the trade_pnl table.

    Parameters
    ----------
    cache : CacheManager
        Anything exposing ``query_trade_pnl(symbol=None, limit=N)``.
    since, until : YYYY-MM-DD strings, optional
        Filter by exit_date inclusive.  ``None`` means open-ended.
    limit : int
        Hard cap on rows pulled from cache (defensive).

    Returns
    -------
    dict::

        {
            "total":          float,       # sum of pnl across the period
            "n_trades":       int,
            "n_wins":         int,
            "n_losses":       int,
            "win_rate_pct":   float,
            "avg_win":        float,
            "avg_loss":       float,
            "by_symbol":      {sym: pnl},   # ordered: largest |pnl| first
            "by_month":       {YYYY-MM: pnl},
            "trades":         list[dict],  # raw rows in the window, sorted by exit_date desc
        }
    """
    raw = cache.query_trade_pnl(limit=limit) if hasattr(cache, "query_trade_pnl") else []
    rows = [r for r in raw if _in_window(r.get("exit_date"), since, until)]
    if not rows:
        return _empty_realized()

    pnls = [float(r.get("pnl", 0) or 0) for r in rows]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    by_symbol: dict = defaultdict(float)
    by_month: dict = defaultdict(float)
    for r in rows:
        pnl = float(r.get("pnl", 0) or 0)
        sym = r.get("symbol", "?")
        by_symbol[sym] += pnl
        month = (r.get("exit_date") or "")[:7]
        if month:
            by_month[month] += pnl

    return {
        "total": float(sum(pnls)),
        "n_trades": len(rows),
        "n_wins": len(wins),
        "n_losses": len(losses),
        "win_rate_pct": (len(wins) / len(rows) * 100) if rows else 0.0,
        "avg_win": (sum(wins) / len(wins)) if wins else 0.0,
        "avg_loss": (sum(losses) / len(losses)) if losses else 0.0,
        "by_symbol": dict(sorted(by_symbol.items(), key=lambda kv: -abs(kv[1]))),
        "by_month": dict(sorted(by_month.items())),
        "trades": sorted(rows, key=lambda r: r.get("exit_date", ""), reverse=True),
    }


# ---------------------------------------------------------------------------
# Unrealized PnL
# ---------------------------------------------------------------------------


def unrealized_pnl_summary(positions: Iterable[dict]) -> dict:
    """Compute current unrealized PnL from open positions.

    Each position dict needs at minimum::

        {"symbol": str, "entry_price": float, "current_price": float,
         "shares": int}

    Optional keys (passed through to per-symbol output): ``strategy``,
    ``entry_date``, ``direction``. Shares default to 1 if missing.

    Returns
    -------
    dict::

        {
            "total":      float,
            "n_positions": int,
            "n_winning":  int,
            "n_losing":   int,
            "by_symbol":  list[dict]   # sorted by |pnl| descending
        }
    """
    rows: list[dict] = []
    total = 0.0
    n_win = n_loss = 0
    for p in positions or []:
        sym = p.get("symbol")
        ep = float(p.get("entry_price", 0) or 0)
        cp = float(p.get("current_price", 0) or 0)
        # Default shares to 1 only when the key is missing entirely; an
        # explicit 0 / None means "invalid position" and is skipped below.
        # (dict.get returns None — not the default — when the value is None,
        # so the simple form below preserves the same semantics.)
        shares_raw = p.get("shares", 1)
        try:
            shares = float(shares_raw) if shares_raw is not None else 0.0
        except (TypeError, ValueError):
            shares = 0.0
        if not sym or ep <= 0 or cp <= 0 or shares <= 0:
            continue
        direction = p.get("direction", "LONG")
        if direction not in ("LONG", "SHORT"):
            continue
        sign = -1 if direction == "SHORT" else 1
        pnl = sign * (cp - ep) * shares
        pnl_pct = sign * (cp / ep - 1) * 100
        if pnl > 0:
            n_win += 1
        elif pnl < 0:
            n_loss += 1
        total += pnl
        rows.append({
            "symbol": sym,
            "entry_price": ep,
            "current_price": cp,
            "shares": shares,
            "direction": direction,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "strategy": p.get("strategy"),
            "entry_date": p.get("entry_date"),
        })
    rows.sort(key=lambda r: -abs(r["pnl"]))
    return {
        "total": float(total),
        "n_positions": len(rows),
        "n_winning": n_win,
        "n_losing": n_loss,
        "by_symbol": rows,
    }


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------


def pnl_summary(
    cache,
    positions: Iterable[dict],
    period: str = "all",
    now: Optional[datetime] = None,
) -> dict:
    """All-in-one: realized + unrealized + grand total.

    Parameters
    ----------
    cache : CacheManager
    positions : iterable of position dicts (see unrealized_pnl_summary)
    period : str
        Window for realized (7d / 30d / 90d / 1y / ytd / all).  Defaults
        to ``all`` — full history.  Unrealized is "as of now" regardless.
    now : datetime, optional
        Reference time for the period window.  Defaults to ``datetime.now()``.
    """
    since, until = resolve_period(period, now)
    realized = realized_pnl_summary(cache, since=since, until=until)
    unrealized = unrealized_pnl_summary(positions)
    return {
        "period": period,
        "since": since,
        "until": until,
        "realized": realized,
        "unrealized": unrealized,
        "total": realized["total"] + unrealized["total"],
    }


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _in_window(exit_date: Optional[str], since: Optional[str],
               until: Optional[str]) -> bool:
    if not exit_date:
        return False
    d = exit_date[:10]
    if since and d < since:
        return False
    if until and d > until:
        return False
    return True


def _empty_realized() -> dict:
    return {
        "total": 0.0,
        "n_trades": 0,
        "n_wins": 0,
        "n_losses": 0,
        "win_rate_pct": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "by_symbol": {},
        "by_month": {},
        "trades": [],
    }
