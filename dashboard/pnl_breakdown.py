"""Dashboard tab — Realized + Unrealized PnL breakdown.

Aggregates the trade_pnl SQLite table (realized) plus current holdings'
mark-to-market (unrealized) into a unified view: total PnL, win rate,
by-symbol/by-month decomposition, monthly bar chart.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import streamlit as st

from analysis.pnl_breakdown import pnl_summary
from live.position_stops import compute_hypothetical_positions

logger = logging.getLogger(__name__)


_PERIODS = {
    "全部历史": "all",
    "7 天": "7d",
    "30 天": "30d",
    "90 天": "90d",
    "1 年": "1y",
    "YTD": "ytd",
}


def render_pnl_breakdown(config: dict, target_date: date, provider, cache):
    """Render realized + unrealized PnL breakdown."""
    st.header("盈亏分析 — Realized vs Unrealized")
    st.caption(
        "**Realized**: 来自 trade_pnl 表的所有已平仓交易. "
        "**Unrealized**: 当前持仓的 (现价 − 入场价) × 股数 浮动盈亏. "
        "两者相加 = 组合总盈亏 (已落地 + 浮在账上)."
    )

    col_period, col_src = st.columns(2)
    with col_period:
        period_label = st.selectbox(
            "Realized 区间", list(_PERIODS.keys()), index=0,
        )
        period = _PERIODS[period_label]
    with col_src:
        source = st.radio(
            "Unrealized 来源",
            ["假设持仓(active 策略)", "等权 watchlist"],
            horizontal=True,
        )

    # ── Build unrealized positions ─────────────────────────────────
    positions = _build_positions(config, target_date, provider, source)

    # ── Run pnl_summary ────────────────────────────────────────────
    summary = pnl_summary(cache, positions, period=period)
    realized = summary["realized"]
    unrealized = summary["unrealized"]

    # ── Top headline ───────────────────────────────────────────────
    st.subheader("总览")
    cols = st.columns(3)
    cols[0].metric("Realized PnL", f"${realized['total']:+,.0f}",
                   help=f"区间内已平仓 {realized['n_trades']} 笔")
    cols[1].metric("Unrealized PnL", f"${unrealized['total']:+,.0f}",
                   help=f"当前 {unrealized['n_positions']} 个持仓")
    cols[2].metric("合计 (R + U)", f"${summary['total']:+,.0f}")

    # ── Realized detail ────────────────────────────────────────────
    st.subheader("Realized 已实现")
    cols = st.columns(4)
    cols[0].metric("交易笔数", realized["n_trades"])
    cols[1].metric("胜率", f"{realized['win_rate_pct']:.1f}%",
                   help=f"赢 {realized['n_wins']} / 亏 {realized['n_losses']}")
    cols[2].metric("均盈", f"${realized['avg_win']:+,.0f}")
    cols[3].metric("均亏", f"${realized['avg_loss']:+,.0f}")

    by_month = realized.get("by_month", {})
    if by_month:
        st.caption("按月份分解")
        df_month = pd.DataFrame(
            [{"月份": m, "Realized PnL": v} for m, v in by_month.items()]
        ).set_index("月份")
        st.bar_chart(df_month)

    by_sym = realized.get("by_symbol", {})
    if by_sym:
        with st.expander(f"按标的分解 ({len(by_sym)} 个)"):
            rows = [{"标的": s, "Realized PnL": f"${v:+,.0f}"}
                    for s, v in by_sym.items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    if realized["trades"]:
        with st.expander(f"明细交易 ({len(realized['trades'])} 笔, 最新在前)"):
            rows = []
            for t in realized["trades"][:200]:  # cap rendering
                rows.append({
                    "日期": t["exit_date"],
                    "标的": t["symbol"],
                    "方向": t["side"],
                    "数量": t["qty"],
                    "入场价": f"${t['entry_price']:.2f}",
                    "出场价": f"${t['exit_price']:.2f}",
                    "PnL": f"${t['pnl']:+,.0f}",
                    "PnL%": f"{t['pnl_pct']:+.2f}%",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.divider()

    # ── Unrealized detail ──────────────────────────────────────────
    st.subheader("Unrealized 未实现 (当前持仓)")
    if unrealized["n_positions"] == 0:
        st.info("无活跃持仓 — active 策略均已平仓 或 watchlist 为空")
        return

    cols = st.columns(3)
    cols[0].metric("持仓数", unrealized["n_positions"])
    cols[1].metric("浮盈", unrealized["n_winning"])
    cols[2].metric("浮亏", unrealized["n_losing"])

    rows = []
    for r in unrealized["by_symbol"]:
        rows.append({
            "标的": r["symbol"],
            "策略": r.get("strategy") or "—",
            "入场日": r.get("entry_date") or "—",
            "入场价": f"${r['entry_price']:.2f}",
            "现价": f"${r['current_price']:.2f}",
            "方向": "空头" if r.get("direction") == "SHORT" else "多头",
            "股数": int(r["shares"]),
            "浮盈": f"${r['pnl']:+,.0f}",
            "浮盈%": f"{r['pnl_pct']:+.2f}%",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_positions(config, target_date, provider, source: str) -> list[dict]:
    """Build the positions list feeding unrealized_pnl_summary.

    Always returns dicts shaped {symbol, entry_price, current_price, shares,
    strategy?, entry_date?}.  Shares default to 1 (we don't track real share
    count in hypothetical positions — the PnL is per-share until broker
    integration provides real qty).
    """
    if "假设持仓" in source:
        rows = compute_hypothetical_positions(config, target_date, provider)
        if rows:
            # compute_hypothetical_positions already returns the right shape
            # (symbol/entry_price/current_price/strategy/entry_date) — just
            # default shares to 1 for per-share PnL display.
            for r in rows:
                r.setdefault("shares", 1)
            return rows
        # fall through to equal-weight watchlist
    # Equal-weight watchlist: synthesise positions with entry=current, no PnL.
    # This is a deliberate "no open positions to show" path.
    return []
