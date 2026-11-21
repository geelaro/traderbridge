"""Dashboard tab — Brinson performance attribution.

Decomposes the portfolio's active return (vs equal-weighted 11-sector
benchmark using SPDR Select Sector ETFs) into allocation, selection
and interaction effects.

The portfolio source is hypothetical positions from
:func:`live.position_stops.compute_hypothetical_positions`, falling back
to equal-weight watchlist.  Per-sector returns come from each symbol's
period total return; benchmark per-sector returns come from the matching
SPDR ETF.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from analysis.brinson import (
    SECTOR_ETF,
    brinson_attribution,
    compute_period_returns,
    portfolio_sector_breakdown,
)
from live.position_stops import compute_hypothetical_positions
from utils.sectors import DEFAULT_SECTORS

logger = logging.getLogger(__name__)


_PERIOD_OPTIONS = {
    "1 个月": 30,
    "3 个月": 90,
    "6 个月": 180,
    "1 年": 365,
    "YTD": "ytd",
}


def render_brinson_attribution(config: dict, target_date: date, provider):
    """Render the Brinson attribution tab."""
    st.header("业绩归因 — Brinson 分解")
    st.caption(
        "把组合超额收益分解为 **行业配置** (overweight 对了行业?) + "
        "**选股能力** (在每个行业内选对了股?) + **交互效应**. "
        "用 SPDR Select Sector ETFs 作为各行业 benchmark, 等权 11 行业 "
        "作为整体基线."
    )

    col_period, col_src = st.columns(2)
    with col_period:
        period_label = st.selectbox("回看区间", list(_PERIOD_OPTIONS.keys()),
                                    index=2)  # default 6m
    with col_src:
        source = st.radio(
            "持仓来源",
            ["假设持仓(active 策略)", "等权 watchlist"],
            horizontal=True,
        )

    # ── Resolve period dates ───────────────────────────────────────
    end = pd.Timestamp(target_date)
    period = _PERIOD_OPTIONS[period_label]
    if period == "ytd":
        start = pd.Timestamp(end.year, 1, 1)
    else:
        start = end - timedelta(days=int(period))
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    st.caption(f"区间: **{start_str} → {end_str}** ({(end - start).days} 天)")

    # ── Build portfolio symbols + weights ──────────────────────────
    if "假设持仓" in source:
        rows = compute_hypothetical_positions(config, target_date, provider)
        short_count = sum(r.get("direction") == "SHORT" for r in rows)
        if short_count:
            st.info(f"Brinson 归因暂不支持空头权重，已排除 {short_count} 个假设空头")
        symbols = [r["symbol"] for r in rows if r.get("direction", "LONG") == "LONG"]
        if not symbols and not rows:
            symbols = [item["symbol"] for item in config.get("watchlist", [])]
            st.info("无假设持仓 — 自动回退到等权 watchlist")
    else:
        symbols = [item["symbol"] for item in config.get("watchlist", [])]

    if not symbols:
        st.warning("没有可计算的多头标的")
        return

    st.write(f"组合: **{len(symbols)}** 个标的, 等权")

    # ── Fetch portfolio symbol prices ──────────────────────────────
    with st.spinner("拉取组合标的价格..."):
        sym_prices = _fetch_panel(symbols, start_str, end_str, provider)
    if sym_prices.empty:
        st.warning("无法拉取组合价格数据")
        return
    period_returns = compute_period_returns(sym_prices, start_str, end_str)
    if not period_returns:
        st.warning("区间内无足够数据计算收益")
        return

    sector_w, sector_r = portfolio_sector_breakdown(
        symbols, DEFAULT_SECTORS, period_returns,
    )
    if not sector_w:
        st.warning("无法按行业分组组合")
        return

    # ── Fetch benchmark sector ETF prices ──────────────────────────
    bench_etfs = sorted(set(SECTOR_ETF.values()))
    with st.spinner("拉取 SPDR Sector ETFs..."):
        etf_prices = _fetch_panel(bench_etfs, start_str, end_str, provider)
    if etf_prices.empty:
        st.warning("无法拉取 SPDR Sector ETFs 数据 — 检查网络/代理")
        return
    etf_period_returns = compute_period_returns(etf_prices, start_str, end_str)

    # Map ETF returns back to sector names; sectors sharing an ETF (e.g.
    # Consumer/Automotive both XLY) get the same return.
    bench_sector_r: dict = {}
    bench_sector_w: dict = {}
    available_sectors = [
        sec for sec, etf in SECTOR_ETF.items() if etf in etf_period_returns
    ]
    if not available_sectors:
        st.warning("SPDR ETFs 价格数据不全, 无法计算基准")
        return
    # Equal-weight only the sectors we actually have data for
    eq_weight = 1.0 / len(available_sectors)
    for sec in available_sectors:
        bench_sector_w[sec] = eq_weight
        bench_sector_r[sec] = etf_period_returns[SECTOR_ETF[sec]]

    # ── Run BHB ────────────────────────────────────────────────────
    result = brinson_attribution(
        portfolio_weights=sector_w,
        portfolio_returns=sector_r,
        benchmark_weights=bench_sector_w,
        benchmark_returns=bench_sector_r,
    )

    totals = result["totals"]
    df = result["by_sector"]

    # ── Top metrics ────────────────────────────────────────────────
    st.subheader("总览")
    cols = st.columns(4)
    cols[0].metric("组合收益", f"{totals['portfolio_return'] * 100:+.2f}%")
    cols[1].metric("基准收益", f"{totals['benchmark_return'] * 100:+.2f}%")
    cols[2].metric("超额收益", f"{totals['active_return'] * 100:+.2f}%")
    cols[3].metric(
        "三效应和",
        f"{(totals['allocation'] + totals['selection'] + totals['interaction']) * 100:+.2f}%",
        help="应该等于超额收益 (BHB 恒等式)",
    )

    st.subheader("三效应分解")
    cols = st.columns(3)
    cols[0].metric("行业配置", f"{totals['allocation'] * 100:+.2f}%",
                   help="(w_p − w_b) × r_b — 你 over/underweight 对了行业?")
    cols[1].metric("选股能力", f"{totals['selection'] * 100:+.2f}%",
                   help="w_b × (r_p − r_b) — 在每个行业内, 你的标的跑赢/输基准?")
    cols[2].metric("交互效应", f"{totals['interaction'] * 100:+.2f}%",
                   help="(w_p − w_b) × (r_p − r_b) — overweight 且选对方向的额外奖励")

    # ── Per-sector breakdown ───────────────────────────────────────
    if df.empty:
        st.info("无 sector 明细")
        return

    st.subheader("按行业明细")
    display = pd.DataFrame({
        "行业": df.index,
        "组合权重": [f"{v * 100:.1f}%" for v in df["w_p"]],
        "基准权重": [f"{v * 100:.1f}%" for v in df["w_b"]],
        "组合收益": [f"{v * 100:+.2f}%" for v in df["r_p"]],
        "基准收益": [f"{v * 100:+.2f}%" for v in df["r_b"]],
        "配置效应": [f"{v * 100:+.3f}pp" for v in df["allocation"]],
        "选股效应": [f"{v * 100:+.3f}pp" for v in df["selection"]],
        "交互效应": [f"{v * 100:+.3f}pp" for v in df["interaction"]],
        "合计": [f"{v * 100:+.3f}pp" for v in df["total"]],
    })
    st.dataframe(display, hide_index=True, use_container_width=True)

    # ── Top-3 sector contributions ─────────────────────────────────
    df_sorted = df.sort_values("total", ascending=False)
    pos = df_sorted[df_sorted["total"] > 0].head(3)
    neg = df_sorted[df_sorted["total"] < 0].sort_values("total").head(3)

    cols = st.columns(2)
    if not pos.empty:
        cols[0].markdown("**贡献最大的 3 个行业**")
        for sec, row in pos.iterrows():
            cols[0].markdown(
                f"- **{sec}**: {row['total'] * 100:+.2f}pp "
                f"(配置 {row['allocation'] * 100:+.2f} + "
                f"选股 {row['selection'] * 100:+.2f} + "
                f"交互 {row['interaction'] * 100:+.2f})"
            )
    if not neg.empty:
        cols[1].markdown("**拖累最大的 3 个行业**")
        for sec, row in neg.iterrows():
            cols[1].markdown(
                f"- **{sec}**: {row['total'] * 100:+.2f}pp "
                f"(配置 {row['allocation'] * 100:+.2f} + "
                f"选股 {row['selection'] * 100:+.2f} + "
                f"交互 {row['interaction'] * 100:+.2f})"
            )

    st.caption(
        "解读: **配置效应 ↑** 说明你押对了行业;"
        "**选股效应 ↑** 说明在行业内选对了股;"
        "**交互效应 ↑** 说明 overweight 的行业内选股也跑赢了 (双重奖励)."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fetch_panel(symbols, start: str, end: str, provider) -> pd.DataFrame:
    """Fetch close prices for ``symbols`` over [start, end] as a DataFrame."""
    series = {}
    for sym in symbols:
        try:
            df = provider.get_daily(sym, start=start, end=end)
        except Exception as exc:
            logger.debug("brinson fetch failed for %s: %s", sym, exc)
            continue
        if df is None or df.empty or "Close" not in df.columns:
            continue
        series[sym] = df["Close"]
    if not series:
        return pd.DataFrame()
    return pd.concat(series, axis=1).sort_index()
