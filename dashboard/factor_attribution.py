"""Dashboard tab: factor attribution for any portfolio.

Reuses the PortfolioBacktest equity curve, then runs
:class:`analysis.factor_attribution.FactorAttribution` against the
ETF-proxy factor model.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from data import DataProvider
from engine.portfolio import Leg, PortfolioBacktest
from analysis.factor_returns import FactorReturns
from analysis.factor_attribution import FactorAttribution, TRADING_DAYS

logger = logging.getLogger(__name__)


def render_factor_attribution(
    config: dict,
    target_date: date,
    backtest_years: int,
    allocation_mode: str,
    pf_strategy: str,
    symbols: list[str],
):
    """Render the factor attribution tab."""
    st.header("因子归因 — 解构组合收益")
    st.caption(
        "用 ETF 代理因子 (MKT/SMB/HML/MOM/QMJ/BAB) 回归组合权益曲线, "
        "拆出 Jensen α 与各因子暴露 β。t-stat ≥ 2 才算统计显著。"
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        factor_mode = st.selectbox(
            "因子集",
            ["full (6 因子)", "ff3 (3 因子)"],
            index=0,
            help="full 需要数据 >= 2013-08; ff3 可回溯到 2000-06",
        )
        mode = "full" if factor_mode.startswith("full") else "ff3"
    with col_b:
        rolling_window = st.number_input(
            "滚动窗口 (交易日)",
            min_value=60, max_value=750, value=TRADING_DAYS, step=21,
            help="252 ≈ 1 年; 126 ≈ 半年",
        )
    with col_c:
        do_run = st.button("跑因子归因", type="primary")
    with col_a:
        do_clear = st.button("清空", type="secondary",
                             help="清除已生成的归因结果")
    if do_clear:
        for k in ("_fa_result", "_fa_equity", "_fa_factors", "_fa_attr"):
            st.session_state.pop(k, None)
        st.rerun()

    # Streamlit reruns the page on every widget interaction.  Stash
    # results in session_state so downstream buttons (e.g. future "导出"
    # additions) don't lose the build.  See dashboard/risk_report.py for
    # the canonical pattern.
    if do_run:
        if not symbols:
            st.warning("watchlist 为空, 无法跑组合回测")
            return
        start = (pd.Timestamp(target_date) - pd.DateOffset(years=backtest_years)).strftime("%Y-%m-%d")
        end = target_date.isoformat()
        with st.spinner("跑组合回测..."):
            legs = [Leg(sym, pf_strategy) for sym in symbols]
            bt = PortfolioBacktest(
                legs=legs,
                initial_capital=100000,
                allocation=allocation_mode,
            )
            try:
                pf_result = bt.run(start=start, end=end)
            except Exception as e:
                st.error(f"组合回测失败: {e}")
                return
        equity = pf_result.equity_curve
        if equity is None or equity.empty:
            st.warning("组合回测无数据")
            return
        with st.spinner("加载因子收益 + 跑回归..."):
            factors = FactorReturns(mode=mode).load(start, end)
            if factors.empty:
                st.error("因子数据加载失败 (检查 SPY/IWM/IVE/IVW/MTUM/QUAL/USMV/SHV 是否可拉)")
                return
            attr = FactorAttribution(equity, factors)
            try:
                result = attr.regress()
            except ValueError as e:
                st.error(f"回归失败: {e}")
                return
        st.session_state["_fa_result"] = result
        st.session_state["_fa_equity"] = equity
        st.session_state["_fa_factors"] = factors
        st.session_state["_fa_attr"] = attr

    if "_fa_result" not in st.session_state:
        return

    result = st.session_state["_fa_result"]
    equity = st.session_state["_fa_equity"]
    factors = st.session_state["_fa_factors"]
    attr = st.session_state["_fa_attr"]

    # -------------------------------------------------------------------
    # Headline metrics
    # -------------------------------------------------------------------
    st.subheader("回归结果")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("α (年化)", f"{result.alpha_annual * 100:+.2f}%",
              f"t={result.alpha_tstat:+.2f}")
    m2.metric("R²", f"{result.r_squared:.3f}",
              f"adj {result.adj_r_squared:.3f}")
    m3.metric("观测数", f"{result.n_obs}")
    sig_marker = "✓ 显著" if result.alpha_is_significant else "✗ 不显著"
    m4.metric("α 显著性", sig_marker, f"p={result.alpha_pvalue:.3f}")

    # Verdict
    if "真 alpha" in result.verdict:
        st.success(f"📊 **{result.verdict}**")
    elif "可疑" in result.verdict or "信号弱" in result.verdict:
        st.warning(f"📊 **{result.verdict}**")
    else:
        st.info(f"📊 **{result.verdict}**")

    # -------------------------------------------------------------------
    # Factor loadings table
    # -------------------------------------------------------------------
    st.subheader("因子载荷")
    table = []
    for name in result.factor_names:
        table.append({
            "因子": name,
            "β": f"{result.betas[name]:+.3f}",
            "t-stat": f"{result.beta_tstats[name]:+.2f}",
            "p-value": f"{result.beta_pvalues[name]:.3f}",
            "显著性": "✓" if abs(result.beta_tstats[name]) >= 2 else "—",
        })
    st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)

    # -------------------------------------------------------------------
    # Contribution decomposition
    # -------------------------------------------------------------------
    st.subheader("收益分解")
    contrib = attr.contribution()
    # Aggregate to annual contribution per source
    annual = contrib.resample("YE").sum()
    if not annual.empty:
        annual_pct = annual * 100
        annual_pct.index = annual_pct.index.year
        st.caption("各年度收益贡献分解 (%)")
        st.dataframe(
            annual_pct.style.format("{:+.2f}").background_gradient(
                cmap="RdYlGn", axis=None
            ),
            width="stretch",
        )

    # -------------------------------------------------------------------
    # Rolling alpha
    # -------------------------------------------------------------------
    st.subheader(f"滚动 α (window={rolling_window})")
    try:
        rolling = attr.rolling_alpha(window_days=int(rolling_window))
    except ValueError as e:
        st.warning(f"滚动 α 失败: {e}")
        return

    if rolling.empty:
        st.info("无足够数据计算滚动 α")
        return

    # Two-row chart: top = annual alpha, bottom = t-stat
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.font import setup_chinese_font
    try:
        setup_chinese_font()
    except Exception:
        pass

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    alpha_pct = rolling["alpha_annual"] * 100
    colors = ["#2ca02c" if a >= 0 else "#d62728" for a in alpha_pct]
    axes[0].bar(rolling.index, alpha_pct, color=colors, width=10)
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].set_ylabel("年化 α (%)")
    axes[0].set_title("滚动 Jensen α")

    axes[1].plot(rolling.index, rolling["alpha_tstat"],
                 color="#1f77b4", linewidth=1.2)
    axes[1].axhline(2, color="gray", linestyle="--", linewidth=0.5)
    axes[1].axhline(-2, color="gray", linestyle="--", linewidth=0.5)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_ylabel("α t-stat")
    axes[1].set_xlabel("窗口结束日期")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "**判读**: t-stat 持续 > 2 表示策略 alpha 稳定; 趋势下行说明 alpha 退化; "
        "围绕 0 振荡说明收益基本被因子模型解释。"
    )
