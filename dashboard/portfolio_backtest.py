"""Portfolio backtest tab + Monte Carlo expander."""

import io
import math

import numpy as np
import pandas as pd
import streamlit as st

import utils
from utils import get_logger
from utils.font import setup_chinese_font
from engine.portfolio import PortfolioBacktest, Leg
from utils.metrics import drawdown_stats, exposure_from_trades

import matplotlib.pyplot as plt

logger = get_logger("dashboard.portfolio_backtest")


def _safe_metric(label, value, fmt="%.1f%%"):
    try:
        if not math.isfinite(value):
            st.metric(label, "N/A")
        else:
            st.metric(label, fmt % value)
    except Exception:
        st.metric(label, "N/A")


def render_portfolio_backtest(config, target_date, backtest_years,
                              allocation_mode, pf_strategy,
                              strategy_options, symbols):
    st.header("组合回测")

    start = (pd.Timestamp(target_date) - pd.DateOffset(years=backtest_years)).strftime("%Y-%m-%d")
    end = target_date.isoformat()

    @st.cache_data(ttl=600)
    def _cached_portfolio_bt(s_start, s_end, alloc, strategy, _cache_buster):
        legs = [Leg(item["symbol"], strategy) for item in config.get("watchlist", [])]
        bt = PortfolioBacktest(
            legs=legs,
            initial_capital=100000,
            allocation=alloc,
        )
        return bt.run(start=s_start, end=s_end)

    # cache_buster = current timestamp to force refresh on strategy switch
    cache_buster = hash((allocation_mode, pf_strategy, start, end))
    pf_result = _cached_portfolio_bt(start, end, allocation_mode, pf_strategy, cache_buster)

    # --- Metrics row ---
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("总收益", f"{pf_result.total_return_pct:+.1f}%")
    m2.metric("夏普", f"{pf_result.sharpe_ratio:.2f}")
    m3.metric("最大回撤", f"{pf_result.max_drawdown_pct:.1f}%")
    m4.metric("交易笔数", pf_result.total_trades)
    m5.metric("胜率", f"{pf_result.win_rate_pct:.1f}%")
    m6.metric("盈亏比", f"{pf_result.profit_factor:.2f}")

    # --- Equity curve + Drawdown ---
    setup_chinese_font()

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    curve = pf_result.equity_curve
    ax1 = axes[0]
    ax1.plot(curve.index, curve, color="#2ca02c", linewidth=1.2, label="组合权益")
    ax1.axhline(y=pf_result.initial_capital, color="gray", linewidth=0.5,
                linestyle=":", alpha=0.5)
    ax1.set_ylabel("Equity ($)")
    ax1.set_title("组合权益曲线", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    rolling_max = curve.expanding().max()
    drawdown = (curve - rolling_max) / rolling_max * 100
    ax2.fill_between(drawdown.index, drawdown, 0, color="#d62728", alpha=0.4)
    ax2.plot(drawdown.index, drawdown, color="#d62728", linewidth=0.6)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.set_title("组合回撤", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # --- Risk dashboard ---
    _render_risk_dashboard(pf_result)

    # --- Trade statistics cards ---
    st.subheader("交易统计")
    if pf_result.total_trades == 0:
        st.info(f"无已完成交易（{len(pf_result.trades)} 笔未平仓）。尝试扩大回测年数或更换策略。")
    else:
        s1, s2, s3, s4, s5, s6 = st.columns(6)
        s1.metric("总笔数", pf_result.total_trades)
        s2.metric("胜率", f"{pf_result.win_rate_pct:.1f}%")
        s3.metric("盈亏比", f"{pf_result.profit_factor:.2f}")
        s4.metric("平均盈利", f"${pf_result.avg_win:,.0f}")
        s5.metric("平均亏损", f"${pf_result.avg_loss:,.0f}")
        s6.metric("平均持仓天", f"{pf_result.avg_hold_days:.1f}")

    # --- PnL Attribution ---
    _render_pnl_attribution(pf_result, pf_strategy)

    # --- Drawdown Attribution ---
    _render_drawdown_attribution(pf_result)

    # --- Filtered trade details ---
    _render_trade_details(pf_result, pf_strategy)


def _render_risk_dashboard(pf_result):
    st.subheader("风险看板")

    curve = pf_result.equity_curve
    current_dd, max_dd_pct, longest_dd_days = drawdown_stats(curve)
    exposure_series, last_exposure, top_weights = exposure_from_trades(pf_result, curve)

    r1, r2, r3, r4, r5 = st.columns(5)
    with r1:
        st.metric("当前回撤", f"{current_dd:.1f}%",
                  delta=f"峰值 {max_dd_pct:.1f}%" if current_dd < -0.1 else None)
    with r2:
        st.metric("最长回撤天数", f"{longest_dd_days} 天")
    with r3:
        st.metric("期末净敞口", f"{last_exposure:.1f}%",
                  delta=f"峰值 {exposure_series.max():.0f}%" if len(exposure_series) > 0 else None)
    with r4:
        avg_exposure = exposure_series.mean() if len(exposure_series) > 0 else 0
        st.metric("平均敞口", f"{avg_exposure:.1f}%")
    with r5:
        active_pct = (exposure_series > 0.1).mean() * 100 if len(exposure_series) > 0 else 0
        st.metric("持仓时间占比", f"{active_pct:.0f}%")

    if top_weights:
        st.caption(f"期末 Top 3 标的全重: " +
                   "  |  ".join(f"**{sym}** {wgt:.1f}%" for sym, wgt in top_weights.items()))


def _render_pnl_attribution(pf_result, pf_strategy):
    st.subheader("收益归因")

    if not pf_result.closed_trades:
        st.info(f"无已完成交易（共 {len(pf_result.trades)} 笔开仓记录，0 笔已平仓）")
        return

    try:
        attr_rows = []
        for t in pf_result.closed_trades:
            pnl = float(t.pnl or 0)
            try:
                month_str = str(t.entry_time)[:7] if t.entry_time else "?"
            except Exception:
                month_str = "?"
            attr_rows.append({
                "symbol": str(t.symbol),
                "month": month_str,
                "pnl": pnl,
                "pnl_pct": float(t.pnl_pct or 0),
            })
        df_attr = pd.DataFrame(attr_rows)
        total_pnl = df_attr["pnl"].sum()
    except Exception as e:
        st.error(f"收益归因解析错误: {e}")
        import traceback
        st.code(traceback.format_exc())
        return

    fc1, fc2, fc3 = st.columns([1, 1, 2])
    with fc1:
        # Use month-based filter instead of entry date
        months = sorted(df_attr["month"].unique().tolist())
        if months:
            attr_start = st.selectbox("起始月", months, index=0, key="attr_start")
            attr_end = st.selectbox("结束月", months, index=len(months)-1, key="attr_end")
        else:
            attr_start = attr_end = ""
    with fc2:
        all_syms = sorted(df_attr["symbol"].unique().tolist())
        attr_sym = st.multiselect("标的", all_syms, default=all_syms, key="attr_sym")

    df_filt = df_attr.copy()
    if attr_start and attr_end:
        df_filt = df_filt[(df_filt["month"] >= attr_start) & (df_filt["month"] <= attr_end)]
    if attr_sym:
        df_filt = df_filt[df_filt["symbol"].isin(attr_sym)]
    filt_pnl = df_filt["pnl"].sum()

    with fc3:
        st.metric("区间总PnL", f"${filt_pnl:+,.0f}",
                  delta=f"全期 ${total_pnl:+,.0f}" if abs(filt_pnl - total_pnl) > 1 else None)

    col_left, col_right = st.columns(2)

    with col_left:
        st.caption("按标的")

        by_sym = df_filt.groupby("symbol")["pnl"].sum().sort_values()
        colors = ["#d62728" if v < 0 else "#2ca02c" for v in by_sym]
        fig_sym, ax_sym = plt.subplots(figsize=(6, 2 + len(by_sym) * 0.35))
        ax_sym.barh(by_sym.index, by_sym.values, color=colors, height=0.6)
        ax_sym.axvline(x=0, color="gray", linewidth=0.5)
        ax_sym.set_xlabel("PnL ($)")
        for i, v in enumerate(by_sym.values):
            ax_sym.text(v + (50 if v >= 0 else -50), i, f"${v:+,.0f}",
                        va="center", fontsize=9, ha="left" if v >= 0 else "right")
        ax_sym.grid(axis="x", alpha=0.3)
        st.pyplot(fig_sym)
        plt.close(fig_sym)

        by_sym_detail = df_filt.groupby("symbol").agg(
            总PnL=("pnl", "sum"), 笔数=("pnl", "count"),
            胜率=("pnl", lambda x: (x > 0).mean() * 100),
            平均PnL=("pnl", "mean"), 最大=("pnl", "max"), 最小=("pnl", "min"),
        ).sort_values("总PnL", ascending=False)
        by_sym_detail["贡献%"] = (by_sym_detail["总PnL"] / filt_pnl * 100).round(1) if filt_pnl != 0 else 0
        by_sym_detail["总PnL"] = by_sym_detail["总PnL"].round(0).astype(int)
        by_sym_detail["平均PnL"] = by_sym_detail["平均PnL"].round(0).astype(int)
        by_sym_detail["胜率"] = by_sym_detail["胜率"].round(1)
        st.dataframe(by_sym_detail, width="stretch")

    with col_right:
        st.caption("按月份")

        by_month = df_filt.groupby("month")["pnl"].sum()
        if len(by_month) > 0:
            colors_m = ["#d62728" if v < 0 else "#2ca02c" for v in by_month]
            fig_mo, ax_mo = plt.subplots(figsize=(6, 2 + len(by_month) * 0.35))
            ax_mo.barh(by_month.index, by_month.values, color=colors_m, height=0.6)
            ax_mo.axvline(x=0, color="gray", linewidth=0.5)
            ax_mo.set_xlabel("PnL ($)")
            for i, v in enumerate(by_month.values):
                ax_mo.text(v + (50 if v >= 0 else -50), i, f"${v:+,.0f}",
                           va="center", fontsize=9, ha="left" if v >= 0 else "right")
            ax_mo.grid(axis="x", alpha=0.3)
            ax_mo.invert_yaxis()
            st.pyplot(fig_mo)
            plt.close(fig_mo)

            by_mo_detail = df_filt.groupby("month").agg(
                总PnL=("pnl", "sum"), 笔数=("pnl", "count"),
                胜率=("pnl", lambda x: (x > 0).mean() * 100),
                月均PnL=("pnl_pct", "mean"),
            ).sort_index(ascending=False)
            by_mo_detail["累计PnL"] = by_mo_detail["总PnL"].cumsum()
            by_mo_detail["总PnL"] = by_mo_detail["总PnL"].round(0).astype(int)
            by_mo_detail["胜率"] = by_mo_detail["胜率"].round(1)
            by_mo_detail["月均PnL"] = by_mo_detail["月均PnL"].round(2)
            by_mo_detail["累计PnL"] = by_mo_detail["累计PnL"].round(0).astype(int)
            st.dataframe(by_mo_detail, width="stretch")

    st.caption("贡献源 & 拖累源")
    c1, c2 = st.columns(2)
    with c1:
        top3 = df_filt.groupby("symbol")["pnl"].sum().nlargest(3)
        for sym, p in top3.items():
            pct = f"{p / filt_pnl * 100:.0f}%" if filt_pnl != 0 else "—"
            st.metric(f"↑ {sym}", f"${p:+,.0f}", delta=pct)
    with c2:
        bot3 = df_filt.groupby("symbol")["pnl"].sum().nsmallest(3)
        for sym, p in bot3.items():
            pct = f"{p / filt_pnl * 100:.0f}%" if filt_pnl != 0 else "—"
            st.metric(f"↓ {sym}", f"${p:+,.0f}", delta=pct)


def _render_drawdown_attribution(pf_result):
    """Per-episode drawdown attribution across the historical equity curve.

    Uses :func:`analysis.drawdown_attribution.historical_drawdown_attribution`
    to scan the top-N deepest drawdown episodes and produce a per-symbol
    breakdown of each.  Each row in the summary table is one episode;
    expanding it reveals the per-symbol contribution.
    """
    from analysis.drawdown_attribution import historical_drawdown_attribution

    st.subheader("回撤归因")
    st.caption(
        "扫描历史回撤区间, 把每次 peak→trough 期间的交易按标的汇总 — "
        "回答\"哪些回撤是谁拖下去的, 是不是同一批标的反复出问题\"."
    )

    curve = pf_result.equity_curve
    if curve.empty or len(curve) < 2:
        st.info("权益曲线为空")
        return

    n_episodes = st.slider("展示历史回撤数", 1, 10, 5, key="dd_attr_n")
    min_depth = st.slider("过滤浅回撤(%)", -10.0, 0.0, -1.0, step=0.5,
                          key="dd_attr_min")

    episodes = historical_drawdown_attribution(
        curve, pf_result.closed_trades or [],
        top_n_episodes=n_episodes, min_depth_pct=min_depth,
    )
    if not episodes:
        st.success(f"回测期内无 < {min_depth}% 的回撤")
        return

    # ── Summary metrics across all returned episodes ────────────
    total_depth = sum(ep["depth_usd"] for ep in episodes)
    n_total_trades = sum(ep["n_trades_in_window"] for ep in episodes)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("回撤次数", len(episodes))
    m2.metric("累计回撤", f"${total_depth:+,.0f}")
    m3.metric("涉及交易笔数", n_total_trades)
    m4.metric("最深", f"{episodes[0]['depth_pct']:.1f}%")

    # ── Episode summary table with top contributors inline ─────
    summary_rows = []
    for i, ep in enumerate(episodes):
        peak = ep["peak_date"].strftime("%Y-%m-%d")
        trough = ep["trough_date"].strftime("%Y-%m-%d")
        top_loser = ep["by_symbol"][0] if ep["by_symbol"] else None
        top_gainer = ep["by_symbol"][-1] if ep["by_symbol"] else None
        summary_rows.append({
            "#": i + 1,
            "Peak → Trough": f"{peak} → {trough}",
            "深度": f"{ep['depth_pct']:.1f}% (${ep['depth_usd']:+,.0f})",
            "天数": ep["duration_to_trough"],
            "恢复天数": (str(ep["duration_to_recovery"])
                       if ep["duration_to_recovery"] is not None else "仍在 DD"),
            "交易笔数": ep["n_trades_in_window"],
            "↓ 最大拖累": (f"{top_loser['symbol']} ${top_loser['pnl']:+,.0f}"
                          if top_loser and top_loser["pnl"] < 0 else "—"),
            "↑ 最大对冲": (f"{top_gainer['symbol']} ${top_gainer['pnl']:+,.0f}"
                          if top_gainer and top_gainer["pnl"] > 0 else "—"),
        })
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True,
                 width="stretch")

    # ── Per-episode drill-down (expander) ──────────────────────
    for i, ep in enumerate(episodes):
        peak = ep["peak_date"].strftime("%Y-%m-%d")
        trough = ep["trough_date"].strftime("%Y-%m-%d")
        label = (f"#{i+1}  {peak} → {trough}   "
                 f"{ep['depth_pct']:.1f}%   ${ep['depth_usd']:+,.0f}")
        with st.expander(label, expanded=(i == 0)):
            if not ep["by_symbol"]:
                st.info("该区间内无已平仓交易")
                continue
            rows = [{
                "标的": r["symbol"],
                "区间 PnL": int(round(r["pnl"])),
                "交易笔数": r["n_trades"],
                "占回撤 %": (f"{r['contribution_pct']:.1f}%"
                            if r["contribution_pct"] is not None else "—"),
            } for r in ep["by_symbol"]]
            st.dataframe(pd.DataFrame(rows), hide_index=True,
                         width="stretch")
            if abs(ep["unexplained_usd"]) > abs(ep["depth_usd"]) * 0.1:
                st.caption(
                    f"⚠ 未解释部分 ${ep['unexplained_usd']:+,.0f} — "
                    "差额来自该区间未平仓持仓的浮动盈亏 (closed_trades 不含)"
                )


def _render_trade_details(pf_result, pf_strategy):
    st.subheader("交易明细")

    if not pf_result.closed_trades:
        st.info(f"无已平仓交易（共 {len(pf_result.trades)} 笔开仓，0 笔已平仓）。低频策略在短回测期内可能尚未平仓，尝试扩大回测年数。")
        return

    try:
        trade_rows = []
        for t in pf_result.closed_trades:
            try:
                entry_str = str(t.entry_time)[:10] if t.entry_time else ""
                exit_str = str(t.exit_time)[:10] if t.exit_time else ""
            except Exception:
                entry_str = str(t.entry_time)[:10] if t.entry_time else ""
                exit_str = ""
            direction_label = "做多" if getattr(t, 'direction', 'LONG') == 'LONG' else "做空"
            trade_rows.append({
                "方向": direction_label,
                "标的": str(t.symbol),
                "入场日": entry_str,
                "出场日": exit_str,
                "数量": int(t.qty) if t.qty else 0,
                "入场价": round(float(t.entry_price or 0), 2),
                "出场价": round(float(t.exit_price or 0), 2) if t.exit_price else 0,
                "PnL": int(round(t.pnl or 0)),
                "PnL%": round(float(t.pnl_pct or 0), 2),
                "原因": str(t.reason or ""),
                "持仓天": int(t.hold_days or 0),
            })
    except Exception as e:
        st.error(f"交易明细解析错误: {e}")
        import traceback
        st.code(traceback.format_exc())
        return
    df_trades = pd.DataFrame(trade_rows)

    fr1, fr2, fr3, fr4 = st.columns(4)
    with fr1:
        sym_list = sorted(df_trades["标的"].unique().tolist())
        filter_sym = st.multiselect("标的", sym_list, default=sym_list, key="pf_filter_sym")
    with fr2:
        reasons = sorted(df_trades["原因"].unique().tolist())
        filter_reason = st.multiselect("原因", reasons, default=reasons, key="pf_filter_reason")
    with fr3:
        min_pnl = int(df_trades["PnL"].min())
        max_pnl = int(df_trades["PnL"].max())
        filter_pnl_range = st.slider("PnL ($)", min_pnl, max_pnl,
                                     (min_pnl, max_pnl), step=100, key="pf_filter_pnl_range")
    with fr4:
        max_hold = int(df_trades["持仓天"].max())
        filter_hold = st.slider("持仓天数", 0, max(1, max_hold),
                                (0, max_hold), step=1, key="pf_filter_hold")

    df_filtered = df_trades[df_trades["标的"].isin(filter_sym)]
    df_filtered = df_filtered[df_filtered["原因"].isin(filter_reason)]
    df_filtered = df_filtered[(df_filtered["PnL"] >= filter_pnl_range[0]) &
                               (df_filtered["PnL"] <= filter_pnl_range[1])]
    df_filtered = df_filtered[(df_filtered["持仓天"] >= filter_hold[0]) &
                               (df_filtered["持仓天"] <= filter_hold[1])]

    ec1, ec2 = st.columns(2)
    with ec1:
        st.caption(f"共 {len(df_filtered)} 笔（筛选自 {len(df_trades)} 笔）")
    with ec2:
        filtered_pnl = df_filtered["PnL"].sum()
        st.metric("筛选PnL合计", f"${filtered_pnl:+,.0f}")

    display_cols = ["方向", "标的", "入场日", "出场日", "数量", "入场价", "出场价", "PnL", "PnL%", "原因", "持仓天"]
    st.dataframe(df_filtered[display_cols], width="stretch", hide_index=True, height=400)


def render_monte_carlo(strategy_options, symbols, target_date):
    with st.expander("Monte Carlo 风控快照", expanded=False):
        mc_sym = st.selectbox("标的", symbols, key="mc_sym")
        mc_strat = st.selectbox("策略", strategy_options, index=3, key="mc_strat")
        mc_years = st.slider("回测年数", 2, 8, 4, key="mc_years")

        if st.button("运行 Monte Carlo", key="mc_run"):
            mc_start = (pd.Timestamp(target_date) - pd.DateOffset(years=mc_years)).strftime("%Y-%m-%d")
            mc_end = target_date.isoformat()

            with st.spinner("Monte Carlo 运行中..."):
                legs = [Leg(mc_sym, mc_strat)]
                pf = PortfolioBacktest(legs, initial_capital=10000)
                result = pf.run(start=mc_start, end=mc_end)

                pnl_pcts = np.array([
                    t.pnl / result.initial_capital * 100
                    for t in result.trades if t.pnl is not None
                ])

                if len(pnl_pcts) >= 5:
                    n_sims = 1000
                    max_dds = []
                    rng = np.random.default_rng(42)
                    for _ in range(n_sims):
                        shuffled = rng.permutation(pnl_pcts)
                        equity = 100.0
                        peak = 100.0
                        max_dd = 0.0
                        for pnl in shuffled:
                            equity *= (1 + pnl / 100)
                            if equity > peak:
                                peak = equity
                            dd = (peak - equity) / peak * 100
                            if dd > max_dd:
                                max_dd = dd
                        max_dds.append(max_dd)
                    max_dds = np.array(max_dds)

                    p95 = np.percentile(max_dds, 95)
                    p50 = np.percentile(max_dds, 50)
                    win_rate = (pnl_pcts > 0).sum() / len(pnl_pcts) * 100

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("P95 最坏回撤", f"{p95:.1f}%")
                    c2.metric("中位回撤", f"{p50:.1f}%")
                    c3.metric("交易数", len(pnl_pcts))
                    c4.metric("胜率", f"{win_rate:.0f}%")

                    if p95 <= 15:
                        st.success("回撤风险: 低 — 策略质量正常")
                    elif p95 <= 25:
                        st.warning("回撤风险: 中 — 关注策略表现")
                    else:
                        st.error("回撤风险: 高 — 建议检查策略参数或市场状态")
                else:
                    st.info(f"交易数量不足 ({len(pnl_pcts)}笔)，至少需要 5 笔")
