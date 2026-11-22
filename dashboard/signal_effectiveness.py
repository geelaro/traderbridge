"""Dashboard tab: signal effectiveness via forward-return analysis.

Answers a different question than backtesting: not "would this strategy
make money end-to-end", but "is the entry signal itself predictive".
A strategy can have weak full-cycle returns while its signal is sharp —
the gap usually points at exit / sizing logic, not the entry rule.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from strategy import STRATEGY_MAP
from analysis.forward_return import (
    compute_forward_returns, summarise, DEFAULT_HORIZONS,
)

logger = logging.getLogger(__name__)


_DIR_LABEL = {1: "买入 (long)", -1: "卖出 (short)"}


def render_signal_effectiveness(
    config: dict,
    target_date: date,
    backtest_years: int,
    provider,
    selected_symbol: str,
    selected_strategy: str,
    strategy_options: list[str],
    symbols: list[str],
):
    """Render the signal effectiveness tab."""
    st.header("信号有效性 — Forward Return 分析")
    st.caption(
        "回答一个跟回测不同的问题: 信号触发后 N 天市场的行为如何? "
        "信号本身是否有预测力 (与完整策略是否赚钱解耦)。"
    )

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        sym = st.selectbox(
            "标的", symbols,
            index=symbols.index(selected_symbol) if selected_symbol in symbols else 0,
            key="fr_symbol",
        )
    with col_b:
        strat = st.selectbox(
            "策略", strategy_options,
            index=strategy_options.index(selected_strategy)
            if selected_strategy in strategy_options else 0,
            key="fr_strategy",
        )
    with col_c:
        direction = st.selectbox(
            "信号方向", options=[1, -1],
            format_func=lambda x: _DIR_LABEL[x],
            index=0, key="fr_direction",
        )
    with col_d:
        horizons_in = st.text_input(
            "前瞻天数 (逗号分隔)",
            value=", ".join(str(h) for h in DEFAULT_HORIZONS),
            key="fr_horizons",
        )

    try:
        horizons = sorted({int(x.strip()) for x in horizons_in.split(",") if x.strip().isdigit()})
        if not horizons:
            raise ValueError("empty")
    except Exception:
        st.error("前瞻天数解析失败, 请用逗号分隔的整数, e.g. 30, 90, 180")
        return

    do_run = st.button("跑 Forward Return 分析", type="primary", key="fr_go")
    if not do_run:
        return

    start = (pd.Timestamp(target_date) - pd.DateOffset(years=backtest_years)).strftime("%Y-%m-%d")
    end = target_date.isoformat()

    with st.spinner(f"拉取 {sym} 数据..."):
        try:
            df = provider.get_daily(sym, start=start, end=end)
        except Exception as exc:
            st.error(f"数据获取失败: {exc}")
            return
        if df is None or df.empty:
            st.warning(f"{sym} 无数据")
            return

    with st.spinner(f"计算 {strat} 指标 + 信号..."):
        params = config.get("strategy", {}).get(strat, {})
        try:
            strategy = STRATEGY_MAP[strat](**params)
            df_sig = strategy.calculate_indicators(df)
        except Exception as exc:
            st.error(f"策略计算失败: {exc}")
            return

    fr = compute_forward_returns(df_sig, horizons=horizons, direction=direction)
    stats = summarise(fr, direction=direction)

    # -------------------------------------------------------------------
    # Headline
    # -------------------------------------------------------------------
    st.subheader(f"信号统计 ({_DIR_LABEL[direction]})")
    if stats.n_signals == 0:
        st.info(f"{sym} {strat} 在该时段内没有 {_DIR_LABEL[direction]} 信号")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("总信号数", stats.n_signals)
    m2.metric("回测年数", f"{backtest_years} 年")
    # Best-horizon shortcut for the headline
    best = max(stats.horizons, key=lambda s: abs(s.sharpe)) if stats.horizons else None
    if best:
        m3.metric(
            f"最强 horizon ({best.horizon}d)",
            f"Sharpe {best.sharpe:+.2f}",
            f"win {best.win_rate*100:.0f}%",
        )

    # Verdict banner
    v = stats.verdict
    if "有效" in v and "边际" not in v:
        st.success(f"📊 **{v}**")
    elif "边际" in v:
        st.warning(f"📊 **{v}**")
    else:
        st.info(f"📊 **{v}**")

    # -------------------------------------------------------------------
    # Per-horizon table
    # -------------------------------------------------------------------
    st.subheader("各 horizon 详细统计")
    rows = []
    for s in stats.horizons:
        rows.append({
            "horizon": f"{s.horizon}d",
            "n": s.n,
            "median": f"{s.median*100:+.2f}%",
            "mean": f"{s.mean*100:+.2f}%",
            "std": f"{s.std*100:.2f}%",
            "win_rate": f"{s.win_rate*100:.0f}%",
            "sharpe": f"{s.sharpe:+.2f}",
            "p25": f"{s.p25*100:+.2f}%",
            "p75": f"{s.p75*100:+.2f}%",
            "min": f"{s.min*100:+.1f}%",
            "max": f"{s.max*100:+.1f}%",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # -------------------------------------------------------------------
    # Distribution histograms (one per horizon)
    # -------------------------------------------------------------------
    st.subheader("收益分布")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.font import setup_chinese_font
    try:
        setup_chinese_font()
    except Exception:
        pass

    n_h = len(stats.horizons)
    if n_h > 0:
        fig, axes = plt.subplots(1, n_h, figsize=(4.5 * n_h, 3.5), squeeze=False)
        for ax, s in zip(axes[0], stats.horizons):
            data = fr[s.horizon].dropna().values * 100
            colors = ["#2ca02c" if v >= 0 else "#d62728" for v in data]
            ax.hist(data, bins=20, color="#7f7f7f", alpha=0.5, edgecolor="white")
            ax.axvline(0, color="black", linewidth=0.6)
            ax.axvline(s.median * 100, color="#1f77b4", linestyle="--",
                       linewidth=1.2, label=f"median {s.median*100:+.1f}%")
            ax.set_title(f"{s.horizon}d  (n={s.n}, win {s.win_rate*100:.0f}%)",
                         fontsize=10)
            ax.set_xlabel("收益 %")
            ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # -------------------------------------------------------------------
    # Raw signal table (small)
    # -------------------------------------------------------------------
    with st.expander(f"信号原始表格 ({len(fr)} 条)", expanded=False):
        st.dataframe(
            (fr * 100).round(2).rename(columns={h: f"{h}d %" for h in fr.columns}),
            width="stretch",
        )

    st.caption(
        "**判读**: Sharpe > 0.5 + win_rate > 55% = 信号有效; "
        "median ≈ 0 = 无预测力; n < 10 = 样本不足。Forward return 与回测 "
        "PnL 是不同的指标 — 前者衡量\"信号能不能选到好时点\", "
        "后者衡量\"完整策略 (入场+出场+止损) 能否盈利\"。"
    )
