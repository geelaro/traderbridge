"""Historical analog tab — find past drops similar to today and show forward returns.

The user-facing question is: "X dropped Y% today on macro release Z; what
happened next time?" This tab surfaces:

1. Today's drop magnitude and macro tag.
2. All historical drops past the threshold, aggregated by macro tag.
3. Top-N past events most similar to today (same tag first, then magnitude).
4. Forward-return distribution restricted to same-tag events.
"""

import pandas as pd
import streamlit as st

from utils import get_logger
from analysis.historical_analog import (
    find_drop_events, analog_summary_by_tag, find_closest_analogs,
)
from analysis.macro_calendar import macro_tag

logger = get_logger("dashboard.historical_analog")


_HORIZONS = (3, 5, 10, 20, 60)
_FWD_COLS = [f"fwd_{n}d" for n in _HORIZONS]


def render_historical_analog(selected_symbol, target_date, provider):
    st.subheader(f"历史类比: {selected_symbol}")
    st.caption(
        "找历史上单日跌幅相似的天,看后续 3/5/10/20/60 天反弹分布。"
        "同 macro tag (NFP/CPI) 优先,其次按跌幅磁吸。"
    )

    col_a, col_b, col_c = st.columns(3)
    drop_threshold = col_a.slider(
        "跌幅阈值 (%)", min_value=-10.0, max_value=-1.0,
        value=-3.0, step=0.5,
        help="只看单日跌幅 ≤ 此值的历史事件",
    )
    top_n = col_b.slider("最相似 N 天", 5, 30, 10)
    lookback_years = col_c.slider("历史回看 (年)", 5, 25, 20)

    @st.cache_data(ttl=3600, show_spinner="获取数据中...")
    def _get_prices(symbol, start, end):
        df = provider.get_daily(symbol, start=start, end=end)
        return df["Close"] if df is not None and not df.empty else None

    start = (pd.Timestamp(target_date) - pd.DateOffset(years=lookback_years)).strftime("%Y-%m-%d")
    end = target_date.isoformat()
    prices = _get_prices(selected_symbol, start, end)
    if prices is None or len(prices) < 2:
        st.info(f"{selected_symbol} 历史数据不足")
        return

    today_close = float(prices.iloc[-1])
    today_prev = float(prices.iloc[-2])
    today_drop = (today_close / today_prev - 1) * 100
    today_date = prices.index[-1].date()
    today_tag = macro_tag(prices.index[-1]) or "OTHER"

    m1, m2, m3 = st.columns(3)
    m1.metric(f"{today_date} close", f"${today_close:.2f}")
    m2.metric("当日跌幅", f"{today_drop:+.2f}%")
    m3.metric("macro tag", today_tag)

    if today_drop > drop_threshold:
        st.info(
            f"今日跌幅 {today_drop:+.2f}% 未达阈值 {drop_threshold:.1f}%, "
            "下方仅展示历史样本分布"
        )

    events = find_drop_events(prices, drop_threshold_pct=drop_threshold,
                              horizons=_HORIZONS)
    if events.empty:
        st.warning(f"历史上 {selected_symbol} 没有 ≤ {drop_threshold:.1f}% 的跌幅事件")
        return

    st.markdown(f"#### 历史 ≤{drop_threshold:.1f}% 跌幅事件总览 ({len(events)} 起)")
    summary = analog_summary_by_tag(events)
    num_cols = summary.select_dtypes("number").columns
    st.dataframe(
        summary.style.format("{:.2f}", subset=num_cols),
        hide_index=True,
        width="stretch",
    )

    st.markdown(f"#### 跟今日最相似的 top {top_n} 次")
    top_events = find_closest_analogs(
        events, today_drop_pct=today_drop, today_macro_tag=today_tag, top_n=top_n,
    )
    fmt = {
        "prev_close": "{:.2f}", "close": "{:.2f}", "drop_pct": "{:+.2f}",
        **{c: "{:+.2f}" for c in _FWD_COLS},
    }
    st.dataframe(
        top_events.style.format(fmt, na_rep="—"),
        width="stretch",
    )

    if today_tag != "OTHER":
        tag_events = events[events["macro_tag"] == today_tag]
        # Exclude today's bar so the stats are about *past* analogs.
        tag_events = tag_events[
            pd.DatetimeIndex(tag_events.index).date != today_date
        ]
        if not tag_events.empty:
            st.markdown(
                f"#### {today_tag} 当天 ≤{drop_threshold:.1f}% 跌幅样本反弹分布 "
                f"(n={len(tag_events)})"
            )
            stats = pd.DataFrame({
                "窗口": [f"{n}d" for n in _HORIZONS],
                "均值%": [tag_events[c].mean() for c in _FWD_COLS],
                "中位%": [tag_events[c].median() for c in _FWD_COLS],
                "胜率%": [(tag_events[c] > 0).mean() * 100 for c in _FWD_COLS],
                "样本": [tag_events[c].notna().sum() for c in _FWD_COLS],
            })
            st.dataframe(
                stats.style.format({
                    "均值%": "{:+.2f}", "中位%": "{:+.2f}", "胜率%": "{:.1f}",
                }),
                hide_index=True,
                width="stretch",
            )
            if len(tag_events) < 10:
                st.caption(
                    f"⚠ {today_tag} 样本仅 {len(tag_events)} 起,"
                    "统计噪声大,请结合个体宏观背景判读"
                )
