"""Dashboard tab: historical signal log — browsable view of signal_history.

What this answers
------------------
"今日信号" (dashboard/signals.py::render_signal_detail) runs a scan every
time the dashboard loads and, via daily.scan_day -> SignalScanner, persists
one row per symbol x strategy into the signal_history table (see
data/cache.py). That table has been quietly accumulating for months but had
no UI — the only way to see it was the CLI (`daily.py --history`).

This tab is a read-only browser over that existing table. It does not
recompute anything: no new persistence, no re-running strategies over full
price history (that's a different question, answered by
signal_effectiveness.py's forward-return analysis).

Dedup note
----------
Weekly strategies re-persist the *same* signal on every daily scan until the
weekly bar rolls over and produces a new cross (SignalScanner writes on every
scan, not only on change). Left un-deduped, one death cross confirmed on 4
consecutive scan days would look like 4 separate sell events. We collapse by
(symbol, strategy, bar_date, signal) and show first-seen / last-confirmed
scan_date plus a confirmation count instead.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from data.protocol import classify_symbol
from strategy import STRATEGY_MAP

_WINDOW_OPTIONS = {"7天": 7, "30天": 30, "90天": 90, "180天": 180, "365天": 365, "全部": None}
_DIR_OPTIONS = {"全部": None, "买入": 1, "卖出": -1}
_DIR_ICON = {1: "📈 买入", -1: "📉 卖出", 0: "—"}


def render_signal_history(cache, config):
    """Render the historical-signal-log tab."""
    st.header("历史信号")
    st.caption(
        "浏览\"今日信号\"每次扫描时写入 signal_history 表的历史记录 — "
        "纯读取已有数据, 不重新对全历史价格跑策略。weekly 策略的同一次信号"
        "会在多天扫描中被重复确认, 已按 (标的, 策略, K线日期, 信号) 去重, "
        "下表的\"确认天数\"就是这个重复计数。"
    )

    symbols = [item["symbol"] for item in config.get("watchlist", [])]
    strategy_options = list(STRATEGY_MAP.keys())

    fc1, fc2, fc3, fc4, fc5 = st.columns([1, 1, 1, 1, 1.2])
    window_label = fc1.selectbox("时间窗", list(_WINDOW_OPTIONS.keys()), index=2)
    symbol_choice = fc2.selectbox("标的", ["全部"] + symbols)
    strategy_choice = fc3.selectbox("策略", ["全部"] + strategy_options)
    direction_choice = fc4.selectbox("方向", list(_DIR_OPTIONS.keys()))
    only_nonzero = fc5.checkbox("只看有效信号(排除0)", value=True)

    since = None
    days = _WINDOW_OPTIONS[window_label]
    if days is not None:
        since = (pd.Timestamp.today() - pd.Timedelta(days=days)).date().isoformat()

    rows = cache.query_signals(scan_date=since)
    if not rows:
        st.info("signal_history 表为空 — 打开过\"今日信号\"或跑过 daily.py 之后才会有数据")
        return

    df = pd.DataFrame(rows)

    if only_nonzero:
        df = df[df["signal"] != 0]
    if symbol_choice != "全部":
        df = df[df["symbol"] == symbol_choice]
    if strategy_choice != "全部":
        df = df[df["strategy"] == strategy_choice]
    direction = _DIR_OPTIONS[direction_choice]
    if direction is not None:
        df = df[df["signal"] == direction]

    if df.empty:
        st.info("当前筛选条件下无匹配记录")
        return

    # ------------------------------------------------------------------
    # Dedup: one row per (symbol, strategy, bar_date, signal) event —
    # collapse repeated daily re-confirmations of the same underlying bar.
    # ------------------------------------------------------------------
    grouped = (
        df.groupby(["symbol", "strategy", "bar_date", "signal"])
        .agg(
            price=("price", "last"),
            atr=("atr", "last"),
            first_seen=("scan_date", "min"),
            last_confirmed=("scan_date", "max"),
            confirmations=("scan_date", "nunique"),
            _latest_id=("id", "max"),
        )
        .reset_index()
        .sort_values("bar_date", ascending=False)
    )

    n_buy = int((grouped["signal"] == 1).sum())
    n_sell = int((grouped["signal"] == -1).sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("信号事件数(去重后)", len(grouped))
    m2.metric("覆盖标的数", grouped["symbol"].nunique())
    m3.metric("📈 买入事件", n_buy)
    m4.metric("📉 卖出事件", n_sell)

    def _fmt_price(row):
        if pd.isna(row["price"]):
            return "—"
        prefix = "¥" if classify_symbol(row["symbol"]) == "cn" else "$"
        return f"{prefix}{row['price']:.2f}"

    display_df = pd.DataFrame({
        "K线日期": grouped["bar_date"],
        "标的": grouped["symbol"],
        "策略": grouped["strategy"],
        "方向": grouped["signal"].map(_DIR_ICON),
        "价格": grouped.apply(_fmt_price, axis=1),
        "ATR": grouped["atr"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—"),
        "首次探测": grouped["first_seen"],
        "最近确认": grouped["last_confirmed"],
        "确认天数": grouped["confirmations"],
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # Raw indicators expander — look up the latest row for a chosen event.
    # ------------------------------------------------------------------
    with st.expander("查看原始指标快照 (调试用)"):
        options = [
            f"{r.bar_date}  {r.symbol}  {r.strategy}  {_DIR_ICON.get(r.signal, r.signal)}"
            for r in grouped.itertuples()
        ]
        sel = st.selectbox("选一条", options) if options else None
        if sel is not None:
            pos = options.index(sel)
            row_id = int(grouped.iloc[pos]["_latest_id"])
            raw = df[df["id"] == row_id]
            if not raw.empty:
                indicators_raw = raw.iloc[0].get("indicators") or "{}"
                try:
                    indicators = json.loads(indicators_raw)
                except (TypeError, ValueError):
                    indicators = {}
                st.json(indicators)
