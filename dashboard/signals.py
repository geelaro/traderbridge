"""Market state display and today's signals."""

from collections import defaultdict

import pandas as pd
import streamlit as st

from utils import get_logger
from utils.market_state import MarketStateClassifier, MarketRegime, Volatility
from strategy import SIGNAL_LABEL
from analysis.risk_monitor import compute_risk_state, RiskLevel
from data.realtime import get_realtime_vix, get_fear_greed
from live.position_stops import compute_hypothetical_positions

logger = get_logger("dashboard.signals")


# Subtle backgrounds — picked so the badge reads at a glance without
# screaming. Foreground stays dark for accessible contrast on all three.
_RISK_THEME = {
    RiskLevel.GREEN:  {"bg": "#d4edda", "border": "#198754", "fg": "#155724", "label": "GREEN · 低压力"},
    RiskLevel.YELLOW: {"bg": "#fff3cd", "border": "#d39e00", "fg": "#664d03", "label": "YELLOW · 警戒"},
    RiskLevel.RED:    {"bg": "#f8d7da", "border": "#dc3545", "fg": "#721c24", "label": "RED · 高压力"},
}


def render_risk_light(config, target_date, provider):
    """Render market sentiment plus a separate market-stress light."""
    st.subheader("市场状态")

    lookback = config.get("scanner", {}).get("lookback_years", 3)
    start = (pd.Timestamp(target_date) - pd.DateOffset(years=max(lookback, 2))).strftime("%Y-%m-%d")
    end = target_date.isoformat()
    try:
        spy_df = provider.get_daily("SPY", start=start, end=end)
        vix_df = provider.get_daily("^VIX", start=start, end=end)
    except Exception as exc:
        logger.warning("risk light data fetch failed: %s", exc)
        st.info("无法加载 SPY/VIX 数据, 风险灯暂不可用")
        st.divider()
        return
    try:
        qqq_df = provider.get_daily("QQQ", start=start, end=end)
    except Exception as exc:
        logger.warning("QQQ market-state data fetch failed: %s", exc)
        qqq_df = pd.DataFrame()

    # Try to overlay Yahoo realtime VIX (~15-minute delayed) so the badge
    # reflects intraday rather than yesterday's close. Failures fall back
    # to the EOD value from vix_df silently.
    live_vix = get_realtime_vix()
    state = compute_risk_state(
        spy_df, vix_df, realtime_vix=live_vix, qqq_df=qqq_df,
    )
    theme = _RISK_THEME[state.level]

    # Regime / volatility tag — uses the existing MarketStateClassifier
    # so the dashboard surfaces the same regime info that drives
    # SignalGate / StrategyEnsemble internally. Just a label, no extra
    # numbers (risk light already shows MA200/ADX/VIX).
    regime_tag = ""
    try:
        classifier = MarketStateClassifier(spy_df)
        ms_state = classifier.classify()
        regime_label = {
            MarketRegime.TRENDING_UP: "上升趋势",
            MarketRegime.TRENDING_DOWN: "下降趋势",
            MarketRegime.RANGING: "震荡",
            MarketRegime.TRANSITIONAL: "过渡期",
        }
        vol_label = {Volatility.HIGH: "高波动", Volatility.NORMAL: "正常波动", Volatility.LOW: "低波动"}
        regime_tag = (
            f"{regime_label.get(ms_state.regime, ms_state.regime.name)} × "
            f"{vol_label.get(ms_state.volatility, ms_state.volatility.name)}"
        )
    except Exception as exc:
        logger.debug("regime tag unavailable: %s", exc)

    # Native Streamlit alerts keep the pressure state visually consistent
    # with the rest of the dashboard and remain readable in dark mode.
    stress_text = f"{state.emoji} {theme['label']}"
    if regime_tag:
        stress_text += f"  ·  {regime_tag}"
    stress_text += f"  ·  截至 {state.indicators.get('as_of', '?')}"
    if state.level == RiskLevel.GREEN:
        st.success(stress_text)
    elif state.level == RiskLevel.YELLOW:
        st.warning(stress_text)
    else:
        st.error(stress_text)

    overall = state.sentiment_score
    spy_sentiment = state.market_sentiments.get("SPY", {})
    qqq_sentiment = state.market_sentiments.get("QQQ", {})
    st.markdown("**市场情绪温度** · 0 极度超卖 · 50 中性 · 100 极度超买")
    s1, s2, s3 = st.columns(3)
    s1.metric("大盘整体", f"{overall:.1f}" if overall is not None else "—", state.sentiment_label)
    s2.metric("SPY", f"{spy_sentiment.get('score', 0):.1f}" if spy_sentiment.get("score") is not None else "—", spy_sentiment.get("label", "数据不足"))
    s3.metric("QQQ", f"{qqq_sentiment.get('score', 0):.1f}" if qqq_sentiment.get("score") is not None else "—", qqq_sentiment.get("label", "数据不可用"))

    ind = state.indicators
    st.markdown("**市场参考**")
    ref_cols = st.columns(5)
    ref_cols[0].metric("SPY", f"${ind.get('spy_close', 0):.2f}")
    ref_cols[1].metric("SPY 5日", f"{ind.get('spy_5d_return_pct', 0):+.2f}%")
    ref_cols[2].metric("SPY 20日", f"{ind.get('spy_20d_return_pct', 0):+.2f}%")
    vix_src = ind.get("vix_source", "eod")
    vix_label = "VIX (实时)" if vix_src == "realtime" else "VIX (昨收)"
    ref_cols[3].metric(vix_label, f"{ind.get('vix', 0):.2f}", help="<18 低波, 18-30 偏高, >30 极端恐慌")

    # CNN Fear & Greed Index — sentiment 互补指标, 不参与压力判定。
    fg = get_fear_greed()
    if fg is not None:
        fg_score, fg_rating = fg
        rating_zh = {
            "extreme fear": "极度恐慌",
            "fear": "恐慌",
            "neutral": "中性",
            "greed": "贪婪",
            "extreme greed": "极度贪婪",
        }.get(fg_rating, fg_rating)
        ref_cols[4].metric("F&G (CNN)", f"{fg_score:.0f}", delta=rating_zh, delta_color="off")
    else:
        ref_cols[4].metric("F&G (CNN)", "—", delta="数据不可用", delta_color="off")

    # Compact comparison table keeps the two market lenses visible together.
    levels = state.market_states
    risk_label = {"green": "GREEN · 低压力", "yellow": "YELLOW · 警戒", "red": "RED · 高压力"}
    market_rows = []
    for symbol, prefix in (("SPY", "spy"), ("QQQ", "qqq")):
        market = levels.get(symbol, {})
        market_rows.append({
            "市场": symbol,
            "压力": risk_label.get(market.get("level"), "数据不可用"),
            "情绪": f"{market.get('sentiment_score', 0):.1f} · {market.get('sentiment_label', '数据不可用')}" if market else "数据不可用",
            "收盘价": f"${ind.get(f'{prefix}_close', 0):.2f}" if ind.get(f"{prefix}_close") is not None else "—",
            "MA200乖离": f"{ind.get(f'{prefix}_vs_ma200_pct', 0):+.1f}%" if ind.get(f"{prefix}_vs_ma200_pct") is not None else "—",
            "5日": f"{ind.get(f'{prefix}_5d_return_pct', 0):+.2f}%" if ind.get(f"{prefix}_5d_return_pct") is not None else "—",
            "20日": f"{ind.get(f'{prefix}_20d_return_pct', 0):+.2f}%" if ind.get(f"{prefix}_20d_return_pct") is not None else "—",
            "RSI14": f"{ind.get(f'{prefix}_rsi14', 0):.1f}" if ind.get(f"{prefix}_rsi14") is not None else "—",
        })
    st.dataframe(pd.DataFrame(market_rows), hide_index=True, use_container_width=True)

    with st.expander("查看判定依据与情绪分解"):
        st.markdown("**压力判定**")
        if state.reasons:
            for reason in state.reasons:
                st.markdown(f"- {reason}")
        else:
            st.caption("暂无压力判定依据")
        st.markdown("**情绪分解**")
        sentiment_rows = []
        for symbol, item in (("SPY", spy_sentiment), ("QQQ", qqq_sentiment)):
            components = item.get("components", {})
            sentiment_rows.append({
                "市场": symbol,
                "RSI贡献": f"{components.get('rsi', 0):.1f}" if components else "—",
                "MA200乖离贡献": f"{components.get('ma200_extension', 0):.1f}" if components else "—",
                "20日动量贡献": f"{components.get('momentum_20d', 0):.1f}" if components else "—",
            })
        st.dataframe(pd.DataFrame(sentiment_rows), hide_index=True, use_container_width=True)

    st.divider()


def render_market_state(config, target_date, provider):
    ms_cfg = config.get("market_state", {})
    if not ms_cfg.get("enabled", False):
        return

    st.subheader("市场状态")

    proxy_sym = ms_cfg.get("proxy_symbol", "SPY")
    lookback = config.get("scanner", {}).get("lookback_years", 3)
    ms_start = (pd.Timestamp(target_date) - pd.DateOffset(years=lookback)).strftime("%Y-%m-%d")
    ms_df = provider.get_daily(proxy_sym, start=ms_start, end=target_date.isoformat())

    if ms_df is not None and not ms_df.empty:
        classifier = MarketStateClassifier(ms_df)
        state = classifier.classify()

        regime = state.regime
        vol = state.volatility
        regime_color = {
            MarketRegime.TRENDING_UP: "#2ca02c",
            MarketRegime.TRENDING_DOWN: "#d62728",
            MarketRegime.RANGING: "#ff7f0e",
            MarketRegime.TRANSITIONAL: "#7f7f7f",
        }
        regime_label = {
            MarketRegime.TRENDING_UP: "上升趋势",
            MarketRegime.TRENDING_DOWN: "下降趋势",
            MarketRegime.RANGING: "震荡",
            MarketRegime.TRANSITIONAL: "过渡期",
        }
        vol_label = {Volatility.HIGH: "高波动", Volatility.NORMAL: "正常", Volatility.LOW: "低波动"}

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("市场状态", regime_label.get(regime, regime.name))
        c2.metric("波动率", vol_label.get(vol, vol.name))
        c3.metric("ADX", f"{state.adx:.1f}")
        c4.metric("MA20", f"{state.ma20:.2f}")
        c5.metric("MA200", f"{state.ma200:.2f}")
        c6.metric("BB带宽分位", f"{state.bb_width_pct:.0f}%")

        st.markdown(
            f"<span style='display:inline-block;padding:4px 12px;border-radius:8px;"
            f"background:{regime_color.get(regime, '#7f7f7f')};color:white;font-weight:bold;'>"
            f"{regime_label.get(regime, regime.name)} × {vol_label.get(vol, vol.name)}</span>",
            unsafe_allow_html=True,
        )
    else:
        st.info(f"市场状态: {proxy_sym} 数据不可用")

    st.divider()


def render_todays_signals(config, target_date, provider, cache):
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"今日信号 ({target_date})")

        from daily import scan_day

        results = scan_day(config, target_date=target_date.isoformat(),
                           provider=provider, cache=cache)

        active_signals = [r for r in results if r["signal"] != 0]
        if active_signals:
            grouped: dict[str, list] = defaultdict(list)
            for r in active_signals:
                grouped[r["symbol"]].append(r)

            for sym, sigs in sorted(grouped.items()):
                parts = []
                has_buy = any(s["signal"] == 1 for s in sigs)
                has_sell = any(s["signal"] == -1 for s in sigs)
                price = sigs[0]["price"]
                for s in sigs:
                    tag = "★" if s["strategy"] == next(
                        (item.get("active", "") for item in config.get("watchlist", [])
                         if item["symbol"] == sym), ""
                    ) else ""
                    parts.append(f"{s['strategy']}{tag} {SIGNAL_LABEL[s['signal']]}")
                line = f"{sym} @ ${price:.2f}:  " + "  |  ".join(parts)
                if has_sell:
                    st.error(line)
                elif has_buy:
                    st.success(line)
                else:
                    st.info(line)
        else:
            st.info("今日无买入/卖出信号")

    with col2:
        st.subheader("策略分布")
        active_counts: dict[str, int] = {}
        for item in config.get("watchlist", []):
            a = item.get("active", "-")
            active_counts[a] = active_counts.get(a, 0) + 1

        for strat, count in sorted(active_counts.items(), key=lambda x: -x[1]):
            st.metric(label=strat, value=f"{count} 个标的")

        st.metric(label="总组合数", value=f"{len(results)} 策略×标的")


# ---------------------------------------------------------------------------
# Phase 2.A — signal detail panel
# ---------------------------------------------------------------------------


# Per-direction copy used in the detail card. Keeps the rendering function
# small and lets non-developers tweak the messaging in one place.
_DIRECTION = {
    1: {"label": "买入", "color": "#28a745", "icon": "📈",
        "stop_factor": -2.0, "target_factor": 3.0,
        "stop_word": "止损", "target_word": "止盈"},
    -1: {"label": "卖出", "color": "#dc3545", "icon": "📉",
         "stop_factor": 2.0, "target_factor": -3.0,
         "stop_word": "止损(空)", "target_word": "止盈(空)"},
}


def render_signal_detail(config, target_date, provider, cache):
    """Today's signal summary + per-symbol expandable detail cards.

    Combines the previous compact list (render_todays_signals) and the
    detail panel into one section to avoid duplicate displays. Top row is
    a metric-style summary; below it, each symbol with a non-zero signal
    gets an expander showing trigger indicators, suggested stop / target,
    and other strategies' signals on the same symbol.
    """
    from daily import scan_day

    st.subheader(f"今日信号 ({target_date})")

    results = scan_day(config, target_date=target_date.isoformat(),
                       provider=provider, cache=cache)

    # Summary metrics — always show, even with no signals
    active = [r for r in results if r["signal"] != 0]
    n_buys = sum(1 for r in active if r["signal"] == 1)
    n_sells = sum(1 for r in active if r["signal"] == -1)
    n_symbols = len({r["symbol"] for r in active})
    n_watchlist = len(config.get("watchlist", []))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📈 买入信号", n_buys)
    m2.metric("📉 卖出信号", n_sells)
    m3.metric("有信号标的", f"{n_symbols} / {n_watchlist}")
    m4.metric("总策略×标的", len(results))

    if not active:
        st.info("今日无买入/卖出信号 — 等待")
        _render_proximity(results, default_expanded=True)
        st.divider()
        return

    # Group active signals by symbol; collect all-strategy snapshot for context
    grouped: dict[str, list] = defaultdict(list)
    for r in active:
        grouped[r["symbol"]].append(r)
    all_by_symbol: dict[str, list] = defaultdict(list)
    for r in results:
        all_by_symbol[r["symbol"]].append(r)

    # Active strategy lookup (so we can mark which signal is the "primary" one)
    active_map = {
        item["symbol"]: item.get("active", "")
        for item in config.get("watchlist", [])
    }

    for sym, sigs in sorted(grouped.items()):
        # The "primary" signal for display is the active strategy's signal
        # if present, otherwise the first signal from the monitors.
        primary = next(
            (s for s in sigs if s["strategy"] == active_map.get(sym, "")),
            sigs[0],
        )
        direction = primary["signal"]
        cfg = _DIRECTION.get(direction)
        if cfg is None:
            continue
        price = primary["price"]
        atr = primary["atr"]
        stop_price = price + cfg["stop_factor"] * atr if atr > 0 else 0
        target_price = price + cfg["target_factor"] * atr if atr > 0 else 0
        stop_pct = (stop_price - price) / price * 100 if price > 0 and atr > 0 else 0
        target_pct = (target_price - price) / price * 100 if price > 0 and atr > 0 else 0

        # bar_date is the actual K-line date the price comes from. Display
        # it inline so users can tell at a glance whether "今日信号" is
        # using yesterday's close (typical pre-US-close) vs today's close.
        bar_date_full = str(primary.get("bar_date", "") or "")
        bar_date_short = bar_date_full[5:10] if len(bar_date_full) >= 10 else "—"

        title = (
            f"{cfg['icon']} {sym}  {cfg['label']}  @ ${price:.2f} ({bar_date_short})  "
            f"({primary['strategy']})"
        )
        with st.expander(title, expanded=False):
            if bar_date_full:
                st.caption(f"📅 数据来自 **{bar_date_full}** 的日 K 收盘价")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("触发价", f"${price:.2f}")
            c2.metric("ATR(14)", f"${atr:.2f}" if atr > 0 else "—")
            if atr > 0:
                c3.metric(
                    f"建议 {cfg['stop_word']}",
                    f"${stop_price:.2f}",
                    f"{stop_pct:+.2f}%",
                )
                c4.metric(
                    f"建议 {cfg['target_word']}",
                    f"${target_price:.2f}",
                    f"{target_pct:+.2f}%",
                )
            else:
                c3.metric(f"建议 {cfg['stop_word']}", "—")
                c4.metric(f"建议 {cfg['target_word']}", "—")

            # Indicator snapshot — the strategy's own context, filtered to
            # human-readable values only. Strategy authors put whatever
            # technicals matter in the indicators dict, so this is the
            # truest representation of "why did the signal fire".
            ind = primary.get("indicators") or {}
            if ind:
                ind_items = [
                    (k, v) for k, v in ind.items()
                    if isinstance(v, (int, float)) and not str(k).startswith("_")
                ]
                # Show top 6 indicators (most strategies use 3-6 indicators)
                if ind_items:
                    st.markdown("**指标快照**")
                    rows = []
                    for k, v in ind_items[:8]:
                        rows.append({"指标": k, "数值": f"{v:.4g}"})
                    st.dataframe(pd.DataFrame(rows), use_container_width=True,
                                 hide_index=True)

            # Other strategies monitoring this symbol — agreement check
            others = [r for r in all_by_symbol[sym] if r["strategy"] != primary["strategy"]]
            if others:
                agree_lines = []
                for r in others:
                    sig_label = SIGNAL_LABEL.get(r["signal"], "—")
                    tag = "(active)" if r["strategy"] == active_map.get(sym, "") else "(monitor)"
                    agree_lines.append(f"{r['strategy']} {tag}: {sig_label}")
                st.markdown("**其它策略观察**: " + "  ·  ".join(agree_lines))

    # Proximity warnings — even when signals fire, near-trigger info on
    # other symbols is useful. Collapsed by default so it doesn't crowd
    # the signal cards.
    _render_proximity(results, default_expanded=False)
    st.divider()


# ---------------------------------------------------------------------------
# Proximity-to-trigger warnings (analysis/proximity)
# ---------------------------------------------------------------------------


_PROX_ICON = {"alert": "🔴", "warn": "🟡", "info": "⚪"}
_PROX_DIR_ARROW = {"bullish": "↑", "bearish": "↓", "neutral": "·"}


def _render_proximity(scan_results, default_expanded: bool = False) -> None:
    """Render the 'near-trigger' warning panel, grouped by symbol.

    All strategies' warnings on the same symbol are collapsed into one
    card (the symbol is the user's mental unit).  Sub-grouped by strategy
    inside the card so you can still see which detector fired what.
    ``default_expanded`` controls whether the expander starts open —
    True for the no-signal fallback, False when shown alongside actual
    signals.
    """
    from analysis.proximity import proximity_summary_by_symbol

    summary = proximity_summary_by_symbol(scan_results)
    if not summary:
        return

    n_alert = sum(1 for r in summary if r["max_level"] == "alert")
    n_warn = sum(1 for r in summary if r["max_level"] == "warn")
    badge = f"🔴 {n_alert} · 🟡 {n_warn}" if (n_alert or n_warn) else f"{len(summary)} 条"
    title = f"📡 接近触发预警 ({badge})"

    with st.expander(title, expanded=default_expanded):
        st.caption(
            "策略未发实际信号, 但部分指标已接近触发阈值. "
            "🔴 alert = 1-2 个 bar 内可能触发, 🟡 warn = 几个 bar 后可能触发. "
            "方向箭头: ↑ 多头机会, ↓ 空头/退出机会."
        )
        for row in summary:
            icon = _PROX_ICON.get(row["max_level"], "")
            header = (f"{icon}  **{row['symbol']}**  ·  "
                      f"{row['n_warnings']} 项  ·  "
                      f"{row['n_strategies']} 个策略")
            st.markdown(header)
            for strat, warnings in row["by_strategy"].items():
                st.markdown(f"&nbsp;&nbsp;*{strat}*", unsafe_allow_html=True)
                for w in warnings:
                    ic = _PROX_ICON.get(w["level"], "")
                    arrow = _PROX_DIR_ARROW.get(w["direction"], "")
                    st.markdown(
                        f"&nbsp;&nbsp;&nbsp;&nbsp;{ic} {arrow} {w['message']}",
                        unsafe_allow_html=True,
                    )


# ---------------------------------------------------------------------------
# Phase 2.B — simulated position watch (trailing-stop proximity warning)
# ---------------------------------------------------------------------------


def render_position_watch(config, target_date, provider):
    """Display 'simulated positions' from each symbol's active strategy and
    flag any within 5% of their trailing stop.

    Calculation lives in :func:`live.position_stops.compute_hypothetical_positions`
    so the daemon's alerter can reuse it; this function only renders.
    """
    st.subheader("假设持仓监控")
    st.caption(
        "按 active 策略最近一次未平 buy 信号假设入场, "
        "用 Chandelier 移动止损估算距离. 仅用于减仓预警, 不替代实盘持仓数据."
    )

    rows = compute_hypothetical_positions(config, target_date, provider)

    if not rows:
        st.info("当前无假设持仓 — active 策略最近的信号均已平仓 (或从未触发买入)")
        st.divider()
        return

    df_display = pd.DataFrame([{
        "标的": r["symbol"],
        "策略": r["strategy"],
        "入场日": r["entry_date"],
        "入场价": f"${r['entry_price']:.2f}",
        "当前价": f"${r['current_price']:.2f}",
        "浮盈%": f"{r['pnl_pct']:+.2f}%",
        "移动止损": f"${r['stop_price']:.2f}",
        "距止损%": f"{r['distance_pct']:.2f}%",
        "持仓天": r["days_held"],
    } for r in rows])
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    danger = [r for r in rows if r["distance_pct"] < 5]
    if danger:
        for r in danger:
            st.warning(
                f"⚠ {r['symbol']} ({r['strategy']}) 距移动止损仅 "
                f"{r['distance_pct']:.2f}% (止损位 ${r['stop_price']:.2f})"
            )
    else:
        st.success(f"全部 {len(rows)} 个假设持仓距移动止损 ≥ 5%, 暂无减仓预警")

    st.divider()


# ---------------------------------------------------------------------------
# Risk alert history — retrospective view of fired alerts
# ---------------------------------------------------------------------------


_ALERT_TYPE_LABEL = {
    "risk_light": "🔴 风险灯",
    "vix_spike":  "📈 VIX 突破",
    "position_stop": "⚠ 持仓临近止损",
}


def _format_alert_detail(alert_type: str, payload: dict) -> str:
    """One-line human-readable summary for a history row."""
    if alert_type == "risk_light":
        ind = payload.get("indicators") or {}
        spy = ind.get("spy_close")
        vix = ind.get("vix")
        adx = ind.get("spy_adx")
        bits = []
        if spy is not None:
            bits.append(f"SPY {spy}")
        if vix is not None:
            bits.append(f"VIX {vix}")
        if adx is not None:
            bits.append(f"ADX {adx}")
        reasons = payload.get("reasons") or []
        head = " | ".join(bits) if bits else ""
        return (head + " — " + "; ".join(reasons)) if reasons else head
    if alert_type == "vix_spike":
        return f"VIX {payload.get('value', '-')} ≥ {payload.get('threshold', '-')}"
    if alert_type == "position_stop":
        return (f"{payload.get('symbol', '?')} "
                f"距止损 {payload.get('distance_pct', 0):.2f}% "
                f"(${payload.get('current_price', 0):.2f} → ${payload.get('stop_price', 0):.2f})")
    return ""


def render_alert_history(cache, days: int = 30):
    """Show the alert audit trail and breakdown by type / day.

    Reads :class:`StateStore.load_alert_history`; renders:
    - summary metrics (total alerts, by-type counts)
    - daily count bar chart
    - chronological table with one-line detail per row
    """
    st.header("风险告警历史")
    st.caption(
        "回看过去 N 天的风险灯/VIX/持仓告警时间线 + 当时指标. "
        "用于验证阈值合不合理 — 太密说明假警多, 太疏说明阈值太宽松."
    )

    col_days, _ = st.columns([1, 4])
    days = col_days.number_input("回看天数", min_value=1, max_value=365, value=days, step=1)

    rows = cache.load_alert_history(days=int(days))
    if not rows:
        st.info(f"过去 {days} 天无风险告警记录")
        return

    # Summary metrics
    by_type = defaultdict(int)
    for r in rows:
        by_type[r["alert_type"]] += 1
    cols = st.columns(4)
    cols[0].metric("总告警数", len(rows))
    cols[1].metric("风险灯 RED", by_type.get("risk_light", 0))
    cols[2].metric("VIX 突破", by_type.get("vix_spike", 0))
    cols[3].metric("持仓临近止损", by_type.get("position_stop", 0))

    # Daily count bar chart
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["ts"]).dt.date
    daily = df.groupby(["date", "alert_type"]).size().unstack(fill_value=0)
    if not daily.empty:
        st.subheader("每日告警分布")
        st.bar_chart(daily)

    # Chronological table
    st.subheader("时间线 (最新在前)")
    display_rows = [{
        "时间": r["ts"].replace("T", " "),
        "类型": _ALERT_TYPE_LABEL.get(r["alert_type"], r["alert_type"]),
        "详情": _format_alert_detail(r["alert_type"], r["payload"]),
    } for r in rows]
    st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)
    st.divider()
