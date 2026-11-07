"""Decision summary — one-glance "should I trade today" card.

Aggregates four existing analyses into a single verdict: risk light +
regime, today's signals, gate-rejected signals, and portfolio risk vs
the latest earlier snapshot. ``build_decision_summary`` is pure compute (no Streamlit
calls) so it can be unit-tested without a UI harness; ``render_decision_
summary`` only renders an already-built summary dict.

v1 scope — rejected-signal detection
-------------------------------------
Only ``SignalGate``'s regime-filter (``RANGING_BLOCK_*``/``TRENDING_
BLOCK_*``) and pause (``PAUSE_*``) reasons are evaluated — both need no
position sizing. ``EXPOSURE_CAP_EXCEEDED`` (needs a real order quantity)
and ``ORPHAN_BUY_BLOCKED`` (a live_trader-only concept) are not
evaluated: the dashboard has no live order sizing or orphan-position
context to feed them, and faking one would be misleading rather than
useful.

``trading_paused``/``pause_reason`` are read from the live operational
snapshot, with the individual ``risk_state`` keys as a compatibility
fallback when no current-day snapshot exists.

v1 scope — risk delta vs the latest earlier snapshot
-----------------------------------------------------
Portfolio risk history is stored through a dedicated cache API in
``portfolio_risk_snapshots``. It stays separate from ``risk_state``,
which is reserved for live operational guardrails. Deltas use the latest
earlier snapshot, so weekends and days without a dashboard visit do not
break the comparison.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from analysis.risk_monitor import compute_risk_state, RiskLevel
from analysis.what_if import compare_portfolios
from daily import scan_day
from data.realtime import get_realtime_vix
from utils import get_logger
from utils.market_state import MarketStateClassifier
from utils.sectors import DEFAULT_SECTORS
from utils.signal_gate import SignalGate

logger = get_logger("dashboard.decision_summary")


def build_decision_summary(config: dict, target_date, provider, cache) -> dict:
    """Aggregate risk/signal/gate/delta data into one decision-summary dict.

    Every section is best-effort — a failure in one (e.g. VIX fetch down)
    degrades that section to a neutral default and is recorded in
    ``errors``, it never raises out of this function.
    """
    end_ts = pd.Timestamp(target_date)
    errors: list[str] = []

    (
        risk_level, risk_reasons, market_state, top_risk_level,
        top_risk_reasons, sentiment_score, sentiment_label,
    ) = _risk_and_regime(
        config, end_ts, provider, errors,
    )
    signals = _today_signals(config, end_ts, provider, cache, errors)
    rejected = _rejected_signals(config, market_state, signals, errors)
    risk_snapshot, risk_delta = _risk_delta(config, target_date, provider, cache, errors)

    live_risk = _live_risk_snapshot(cache, as_of=end_ts.date().isoformat())
    trading_paused = (
        bool(live_risk.get("paused"))
        if live_risk is not None
        else cache.load_risk_state("trading_paused") == "True"
    )
    pause_reason = (
        live_risk.get("pause_reason", "")
        if live_risk is not None
        else cache.load_risk_state("pause_reason") or ""
    )

    verdict_code, verdict_text = _verdict(
        risk_level, trading_paused, pause_reason, signals, rejected,
        top_risk_level=top_risk_level,
    )

    return {
        "verdict_code": verdict_code,
        "verdict_text": verdict_text,
        "risk_level": risk_level,
        "risk_reasons": risk_reasons,
        "top_risk_level": top_risk_level,
        "top_risk_reasons": top_risk_reasons,
        "sentiment_score": sentiment_score,
        "sentiment_label": sentiment_label,
        "regime": market_state.regime if market_state else None,
        "volatility": market_state.volatility if market_state else None,
        "signals": signals,
        "signals_total": len(signals),
        "rejected": rejected,
        "rejected_count": len(rejected),
        "risk_snapshot": risk_snapshot,
        "risk_delta": risk_delta,
        "live_risk": live_risk,
        "trading_paused": trading_paused,
        "pause_reason": pause_reason,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _risk_and_regime(config, end_ts, provider, errors):
    lookback = config.get("scanner", {}).get("lookback_years", 3)
    start = (end_ts - pd.DateOffset(years=max(lookback, 2))).strftime("%Y-%m-%d")
    end = end_ts.strftime("%Y-%m-%d")
    try:
        spy_df = provider.get_daily("SPY", start=start, end=end)
        vix_df = provider.get_daily("^VIX", start=start, end=end)
        try:
            qqq_df = provider.get_daily("QQQ", start=start, end=end)
        except Exception as exc:
            logger.warning("decision_summary: QQQ 数据不可用: %s", exc)
            qqq_df = pd.DataFrame()
        live_vix = get_realtime_vix()
        state = compute_risk_state(
            spy_df, vix_df, realtime_vix=live_vix, qqq_df=qqq_df,
        )
        market_state = MarketStateClassifier(spy_df).classify()
        return (
            state.level, state.reasons, market_state,
            state.top_risk_level, state.top_risk_reasons,
            state.sentiment_score, state.sentiment_label,
        )
    except Exception as exc:
        logger.warning("decision_summary: risk/regime 计算失败: %s", exc)
        errors.append("风险灯/regime 数据不可用")
        return None, [], None, "unknown", [], None, "unknown"


def _today_signals(config, end_ts, provider, cache, errors):
    try:
        results = scan_day(
            config, target_date=end_ts.strftime("%Y-%m-%d"),
            provider=provider, cache=cache,
        )
        return [r for r in results if r.get("signal", 0) != 0]
    except Exception as exc:
        logger.warning("decision_summary: 今日信号扫描失败: %s", exc)
        errors.append("今日信号数据不可用")
        return []


def _rejected_signals(config, market_state, signals, errors):
    """Run today's non-zero signals through SignalGate — see module
    docstring for the v1 scope (regime + pause only, no exposure cap)."""
    if not signals:
        return []
    try:
        ms_cfg = config.get("market_state", {})
        gate = SignalGate(
            ms_enabled=ms_cfg.get("enabled", False),
            market_state=market_state,
            vol_high_scalar=ms_cfg.get("vol_high_scalar", 0.7),
        )
        rejected = []
        for sig in signals:
            if sig["signal"] == 1:
                ok, reason = gate.allow_buy(sig, {}, None)
            else:
                ok, reason = gate.allow_sell(sig)
            if not ok:
                rejected.append({
                    "symbol": sig["symbol"], "strategy": sig["strategy"], "reason": reason,
                })
        return rejected
    except Exception as exc:
        logger.warning("decision_summary: 信号门禁检查失败: %s", exc)
        errors.append("被拒信号数据不可用")
        return []


def _risk_delta(config, target_date, provider, cache, errors):
    try:
        from dashboard.risk_analytics import _build_weights, _fetch_prices

        lookback = config.get("scanner", {}).get("lookback_years", 3)
        weights = _build_weights(config, target_date, provider, "假设持仓")
        if not weights:
            errors.append("风险快照不可用 (无持仓/watchlist 权重)")
            return None, None
        prices = _fetch_prices(list(weights.keys()), target_date, provider, years=max(lookback, 1))
        if prices.empty or len(weights) < 2:
            errors.append("风险快照数据不足")
            return None, None

        cmp = compare_portfolios(prices, weights, weights, sector_map=DEFAULT_SECTORS)
        snapshot = cmp["before"]

        as_of = pd.Timestamp(target_date).date().isoformat()
        cache.save_risk_snapshot(as_of, snapshot)
        previous = cache.load_previous_risk_snapshot(as_of)
        if previous is None:
            return snapshot, None

        comparison_date, previous_snapshot = previous
        delta = {
            k: snapshot.get(k, 0.0) - previous_snapshot.get(k, 0.0)
            for k in ("var_pct", "hhi", "sector_hhi", "effective_n")
            if k in snapshot
        }
        delta["comparison_date"] = comparison_date
        return snapshot, delta
    except Exception as exc:
        logger.warning("decision_summary: 风险快照计算失败: %s", exc)
        errors.append("风险快照不可用")
        return None, None


def _live_risk_snapshot(cache, as_of=None):
    """Load a guardrail snapshot only when it matches the requested day."""
    raw = cache.load_risk_state("risk_operational_snapshot")
    if not raw:
        return None
    try:
        value = json.loads(raw)
        if not isinstance(value, dict) or value.get("as_of") != as_of:
            return None
        return value
    except (TypeError, ValueError):
        logger.warning("decision_summary: live risk snapshot is malformed")
        return None


def _verdict(
    risk_level, trading_paused, pause_reason, signals, rejected,
    top_risk_level="normal",
):
    if trading_paused:
        return "paused", f"🛑 已暂停交易 — {pause_reason}"
    if risk_level == RiskLevel.RED:
        return "red", "🔴 高风险，不建议新开仓"
    if not signals:
        return "no_signal", "⚪ 今日无信号，无需操作"
    if len(rejected) >= len(signals):
        return "all_blocked", "🟡 有信号但被过滤，无可执行操作"
    actionable = len(signals) - len(rejected)
    if top_risk_level == "high":
        return "top_caution", f"🟡 有 {actionable} 个信号，但大盘上涨过热，不建议追高"
    return "actionable", f"🟢 可正常交易 — {actionable} 个信号可执行"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_REGIME_LABEL = {
    "TRENDING_UP": "上升趋势", "TRENDING_DOWN": "下降趋势",
    "RANGING": "震荡", "TRANSITIONAL": "过渡期",
}


def render_decision_summary(summary: dict) -> None:
    """Render the decision-summary card. Pure rendering — no computation."""
    st.subheader("今日决策摘要")

    code = summary["verdict_code"]
    if code in {"paused", "red"}:
        st.error(summary["verdict_text"])
    elif code in {"all_blocked", "top_caution"}:
        st.warning(summary["verdict_text"])
    elif code == "actionable":
        st.success(summary["verdict_text"])
    else:
        st.info(summary["verdict_text"])

    if summary["errors"]:
        st.caption("⚠ " + " / ".join(summary["errors"]))

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        level = summary["risk_level"]
        level_label = level.value.upper() if level else "—"
        regime = summary["regime"]
        regime_label = _REGIME_LABEL.get(regime.name, regime.name) if regime else "—"
        st.metric("风险灯 × Regime", f"{level_label} × {regime_label}")
        if summary.get("sentiment_score") is not None:
            st.caption(
                f"大盘情绪 {summary['sentiment_score']:.1f}/100 "
                f"({summary.get('sentiment_label', '未知')})"
            )
        if summary.get("top_risk_level") == "high":
            st.caption("顶部风险偏高: 超买 / 乖离 / 短期涨幅出现共振，避免追高")
        with st.expander("判定依据"):
            if summary["risk_reasons"]:
                for r in summary["risk_reasons"]:
                    st.write(f"- {r}")
            else:
                st.caption("无额外依据 / 数据不可用")
            if summary.get("top_risk_reasons"):
                st.markdown("**上涨过热依据**")
                for reason in summary["top_risk_reasons"]:
                    st.write(f"- {reason}")
            live = summary.get("live_risk")
            if live:
                status = "暂停" if live.get("paused") else "运行中"
                st.caption(
                    f"实盘风控: {status} | 敞口 {live.get('exposure_pct', 0):.1f}% | "
                    f"日内交易 {live.get('daily_trade_count', 0)} 笔 | "
                    f"连续亏损 {live.get('consecutive_losses', 0)} 笔"
                )
                if live.get("pause_reason"):
                    st.caption(f"暂停原因: {live['pause_reason']}")

    with col2:
        st.metric("被拒信号", f"{summary['rejected_count']} / {summary['signals_total']}")
        with st.expander("被拒明细"):
            if summary["rejected"]:
                for r in summary["rejected"]:
                    st.write(f"- {r['symbol']} ({r['strategy']}): {r['reason']}")
            else:
                st.caption("无被拒信号")

    with col3:
        delta = summary["risk_delta"]
        if delta is None:
            st.metric("组合风险 Δ (较前次快照)", "—")
            delta_caption = "暂无较早快照数据"
        else:
            var_d = delta.get("var_pct", 0.0)
            comparison_date = delta.get("comparison_date", "前次快照")
            st.metric(f"组合风险 Δ (较 {comparison_date})", f"VaR {var_d:+.2f}pp")
            delta_caption = None
        with st.expander("风险快照明细"):
            snap = summary["risk_snapshot"]
            if snap:
                st.write(f"- VaR(95%,1d): {snap.get('var_pct', 0):.2f}%")
                st.write(f"- 符号级 HHI: {snap.get('hhi', 0):.0f}")
                st.write(f"- 行业 HHI: {snap.get('sector_hhi', 0):.0f}")
                st.write(f"- 有效N: {snap.get('effective_n', 0):.2f}")
                if delta is not None:
                    st.write(
                        f"- 较 {delta.get('comparison_date', '前次快照')}: "
                        f"HHI {delta.get('hhi', 0):+.0f} / "
                        f"行业HHI {delta.get('sector_hhi', 0):+.0f} / "
                        f"有效N {delta.get('effective_n', 0):+.2f}"
                    )
            else:
                st.caption(delta_caption or "数据不可用")

    with col4:
        st.metric("今日信号", summary["signals_total"])
        with st.expander("信号明细"):
            if summary["signals"]:
                for s in summary["signals"]:
                    tag = "买入" if s["signal"] == 1 else "卖出"
                    sentiment = s.get("symbol_sentiment")
                    sentiment_tag = (
                        f" · {sentiment['label']} {sentiment['score']:.0f}/100"
                        if sentiment else " · 情绪数据不足"
                    )
                    st.write(
                        f"- {s['symbol']} {s['strategy']}: {tag} "
                        f"@ {s.get('price', 0):.2f}{sentiment_tag}"
                    )
            else:
                st.caption("今日无信号")

    st.divider()
