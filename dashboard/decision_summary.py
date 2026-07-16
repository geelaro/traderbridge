"""Decision summary — one-glance "should I trade today" card.

Aggregates four existing analyses into a single verdict: risk light +
regime, today's signals, gate-rejected signals, and portfolio risk vs
yesterday. ``build_decision_summary`` is pure compute (no Streamlit
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

``trading_paused``/``pause_reason`` are read from ``risk_state`` best-
effort. Nothing currently writes those keys (only an in-memory
``RiskController`` attribute set during a live session), so today this
always reads back empty — expected while no daemon runs (see
ROADMAP.md P2-1 / 阶段八). Reading it anyway means this starts working
for free the day a writer exists.

v1 scope — risk delta vs yesterday
------------------------------------
No historical VaR/HHI persistence existed before this feature. Rather
than add a new table (would need a schema migration + violate the
"only live/scripts write new trading tables" layering convention for
no real benefit — see AGENTS.md), this snapshots today's portfolio risk
into the existing ``risk_state`` key/value store under
``risk_snapshot:<date>`` and diffs against yesterday's key if present.
"""

from __future__ import annotations

import json
from datetime import timedelta

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

    risk_level, risk_reasons, market_state = _risk_and_regime(config, end_ts, provider, errors)
    signals = _today_signals(config, end_ts, provider, cache, errors)
    rejected = _rejected_signals(config, market_state, signals, errors)
    risk_snapshot, risk_delta = _risk_delta(config, target_date, provider, cache, errors)

    trading_paused = (cache.load_risk_state("trading_paused") == "True")
    pause_reason = cache.load_risk_state("pause_reason") or ""

    verdict_code, verdict_text = _verdict(
        risk_level, trading_paused, pause_reason, signals, rejected,
    )

    return {
        "verdict_code": verdict_code,
        "verdict_text": verdict_text,
        "risk_level": risk_level,
        "risk_reasons": risk_reasons,
        "regime": market_state.regime if market_state else None,
        "volatility": market_state.volatility if market_state else None,
        "signals": signals,
        "signals_total": len(signals),
        "rejected": rejected,
        "rejected_count": len(rejected),
        "risk_snapshot": risk_snapshot,
        "risk_delta": risk_delta,
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
        live_vix = get_realtime_vix()
        state = compute_risk_state(spy_df, vix_df, realtime_vix=live_vix)
        market_state = MarketStateClassifier(spy_df).classify()
        return state.level, state.reasons, market_state
    except Exception as exc:
        logger.warning("decision_summary: risk/regime 计算失败: %s", exc)
        errors.append("风险灯/regime 数据不可用")
        return None, [], None


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

        today_key = f"risk_snapshot:{pd.Timestamp(target_date).date().isoformat()}"
        cache.save_risk_state(today_key, json.dumps(snapshot))

        yesterday = pd.Timestamp(target_date).date() - timedelta(days=1)
        y_raw = cache.load_risk_state(f"risk_snapshot:{yesterday.isoformat()}")
        if not y_raw:
            return snapshot, None

        y_snapshot = json.loads(y_raw)
        delta = {
            k: snapshot.get(k, 0.0) - y_snapshot.get(k, 0.0)
            for k in ("var_pct", "hhi", "sector_hhi", "effective_n")
            if k in snapshot
        }
        return snapshot, delta
    except Exception as exc:
        logger.warning("decision_summary: 风险快照计算失败: %s", exc)
        errors.append("风险快照不可用")
        return None, None


def _verdict(risk_level, trading_paused, pause_reason, signals, rejected):
    if trading_paused:
        return "paused", f"🛑 已暂停交易 — {pause_reason}"
    if risk_level == RiskLevel.RED:
        return "red", "🔴 高风险，不建议新开仓"
    if not signals:
        return "no_signal", "⚪ 今日无信号，无需操作"
    if len(rejected) >= len(signals):
        return "all_blocked", "🟡 有信号但被过滤，无可执行操作"
    actionable = len(signals) - len(rejected)
    return "actionable", f"🟢 可正常交易 — {actionable} 个信号可执行"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_VERDICT_THEME = {
    "paused":     {"bg": "#e2e3e5", "border": "#6c757d", "fg": "#383d41"},
    "red":        {"bg": "#f8d7da", "border": "#dc3545", "fg": "#721c24"},
    "no_signal":  {"bg": "#e2e3e5", "border": "#6c757d", "fg": "#383d41"},
    "all_blocked": {"bg": "#fff3cd", "border": "#ffc107", "fg": "#856404"},
    "actionable": {"bg": "#d4edda", "border": "#28a745", "fg": "#155724"},
}

_REGIME_LABEL = {
    "TRENDING_UP": "上升趋势", "TRENDING_DOWN": "下降趋势",
    "RANGING": "震荡", "TRANSITIONAL": "过渡期",
}


def render_decision_summary(summary: dict) -> None:
    """Render the decision-summary card. Pure rendering — no computation."""
    st.subheader("今日决策摘要")

    theme = _VERDICT_THEME.get(summary["verdict_code"], _VERDICT_THEME["no_signal"])
    st.markdown(
        f"""
        <div style="background:{theme['bg']};border-left:6px solid {theme['border']};
                    padding:14px 20px;border-radius:8px;margin-bottom:12px;
                    color:{theme['fg']};font-size:1.3em;font-weight:bold;">
          {summary['verdict_text']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if summary["errors"]:
        st.caption("⚠ " + " / ".join(summary["errors"]))

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        level = summary["risk_level"]
        level_label = level.value.upper() if level else "—"
        regime = summary["regime"]
        regime_label = _REGIME_LABEL.get(regime.name, regime.name) if regime else "—"
        st.metric("风险灯 × Regime", f"{level_label} × {regime_label}")
        with st.expander("判定依据"):
            if summary["risk_reasons"]:
                for r in summary["risk_reasons"]:
                    st.write(f"- {r}")
            else:
                st.caption("无额外依据 / 数据不可用")

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
            st.metric("组合风险 Δ (较昨日)", "—")
            delta_caption = "无昨日快照数据"
        else:
            var_d = delta.get("var_pct", 0.0)
            st.metric("组合风险 Δ (较昨日)", f"VaR {var_d:+.2f}pp")
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
                        f"- 较昨日: HHI {delta.get('hhi', 0):+.0f} / "
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
                    st.write(f"- {s['symbol']} {s['strategy']}: {tag} @ {s.get('price', 0):.2f}")
            else:
                st.caption("今日无信号")

    st.divider()
