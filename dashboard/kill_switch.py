"""Dashboard tab — Kill Switch emergency liquidation control.

Manual-only.  No automatic triggers.  Requires:
1. Typed "CONFIRM" in a text box
2. Non-empty reason
3. Click of the red button

When active, shows a persistent banner and exposes a Reset button.
"""

from __future__ import annotations

import logging

import streamlit as st

from live.kill_switch import KillSwitch

logger = logging.getLogger(__name__)


def render_kill_switch(broker, risk_ctrl, notifier, cache):
    """Render the Kill Switch tab.

    Parameters are dependency-injected so the panel works in any context
    (dry-run dashboard, live dashboard with FutuBroker, etc.).
    """
    st.header("🚨 紧急停机 / Kill Switch")
    st.caption(
        "**手动触发**, 无任何自动阈值绑定. 基于历史数据 (VIX>50 共 5 次, "
        "之后 SPY 250 日平均涨 +44.6%, 是抄底信号而非清仓信号), "
        "VIX/回撤阈值都不能可靠地预测\"该不该跑\". 用户自己判断后按按钮."
    )

    if broker is None:
        st.warning("Broker 未连接 — Kill Switch 仅在有 broker 时可用")
        return

    ks = KillSwitch(broker, risk_ctrl, notifier, cache)
    state = ks.get_state()

    # ── ACTIVE banner ──────────────────────────────────────────────
    if state["active"]:
        st.error(
            f"🚨 **KILL SWITCH ACTIVE** — 交易已暂停, "
            f"daemon 不会开新仓\n\n"
            f"**原因**: {state['reason']}\n\n"
            f"**触发时间**: {state['triggered_at']}"
        )
        st.markdown(
            "**接下来要做**:\n"
            "1. 登录 broker 确认实际持仓状态(可能有部分订单未成交)\n"
            "2. 复盘市场, 判断风险是否过去\n"
            "3. 全部 OK 后点 Reset 解除暂停, daemon 才会重新开仓"
        )
        if st.button("解除 Kill Switch (Reset)", type="primary"):
            ks.reset("Manual reset via dashboard")
            st.rerun()
        st.divider()
        _render_recent_history(cache)
        return

    # ── Idle: arming UI ────────────────────────────────────────────
    try:
        positions = list(broker.get_positions() or [])
    except Exception as exc:
        logger.warning("Kill Switch panel: failed to read positions: %s", exc)
        positions = []

    st.markdown(
        f"**当前持仓**: {len(positions)} 个 ({broker.name} broker)"
    )
    if positions:
        rows = [{
            "标的": p.symbol,
            "数量": p.quantity,
            "均价": f"${p.avg_price:.2f}",
            "市值": f"${p.market_value:,.0f}",
            "浮盈": f"${p.unrealized_pnl:+,.0f}",
        } for p in positions if p.quantity != 0]
        if rows:
            st.dataframe(rows, hide_index=True, use_container_width=True)
        else:
            st.info("无持仓 — 触发 Kill Switch 不会下任何单(只会 pause daemon)")
    else:
        st.info("无持仓数据")

    st.markdown("---")
    st.warning(
        "⚠ 点击按钮会**立即对所有持仓下市价单平仓**, 不可撤销。"
        "请确认你确实要这样做."
    )

    col1, col2 = st.columns(2)
    with col1:
        confirm = st.text_input(
            "输入 CONFIRM 解锁按钮",
            value="",
            help="防止误点 — 必须显式输入 CONFIRM 才会激活下方按钮",
        )
    with col2:
        reason = st.text_input(
            "触发原因 (必填)",
            value="",
            placeholder="例如: 闪崩 / 突发地缘 / VIX 暴涨 / ...",
        )

    preview = st.session_state.get("_ks_dry_run_result")
    if preview and preview.get("reason") != reason.strip():
        st.session_state.pop("_ks_dry_run_result", None)

    armed = confirm.strip() == "CONFIRM" and reason.strip()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚨 紧急平仓全部 (实盘下单)",
                     type="primary", disabled=not armed):
            st.session_state.pop("_ks_dry_run_result", None)
            with st.spinner("Triggering kill switch..."):
                result = ks.trigger(reason.strip())
            st.success(
                f"Kill Switch 已触发. 下单 {len(result['orders'])} 笔, "
                f"错误 {len(result['errors'])}. 飞书已推送."
            )
            st.json(result)
            st.rerun()
    with col2:
        if st.button("Dry Run (预演, 不实际下单)",
                     type="secondary", disabled=not armed):
            with st.spinner("Dry-run..."):
                st.session_state["_ks_dry_run_result"] = ks.trigger(
                    reason.strip(), dry_run=True,
                )

    preview = st.session_state.get("_ks_dry_run_result")
    if preview:
        st.info("Dry Run 完成 — 未下单，未改变交易状态")
        st.json(preview)

    st.divider()
    _render_recent_history(cache)


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


def _render_recent_history(cache):
    """Show the last Kill Switch triggers + resets (audit trail)."""
    st.subheader("历史记录")
    rows = []
    try:
        for r in cache.load_alert_history(days=365):
            if r["alert_type"] not in ("kill_switch", "kill_switch_reset"):
                continue
            payload = r["payload"]
            rows.append({
                "时间": r["ts"].replace("T", " "),
                "事件": "触发" if r["alert_type"] == "kill_switch" else "重置",
                "原因": payload.get("reason", ""),
                "下单数": len(payload.get("orders", [])) if r["alert_type"] == "kill_switch" else "—",
                "错误数": len(payload.get("errors", [])) if r["alert_type"] == "kill_switch" else "—",
                "Dry Run": "是" if payload.get("dry_run") else "否",
            })
    except Exception as exc:
        logger.debug("Kill Switch history fetch failed: %s", exc)

    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)
    else:
        st.caption("无历史记录")
