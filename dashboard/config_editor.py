"""Dashboard config editor — manage watchlist.toml via Streamlit UI."""

import streamlit as st

from utils.env import save_toml
from strategy import STRATEGY_MAP


def _has_duplicate_symbol(watchlist, symbol, exclude_index=None):
    normalized = symbol.strip().upper()
    if not normalized:
        return False
    return any(
        index != exclude_index
        and isinstance(item.get("symbol"), str)
        and item["symbol"].strip().upper() == normalized
        for index, item in enumerate(watchlist)
    )


def render_config_editor(config: dict):
    """Render a watchlist configuration editor tab."""
    st.header("Watchlist 配置管理")

    strategy_names = list(STRATEGY_MAP.keys())
    config_path = "watchlist.toml"
    watchlist = config.get("watchlist", [])

    # -- Current list -----------------------------------------------------
    st.subheader("当前标的")
    if not watchlist:
        st.info("watchlist 为空")

    edit_idx = None
    for idx, item in enumerate(watchlist):
        cols = st.columns([1, 2, 3, 1, 1])
        sym = item.get("symbol", "?")
        active = item.get("active", "")
        monitors = ", ".join(item.get("monitor", [])) or "—"
        cols[0].write(sym)
        cols[1].write(active)
        cols[2].write(monitors)
        if cols[3].button("编辑", key=f"edit_{idx}"):
            edit_idx = idx
        if cols[4].button("删除", key=f"del_{idx}"):
            watchlist.pop(idx)
            config["watchlist"] = watchlist
            save_toml(config_path, config)
            st.rerun()

    edit_item = watchlist[edit_idx] if edit_idx is not None else None

    # -- Add / Edit form --------------------------------------------------
    st.subheader("添加标的" if edit_item is None else f"编辑: {edit_item['symbol']}")

    with st.form("watchlist_form"):
        col1, col2 = st.columns([1, 1])
        default_sym = edit_item["symbol"] if edit_item else ""
        default_active = edit_item.get("active", strategy_names[0]) if edit_item else strategy_names[0]
        default_monitor = edit_item.get("monitor", []) if edit_item else []

        symbol = col1.text_input(
            "代码", value=default_sym, placeholder="AAPL",
        ).strip().upper()
        active_strat = col2.selectbox(
            "主策略",
            strategy_names,
            index=strategy_names.index(default_active) if default_active in strategy_names else 0,
        )
        monitor_strats = st.multiselect(
            "监控策略",
            [s for s in strategy_names if s != active_strat],
            default=[s for s in default_monitor if s in strategy_names and s != active_strat],
        )

        submitted = st.form_submit_button("保存")
        if submitted:
            if not symbol:
                st.error("代码不能为空")
                return
            if _has_duplicate_symbol(
                    watchlist, symbol,
                    exclude_index=edit_idx if edit_item is not None else None):
                st.error(f"{symbol} 已存在")
                return

            new_item = {
                "symbol": symbol,
                "name": edit_item.get("name", symbol) if edit_item else symbol,
                "active": active_strat,
                "monitor": monitor_strats,
            }

            if edit_item is not None:
                watchlist[edit_idx] = new_item
            else:
                watchlist.append(new_item)

            config["watchlist"] = watchlist
            save_toml(config_path, config)
            st.success(f"已保存 {symbol}")
            st.rerun()
