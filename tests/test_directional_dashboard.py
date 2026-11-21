"""Offline rendering checks for four actions and two-sided risk."""

from streamlit.testing.v1 import AppTest


def _app():
    from unittest.mock import MagicMock, patch
    import pandas as pd
    from tests.test_trade_actions import market_frame
    from dashboard.signals import render_signal_detail, render_risk_light, render_position_watch

    provider = MagicMock()
    provider.get_daily.return_value = market_frame()
    actions = ["open_long", "open_short", "close_long", "close_short"]
    signals = [{
        "symbol": f"TEST{i}", "name": f"Test {i}", "strategy": "turtle_trading",
        "signal": 0, "action": action, "action_reason": "strategy_exit",
        "price": 100., "atr": 2., "bar_date": "2025-12-30", "indicators": {},
    } for i, action in enumerate(actions)]
    config = {"watchlist": [{"symbol": r["symbol"], "active": r["strategy"]} for r in signals]}
    with patch("daily.scan_day", return_value=signals), patch("dashboard.signals._render_proximity"):
        render_signal_detail(config, pd.Timestamp("2025-12-30"), provider, MagicMock())
    with patch("dashboard.signals.get_realtime_vix", return_value=20.), patch("dashboard.signals.get_fear_greed", return_value=None):
        render_risk_light(config, pd.Timestamp("2025-12-30"), provider)
    row = {
        "symbol": "TEST1", "strategy": "turtle_trading", "status": "持有",
        "direction": "SHORT", "action": "hold", "entry_date": "2025-12-01",
        "entry_price": 110., "current_price": 100., "pnl_pct": 9.09,
        "stop_price": 103., "distance_pct": 3., "days_held": 29,
    }
    with patch("dashboard.signals.compute_hypothetical_positions", return_value=[row]):
        render_position_watch(config, pd.Timestamp("2025-12-30"), provider)


def test_four_action_and_original_market_state_layout():
    app = AppTest.from_function(_app).run(timeout=30)
    assert not app.exception
    metrics = {m.label: m.value for m in app.metric}
    for label in ("做多", "做空", "多头平仓", "空头平仓"):
        assert metrics[label] == "1"
    market_tables = [
        t.value for t in app.dataframe
        if {"市场", "压力", "情绪"}.issubset(t.value.columns)
    ]
    assert len(market_tables) == 1
    assert market_tables[0]["市场"].tolist() == ["SPY", "QQQ"]
    assert not any("多头风险" in t.value.columns for t in app.dataframe)
    labels = [metric.label for metric in app.metric]
    assert labels.index("中国大盘整体") == labels.index("QQQ") + 1
    positions = [t.value for t in app.dataframe if "今日动作" in t.value.columns][0]
    assert positions.iloc[0]["状态"] == "空头持有"
    assert positions.iloc[0]["浮盈%"] == "+9.09%"
