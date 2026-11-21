"""Four-way research signals must not become broker short entries."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from analysis.hypothetical_positions import compute_hypothetical_positions
from analysis.pnl_breakdown import unrealized_pnl_summary
from analysis.risk_monitor import compute_directional_risk, compute_risk_state, compute_china_market_state
from strategy import STRATEGY_MAP
from strategy.base import BaseStrategy
from strategy.actions import replay_actions
from utils.signal_scanner import SignalScanner
from utils.trade_actions import TradeAction, action_label, has_action, resolve_action, restore_action


class BothWays(BaseStrategy):
    long_only = False
    min_bars = 0

    def __init__(self, **kwargs):
        self.params = SimpleNamespace(trail_atr_mult=2.5)

    def calculate_indicators(self, df, df_weekly=None):
        return df.copy()


class ShortStop(BothWays):
    def check_exit(self, df, i, entry_price, highest_since_entry,
                   lowest_since_entry=None, position=None):
        return df["Close"].iloc[i] >= lowest_since_entry + 5, "short_stop"


def frame(signals, prices=None):
    prices = np.array(prices if prices is not None else [100.] * len(signals))
    return pd.DataFrame({
        "Open": prices, "Close": prices, "High": prices + 1, "Low": prices - 1,
        "Volume": 1000, "ATR": 2., "Signal": signals,
    }, index=pd.bdate_range("2025-01-01", periods=len(signals)))


@pytest.mark.parametrize("signal,position,long_only,expected", [
    (1, 0, True, "open_long"), (-1, 0, True, "hold"),
    (-1, 0, False, "open_short"), (-1, 10, True, "close_long"),
    (1, -10, True, "close_short"), (1, 10, False, "hold"),
    (-1, -10, False, "hold"), (0, 0, False, "hold"),
])
def test_resolve(signal, position, long_only, expected):
    assert resolve_action(signal, position, long_only=long_only) == expected


def test_replay_all_actions_and_no_same_bar_reverse():
    result = replay_actions(BothWays(), frame([-1, -1, 1, 1, -1, -1]))
    assert result.action.tolist() == ["open_short", "hold", "close_short", "open_long", "close_long", "open_short"]
    assert result.position_after.tolist() == [-1, -1, 0, 1, 0, -1]
    assert result.entry_index.iloc[1] == 0


def test_long_only_does_not_short():
    strategy = BothWays()
    strategy.long_only = True
    result = replay_actions(strategy, frame([-1, 1, 1, -1, -1]))
    assert result.action.tolist() == ["hold", "open_long", "hold", "close_long", "hold"]
    assert result.entry_index.iloc[2] == 1


def test_strategy_exit_without_opposite_signal():
    result = replay_actions(ShortStop(), frame([-1, 0, 0], [100, 90, 96]))
    assert result.action.iloc[-1] == "close_short"
    assert result.action_reason.iloc[-1] == "short_stop"


def test_warmup_and_invalid_prices():
    strategy = BothWays()
    strategy.min_bars = 2
    result = replay_actions(strategy, frame([1, -1, 1, -1], [100, 100, 100, np.nan]))
    assert result.action.tolist() == ["hold", "hold", "open_long", None]
    assert result.position_after.iloc[-1] == 1


def test_stateful_strategy_is_not_mutated():
    class Stateful(BothWays):
        calls = 0

        def check_exit(self, *args, **kwargs):
            self.calls += 1
            return False, ""

    strategy = Stateful()
    replay_actions(strategy, frame([1, 0, 0]))
    assert strategy.calls == 0


def test_persisted_action_and_legacy_labels():
    row = {"signal": 0, "indicators": json.dumps({"_action": "close_short"})}
    assert has_action(row)
    assert action_label(row) == "空头平仓"
    assert restore_action(row)["action"] == "close_short"
    assert action_label({"signal": -1}) == "卖出"
    assert not has_action({"signal": -1, "action": "hold"})
    assert not has_action({"signal": 1, "action": None})
    assert restore_action({"signal": 0, "indicators": "broken"}) == {"signal": 0, "indicators": "broken"}


def test_scanner_preserves_raw_signal_and_stores_exit(monkeypatch, temp_cache):
    monkeypatch.setitem(STRATEGY_MAP, "short_test", ShortStop)
    df = frame([-1, 0, 0, 1], [100, 90, 96, 105])
    provider = MagicMock()
    provider.get_daily.return_value = df
    result = SignalScanner(provider, temp_cache).scan(
        {"watchlist": [{"symbol": "TEST", "active": "short_test"}]},
        target_date=df.index[2].date().isoformat(),
    )[0]
    assert result["signal"] == 0
    assert result["action"] == "close_short"
    assert result["position_source"] == "hypothetical_close"
    assert result["bar_date"] == df.index[2].date().isoformat()
    history = temp_cache.query_signals()[0]
    assert restore_action(history)["action"] == "close_short"


def test_monitor_params_are_separate_from_active(monkeypatch):
    seen = []

    class Capture(BothWays):
        def __init__(self, **kwargs):
            seen.append(kwargs)
            super().__init__()

    monkeypatch.setitem(STRATEGY_MAP, "a", Capture)
    monkeypatch.setitem(STRATEGY_MAP, "b", Capture)
    provider = MagicMock()
    provider.get_daily.return_value = frame([1, 0])
    SignalScanner(provider).scan({
        "watchlist": [{"symbol": "TEST", "active": "a", "monitor": ["b"], "params": {"x": 3}}],
        "strategy": {"a": {"x": 1}, "b": {"y": 2}},
    }, target_date="2025-01-02")
    assert seen == [{"x": 3}, {"y": 2}]


def test_short_position_pnl_and_stop(monkeypatch):
    monkeypatch.setitem(STRATEGY_MAP, "both_test", BothWays)
    provider = MagicMock()
    provider.get_daily.return_value = frame([-1, -1, 0], [100, 95, 90])
    rows = compute_hypothetical_positions(
        {"watchlist": [{"symbol": "TEST", "active": "both_test"}]},
        "2025-01-03", provider,
    )
    row = rows[0]
    assert row["direction"] == "SHORT"
    assert row["entry_date"] == "2025-01-01"
    assert row["pnl_pct"] == pytest.approx(10)
    assert row["stop_price"] == pytest.approx(94)
    assert row["distance_pct"] == pytest.approx(4 / 90 * 100)


def test_closed_short_not_reported_as_long(monkeypatch):
    monkeypatch.setitem(STRATEGY_MAP, "both_test", BothWays)
    provider = MagicMock()
    provider.get_daily.return_value = frame([-1, 1], [100, 90])
    row = compute_hypothetical_positions(
        {"watchlist": [{"symbol": "TEST", "active": "both_test"}]},
        "2025-01-02", provider, include_closed=True,
    )[0]
    assert row["direction"] == "FLAT"
    assert row["action"] == "close_short"


def test_unrealized_pnl_uses_signed_direction():
    summary = unrealized_pnl_summary([
        {"symbol": "LONG", "entry_price": 100, "current_price": 110,
         "shares": 2, "direction": "LONG"},
        {"symbol": "SHORT", "entry_price": 100, "current_price": 90,
         "shares": 2, "direction": "SHORT"},
    ])
    assert summary["total"] == 40
    assert summary["n_winning"] == 2
    short = next(r for r in summary["by_symbol"] if r["symbol"] == "SHORT")
    assert short["pnl"] == 20
    assert short["pnl_pct"] == pytest.approx(10)
    assert short["direction"] == "SHORT"


def test_long_only_risk_view_excludes_hypothetical_shorts(monkeypatch):
    import dashboard.risk_analytics as view

    monkeypatch.setattr(view, "compute_hypothetical_positions", lambda *a, **k: [
        {"symbol": "LONG", "direction": "LONG"},
        {"symbol": "SHORT", "direction": "SHORT"},
    ])
    notices = []
    monkeypatch.setattr(view.st, "info", notices.append)
    assert view._build_weights({}, None, None, "假设持仓") == {"LONG": 1.0}
    assert "空头" in notices[0]
    monkeypatch.setattr(view, "compute_hypothetical_positions", lambda *a, **k: [
        {"symbol": "SHORT", "direction": "SHORT"},
    ])
    assert view._build_weights({}, None, None, "假设持仓") == {}


def market_frame(down=False):
    prices = np.linspace(150, 100, 260) if down else np.linspace(100, 150, 260)
    return frame([0] * 260, prices)


@pytest.mark.parametrize("down", [True, False])
def test_both_directions_can_be_high_risk(down):
    result = compute_directional_risk(market_frame(down))
    assert result["long"]["level"] == result["short"]["level"] == "red"
    assert result["long"]["reasons"] != result["short"]["reasons"]


def test_extreme_vix_raises_both_sides_without_actions():
    df = frame([0] * 260, 100 + np.sin(np.arange(260)))
    risk = compute_directional_risk(df, vix=55)
    assert risk["long"]["level"] == risk["short"]["level"] == "red"
    assert "action" not in risk


def test_risk_missing_data_is_unknown():
    risk = compute_directional_risk(pd.DataFrame())
    assert not risk["available"]
    assert risk["short"]["level"] == "unknown"


def test_us_and_cn_market_risk_expose_both_sides():
    df = market_frame()
    state = compute_risk_state(df, frame([0] * 260, [16.] * 260), qqq_df=market_frame(True))
    assert state.directional_risks["available"]
    assert state.market_states["QQQ"]["directional_risks"]["short"]["level"] == "red"
    cn = compute_china_market_state(df, market_frame(True))
    assert cn["directional_risks"]["available"]
    assert "long" in cn["markets"]["沪深300"]["directional_risks"]
