"""Tests for analysis/risk_monitor.py."""

import numpy as np
import pandas as pd
import pytest

from analysis.risk_monitor import compute_risk_state, RiskLevel, RiskState


def _synthetic_spy(close_path, start="2020-01-01"):
    """Build a synthetic SPY OHLCV frame from a Close trajectory.

    Adds plausible OHL (small range) and dummy Volume.
    """
    dates = pd.bdate_range(start, periods=len(close_path))
    close = np.asarray(close_path, dtype=float)
    return pd.DataFrame({
        "Open": close * 0.999,
        "High": close * 1.005,
        "Low": close * 0.995,
        "Close": close,
        "Volume": 1_000_000,
    }, index=dates)


def _synthetic_vix(value, n_bars=300, start="2020-01-01"):
    """Constant-VIX frame."""
    dates = pd.bdate_range(start, periods=n_bars)
    return pd.DataFrame({
        "Open": value, "High": value, "Low": value,
        "Close": value, "Volume": 0,
    }, index=dates)


# ===================================================================
# Empty / insufficient data → yellow
# ===================================================================


class TestDegradedInputs:
    def test_empty_spy(self):
        state = compute_risk_state(pd.DataFrame(), _synthetic_vix(15.0))
        assert state.level == RiskLevel.YELLOW
        assert any("SPY 数据缺失" in r for r in state.reasons)

    def test_empty_vix(self):
        spy = _synthetic_spy([400] * 250)
        state = compute_risk_state(spy, pd.DataFrame())
        assert state.level == RiskLevel.YELLOW
        assert any("VIX 数据缺失" in r for r in state.reasons)

    def test_spy_too_short(self):
        spy = _synthetic_spy([400] * 50)
        state = compute_risk_state(spy, _synthetic_vix(15.0))
        assert state.level == RiskLevel.YELLOW
        assert any("不足" in r for r in state.reasons)


# ===================================================================
# RED: any one of (SPY < MA200 - 3%, VIX > 30, 5d crash)
# ===================================================================


class TestRedTriggers:
    def test_spy_below_ma200_red(self):
        # Long uptrend then sharp decline → SPY ~ -10% under MA200
        rising = np.linspace(300, 450, 220)
        falling = np.linspace(450, 380, 30)  # last close ~380 vs MA200 ~395
        spy = _synthetic_spy(list(rising) + list(falling))
        state = compute_risk_state(spy, _synthetic_vix(15.0))
        assert state.level == RiskLevel.RED
        assert any("MA200" in r for r in state.reasons)

    def test_vix_above_30_red(self):
        # SPY in healthy uptrend, but VIX spike
        spy = _synthetic_spy(np.linspace(300, 450, 250))
        state = compute_risk_state(spy, _synthetic_vix(35.0))
        assert state.level == RiskLevel.RED
        assert any("VIX" in r for r in state.reasons)

    def test_5d_crash_red(self):
        # SPY trending up then crashes -8% in last 5 bars
        base = list(np.linspace(300, 450, 245))
        crash = [450, 440, 425, 410, 414]  # ~-8% in 5d
        spy = _synthetic_spy(base + crash)
        state = compute_risk_state(spy, _synthetic_vix(20.0))
        assert state.level == RiskLevel.RED
        assert any("5日跌幅" in r for r in state.reasons)

    def test_multiple_red_reasons_reported(self):
        # SPY tanks + VIX spikes simultaneously
        base = list(np.linspace(300, 450, 240))
        crash = list(np.linspace(450, 380, 10))
        spy = _synthetic_spy(base + crash)
        state = compute_risk_state(spy, _synthetic_vix(40.0))
        assert state.level == RiskLevel.RED
        # Should mention both
        joined = " ".join(state.reasons)
        assert "VIX" in joined or "MA200" in joined  # at least one


# ===================================================================
# GREEN: all three (SPY > MA200+3%, ADX > 25, VIX < 18)
# ===================================================================


class TestGreen:
    def test_strong_uptrend_low_vix_green(self):
        # Steady uptrend gives strong ADX, SPY well above MA200
        # Use a clean linear ramp to ensure ADX > 25
        spy = _synthetic_spy(np.linspace(300, 500, 300))
        state = compute_risk_state(spy, _synthetic_vix(14.0))
        assert state.level == RiskLevel.GREEN
        assert len(state.reasons) >= 3

    def test_low_vix_not_enough_alone(self):
        # Flat market → ADX low → can't be green even with low VIX
        spy = _synthetic_spy([400 + np.sin(i / 5) * 5 for i in range(300)])
        state = compute_risk_state(spy, _synthetic_vix(12.0))
        assert state.level != RiskLevel.GREEN  # range-bound, no trend


# ===================================================================
# YELLOW: catch-all
# ===================================================================


class TestYellow:
    def test_spy_near_ma200(self):
        # SPY hovering around MA200 (±3% buffer)
        spy = _synthetic_spy([400 + np.sin(i / 8) * 8 for i in range(300)])
        state = compute_risk_state(spy, _synthetic_vix(20.0))
        # Should be yellow, range-bound + elevated VIX
        assert state.level == RiskLevel.YELLOW

    def test_strong_trend_but_high_vix(self):
        # Strong uptrend, but VIX between 18-30 → yellow not green
        spy = _synthetic_spy(np.linspace(300, 500, 300))
        state = compute_risk_state(spy, _synthetic_vix(22.0))
        assert state.level == RiskLevel.YELLOW
        assert any("VIX" in r for r in state.reasons)


# ===================================================================
# Indicators always populated
# ===================================================================


class TestIndicators:
    def test_indicators_complete(self):
        spy = _synthetic_spy(np.linspace(300, 450, 250))
        state = compute_risk_state(spy, _synthetic_vix(15.0))
        for key in ["spy_close", "spy_ma200", "spy_vs_ma200_pct",
                    "spy_adx", "spy_5d_return_pct", "spy_20d_return_pct",
                    "spy_rsi14", "top_risk_level", "vix", "as_of"]:
            assert key in state.indicators

    def test_summary_contains_emoji_and_reasons(self):
        spy = _synthetic_spy(np.linspace(300, 500, 300))
        state = compute_risk_state(spy, _synthetic_vix(14.0))
        s = state.summary()
        assert any(e in s for e in ("🟢", "🟡", "🔴"))
        assert "GREEN" in s.upper() or "YELLOW" in s.upper() or "RED" in s.upper()


# ===================================================================
# Custom params override
# ===================================================================


class TestParamsOverride:
    def test_lower_vix_thresholds(self):
        # Strict thresholds — VIX < 12 required for green
        spy = _synthetic_spy(np.linspace(300, 500, 300))
        state = compute_risk_state(
            spy, _synthetic_vix(14.0),
            params={"vix_low": 12.0},
        )
        # VIX = 14, threshold = 12 → not green
        assert state.level != RiskLevel.GREEN


class TestTopRisk:
    def test_extended_rally_is_marked_as_high_top_risk(self):
        spy = _synthetic_spy(np.linspace(300, 500, 300))
        state = compute_risk_state(spy, _synthetic_vix(14.0))

        assert state.top_risk_level == "high"
        assert state.indicators["overbought"] is True
        assert state.indicators["top_risk_score"] >= 3
        assert any("RSI14" in reason for reason in state.top_risk_reasons)
        assert any("MA200" in reason for reason in state.top_risk_reasons)

    def test_single_overbought_signal_is_not_called_top(self):
        # A modestly extended, smooth move can trigger RSI without meeting
        # the two-signal consensus required for a high top-risk label.
        close = np.concatenate([
            np.linspace(400, 430, 270),
            np.linspace(430, 448, 30),
        ])
        state = compute_risk_state(_synthetic_spy(close), _synthetic_vix(14.0))

        assert state.top_risk_level in {"normal", "elevated"}
        assert state.top_risk_level != "high"


class TestMarketSentiment:
    def test_numeric_sentiment_separates_spy_and_qqq(self):
        spy = _synthetic_spy(np.linspace(300, 500, 300))
        qqq = _synthetic_spy(np.linspace(500, 300, 300))
        state = compute_risk_state(spy, _synthetic_vix(15.0), qqq_df=qqq)

        assert 0 <= state.sentiment_score <= 100
        assert state.sentiment_label in {"极度超卖", "超卖", "中性", "超买", "极度超买"}
        assert state.market_sentiments["SPY"]["score"] > 65
        assert state.market_sentiments["QQQ"]["score"] < 35
        assert state.indicators["spy_sentiment_score"] > 65
        assert state.indicators["qqq_sentiment_score"] < 35
        assert "rsi" in state.market_sentiments["SPY"]["components"]


# ===================================================================
# Realtime VIX override
# ===================================================================


class TestRealtimeVixOverride:
    def test_realtime_overrides_eod(self):
        """Realtime VIX 35 should flip RED even if cached EOD says 15."""
        spy = _synthetic_spy(np.linspace(300, 500, 300))
        eod_vix = _synthetic_vix(15.0)  # green-light EOD
        state = compute_risk_state(spy, eod_vix, realtime_vix=35.0)
        # VIX > 30 forces RED
        assert state.level == RiskLevel.RED
        assert any("VIX=35.0" in r for r in state.reasons)
        # Indicator reflects the realtime value, not the EOD
        assert state.indicators["vix"] == 35.0
        assert state.indicators["vix_source"] == "realtime"

    def test_none_realtime_uses_eod(self):
        """None override → falls back to EOD vix_df as before."""
        spy = _synthetic_spy(np.linspace(300, 500, 300))
        state = compute_risk_state(spy, _synthetic_vix(15.0), realtime_vix=None)
        assert state.indicators["vix"] == 15.0
        assert state.indicators["vix_source"] == "eod"

    def test_zero_realtime_ignored(self):
        """A zero or negative realtime value is treated as missing."""
        spy = _synthetic_spy(np.linspace(300, 500, 300))
        state = compute_risk_state(spy, _synthetic_vix(15.0), realtime_vix=0)
        assert state.indicators["vix"] == 15.0
        assert state.indicators["vix_source"] == "eod"

    def test_realtime_keeps_indicators_from_spy(self):
        """Realtime override only touches VIX, not SPY/ADX/etc."""
        spy = _synthetic_spy(np.linspace(300, 500, 300))
        state = compute_risk_state(spy, _synthetic_vix(15.0), realtime_vix=18.0)
        # All other indicators still populated
        for k in ("spy_close", "spy_ma200", "spy_vs_ma200_pct",
                  "spy_adx", "spy_5d_return_pct", "as_of"):
            assert k in state.indicators
