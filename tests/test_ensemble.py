"""Tests for strategy/ensemble.py — StrategyEnsemble weighted voting."""

import numpy as np
import pandas as pd
import pytest
from strategy.ensemble import StrategyEnsemble, EnsembleParams
from strategy.turtle_trading import TurtleTrading
from strategy.bollinger_mean_reversion import BollingerMeanReversion
from strategy.weekly_macd import WeeklyMACD
from utils.market_state import MarketRegime, Volatility


@pytest.fixture
def proxy_df():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    close = 100 + np.cumsum(np.random.randn(300) * 0.3)
    return pd.DataFrame({
        "Open": close - 0.3, "High": close + 1,
        "Low": close - 1, "Close": close,
        "Volume": np.random.randint(1000, 10000, 300),
    }, index=dates)


@pytest.fixture
def stock_df(proxy_df):
    return proxy_df.copy()


class TestEnsembleParams:
    def test_defaults(self):
        p = EnsembleParams()
        assert p.long_bias_threshold == 0.3
        assert p.short_bias_threshold == 0.3
        assert p.min_agreement == 1

    def test_validation_rejects_negative(self):
        with pytest.raises(ValueError):
            EnsembleParams(long_bias_threshold=-0.1)


class TestEnsembleBasic:
    def test_creates_with_two_members(self, stock_df, proxy_df):
        members = [
            (TurtleTrading(), "trend"),
            (BollingerMeanReversion(), "mean_reversion"),
        ]
        ens = StrategyEnsemble(members, proxy_df)
        assert ens.min_bars > 0

    def test_calculate_indicators_returns_df(self, stock_df, proxy_df):
        members = [
            (TurtleTrading(), "trend"),
            (BollingerMeanReversion(), "mean_reversion"),
        ]
        ens = StrategyEnsemble(members, proxy_df)
        result = ens.calculate_indicators(stock_df)
        assert "Signal" in result.columns
        assert "ATR" in result.columns
        assert len(result) == len(stock_df)

    def test_signal_values_are_valid(self, stock_df, proxy_df):
        members = [
            (TurtleTrading(), "trend"),
            (BollingerMeanReversion(), "mean_reversion"),
        ]
        ens = StrategyEnsemble(members, proxy_df)
        result = ens.calculate_indicators(stock_df)
        assert set(result["Signal"].dropna().unique()).issubset({-1, 0, 1})

    def test_long_only_mode(self, stock_df, proxy_df):
        t = TurtleTrading()
        t.long_only = True
        members = [
            (t, "trend"),
            (BollingerMeanReversion(), "mean_reversion"),
        ]
        ens = StrategyEnsemble(members, proxy_df)
        result = ens.calculate_indicators(stock_df)
        # Should still produce valid signals
        assert "Signal" in result.columns

    def test_position_size_positive(self, stock_df, proxy_df):
        members = [(TurtleTrading(), "trend")]
        ens = StrategyEnsemble(members, proxy_df)
        qty = ens.position_size(10000, 100, 2.0)
        assert qty > 0

    def test_position_size_zero_for_bad_input(self, stock_df, proxy_df):
        members = [(TurtleTrading(), "trend")]
        ens = StrategyEnsemble(members, proxy_df)
        assert ens.position_size(10000, 0, 2.0) == 0


class TestEnsembleWithWeekly:
    def test_df_weekly_passed_through(self, stock_df, proxy_df):
        weekly = stock_df.resample("W-FRI").agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum",
        }).dropna()
        members = [
            (TurtleTrading(), "trend"),
        ]
        ens = StrategyEnsemble(members, proxy_df)
        result = ens.calculate_indicators(stock_df, df_weekly=weekly)
        assert "Signal" in result.columns

    def test_weekly_member_atr_is_aligned_to_daily_index(self, stock_df, proxy_df):
        members = [
            (WeeklyMACD(), "trend"),
            (TurtleTrading(), "trend"),
        ]
        ens = StrategyEnsemble(members, proxy_df)
        result = ens.calculate_indicators(stock_df)

        warmed = result["ATR"].iloc[ens.min_bars:]
        assert not warmed.isna().any()
        assert (warmed > 0).all()
