"""Tests for strategy/base.py:register — auto-registration decorator."""

from inspect import signature

import pytest

from strategy import STRATEGY_MAP
from strategy.base import BaseStrategy, _STRATEGY_REGISTRY, get_strategy_map, register


class TestStrategyMap:
    """STRATEGY_MAP is populated by @register decorators at import time."""

    def test_all_expected_strategies_registered(self):
        expected = {
            "trend_follower", "weekly_macd", "macd_kdj", "weekly_macd_kdj",
            "donchian_breakout", "atr_breakout", "turtle_trading",
            "daily_macd_kdj", "spy_ma_breakout", "rsi2_mean_reversion",
        }
        assert set(STRATEGY_MAP.keys()) == expected

    def test_map_values_are_strategy_subclasses(self):
        for name, cls in STRATEGY_MAP.items():
            assert issubclass(cls, BaseStrategy), \
                f"{name} → {cls.__name__} not a BaseStrategy subclass"

    def test_get_strategy_map_returns_live_view(self):
        live = get_strategy_map()
        assert live is _STRATEGY_REGISTRY

    def test_classes_carry_their_registered_name(self):
        for name, cls in STRATEGY_MAP.items():
            assert getattr(cls, "_strategy_name", None) == name

    def test_check_exit_signatures_follow_base_contract(self):
        for name, cls in STRATEGY_MAP.items():
            parameters = signature(cls.check_exit).parameters
            assert "lowest_since_entry" in parameters, name
            assert "position" in parameters, name

    def test_indicator_signatures_follow_base_contract(self):
        for name, cls in STRATEGY_MAP.items():
            parameters = signature(cls.calculate_indicators).parameters
            assert "df_weekly" in parameters, name


class TestRegisterDecorator:
    def test_duplicate_name_different_class_raises(self):
        # Pre-register
        @register("dup_test_strategy")
        class _A(BaseStrategy):
            min_bars = 1

            def calculate_indicators(self, df, df_weekly=None):
                return df

        # Same name on a different class must raise
        with pytest.raises(ValueError, match="already registered"):
            @register("dup_test_strategy")
            class _B(BaseStrategy):
                min_bars = 1

                def calculate_indicators(self, df, df_weekly=None):
                    return df

        # Cleanup so other tests don't inherit
        _STRATEGY_REGISTRY.pop("dup_test_strategy", None)

    def test_re_register_same_class_is_idempotent(self):
        @register("idem_test")
        class _Idem(BaseStrategy):
            min_bars = 1

            def calculate_indicators(self, df, df_weekly=None):
                return df

        # Re-applying the decorator to the same class shouldn't raise
        _STRATEGY_REGISTRY.pop("idem_test")
        decorated_again = register("idem_test")(_Idem)
        assert decorated_again is _Idem
        assert _STRATEGY_REGISTRY["idem_test"] is _Idem
        _STRATEGY_REGISTRY.pop("idem_test", None)

    def test_register_sets_strategy_name_attr(self):
        @register("attr_test")
        class _AttrTest(BaseStrategy):
            min_bars = 1

            def calculate_indicators(self, df, df_weekly=None):
                return df

        assert _AttrTest._strategy_name == "attr_test"
        _STRATEGY_REGISTRY.pop("attr_test", None)
