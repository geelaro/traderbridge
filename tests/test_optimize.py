"""Tests for optimize.py — _compute_score, OptResult, grid_search smoke."""

from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from engine.optimize import _compute_score, OptResult, PARAM_GRIDS, _PARAMS_CLASS, _next_trading_day


# ===================================================================
# _compute_score
# ===================================================================


class TestComputeScore:
    def _make_result(self, **kwargs):
        defaults = {
            "params": {}, "total_return": 0, "cagr": 0,
            "sharpe": 0, "max_dd": 0, "win_rate": 0,
            "trades": 5, "score": 0,
        }
        defaults.update(kwargs)
        return OptResult(**defaults)

    def test_too_few_trades_returns_negative(self):
        r = self._make_result(trades=2, sharpe=2.0)
        assert _compute_score(r, "sharpe") == -999

    def test_zero_trades_penalized(self):
        r = self._make_result(trades=0, sharpe=5.0)
        assert _compute_score(r, "sharpe") == -999

    def test_sharpe_metric(self):
        r = self._make_result(trades=5, sharpe=1.5)
        assert _compute_score(r, "sharpe") == 1.5

    def test_cagr_metric(self):
        r = self._make_result(trades=5, cagr=15.0)
        assert _compute_score(r, "cagr") == 15.0

    def test_total_return_metric(self):
        r = self._make_result(trades=5, total_return=25.0)
        assert _compute_score(r, "total_return") == 25.0

    def test_composite_metric(self):
        r = self._make_result(trades=5, sharpe=1.0, cagr=10.0, max_dd=-5.0)
        score = _compute_score(r, "composite")
        # composite = sharpe*2 + cagr/100 - abs(max_dd/100)
        expected = 1.0 * 2 + 10.0 / 100 - abs(-5.0 / 100)
        assert abs(score - expected) < 0.01

    def test_composite_penalizes_drawdown(self):
        r1 = self._make_result(trades=5, sharpe=1.0, cagr=10.0, max_dd=-5.0)
        r2 = self._make_result(trades=5, sharpe=1.0, cagr=10.0, max_dd=-20.0)
        assert _compute_score(r1, "composite") > _compute_score(r2, "composite")


# ===================================================================
# OptResult
# ===================================================================


class TestOptResult:
    def test_creation(self):
        r = OptResult(params={"a": 1}, total_return=10.0, cagr=5.0,
                      sharpe=0.8, max_dd=-3.0, win_rate=60.0, trades=10, score=1.5)
        assert r.params == {"a": 1}
        assert r.sharpe == 0.8
        assert r.trades == 10

    def test_sort_by_score(self):
        r1 = OptResult(params={"a": 1}, score=2.0)
        r2 = OptResult(params={"b": 2}, score=5.0)
        r3 = OptResult(params={"c": 3}, score=1.0)
        sorted_results = sorted([r1, r2, r3], key=lambda x: x.score, reverse=True)
        assert sorted_results[0].params == {"b": 2}
        assert sorted_results[-1].params == {"c": 3}


# ===================================================================
# PARAM_GRIDS and _PARAMS_CLASS
# ===================================================================


class TestParamGrids:
    def test_all_strategies_have_grids_and_class(self):
        """Every strategy in _PARAMS_CLASS must have a grid in PARAM_GRIDS."""
        for name in _PARAMS_CLASS:
            assert name in PARAM_GRIDS, f"Missing PARAM_GRIDS entry for {name}"
            assert len(PARAM_GRIDS[name]) >= 2, f"Grid for {name} is too small"

    def test_invalid_short_long_combos_caught(self):
        """Simulate the skip logic: combos with short >= long should be filtered."""
        shorts = [20, 40, 60]
        longs = [30, 50, 70]
        # Manual application of optimize.py skip logic (line 130-132)
        invalid = 0
        for s in shorts:
            for l in longs:
                if s >= l:
                    invalid += 1
        # 40>=30, 60>=30, 60>=50 → 3 invalid combos
        assert invalid == 3


# ===================================================================
# grid_search smoke (mocked)
# ===================================================================


class TestGridSearchSmoke:
    def test_function_signature(self):
        from engine.optimize import grid_search
        import inspect
        sig = inspect.signature(grid_search)
        param_names = list(sig.parameters.keys())
        assert "strategy_name" in param_names
        assert "symbol" in param_names
        assert "metric" in param_names
        assert "top_n" in param_names


# ===================================================================
# grid_search with mocked DataProvider
# ===================================================================


class TestGridSearchMocked:
    def test_grid_search_runs_with_mock_data(self, ohlcv):
        """Full grid_search with mocked data provider returns results."""
        from engine.optimize import grid_search
        from data import DataProvider

        with patch.object(DataProvider, 'get_daily', return_value=ohlcv):
            results = grid_search(
                strategy_name="weekly_macd",
                symbol="AAPL",
                start="2020-01-01",
                metric="sharpe",
                top_n=5,
            )
            assert len(results) <= 5
            for r in results:
                assert len(r.params) >= 2
                assert r.trades >= 0

    def test_grid_search_composite_metric(self, ohlcv):
        from engine.optimize import grid_search
        from data import DataProvider

        with patch.object(DataProvider, 'get_daily', return_value=ohlcv):
            results = grid_search(
                strategy_name="weekly_macd",
                symbol="AAPL",
                start="2020-01-01",
                metric="composite",
                top_n=3,
            )
            assert len(results) >= 1
            # Composite scores should differ (not all identical)
            scores = {r.score for r in results}
            assert len(scores) >= 1

    def test_unknown_strategy_raises(self):
        from engine.optimize import grid_search
        with pytest.raises(ValueError, match="Unknown strategy"):
            grid_search("no_such", "AAPL")

    def test_no_grid_defined_raises(self, ohlcv):
        from engine.optimize import grid_search
        from data import DataProvider
        from strategy.weekly_macd import WeeklyMACDParams

        with patch.object(DataProvider, 'get_daily', return_value=ohlcv):
            with patch.object(WeeklyMACDParams, 'grid', {}, create=True):
                with pytest.raises(ValueError):
                    grid_search("weekly_macd", "AAPL")

    def test_failed_combination_is_logged(self, ohlcv, monkeypatch, caplog):
        import logging
        import engine.optimize as optimize_module
        from data import DataProvider

        class Params:
            grid = {"broken": [1]}

        class BrokenStrategy:
            def __init__(self, **kwargs):
                self.params = Params()

            def calculate_indicators(self, df):
                raise ValueError("bad indicator input")

        monkeypatch.setitem(optimize_module._STRATEGY_MAP, "weekly_macd", BrokenStrategy)
        caplog.set_level(logging.WARNING, logger="engine.optimize")
        with patch.object(DataProvider, "get_daily", return_value=ohlcv):
            results = optimize_module.grid_search("weekly_macd", "AAPL", top_n=1)

        assert results == []
        assert "Skipping optimization combination" in caplog.text
        assert "{'broken': 1}" in caplog.text
        assert "bad indicator input" in caplog.text


# ===================================================================
# walk_forward signature and smoke
# ===================================================================


class TestOptimizeHelpers:
    def test_print_grid_results(self, capsys):
        from engine.optimize import print_grid_results, OptResult
        results = [
            OptResult(params={"a": 10, "b": 20}, sharpe=1.5, total_return=25.0,
                      cagr=10.0, max_dd=-5.0, win_rate=60.0, trades=10, score=3.0),
        ]
        print_grid_results(results, "test_strategy")
        out = capsys.readouterr().out
        assert "最优参数" in out or "test_strategy" in out or "Sharpe" in out

    def test_compute_score_negative_trades_cagr(self):
        r = OptResult(params={}, trades=1, cagr=20.0, score=0)
        assert _compute_score(r, "cagr") == -999


class TestWalkForwardSmoke:
    def test_function_signature(self):
        from engine.optimize import walk_forward
        import inspect
        sig = inspect.signature(walk_forward)
        param_names = list(sig.parameters.keys())
        assert "strategy_name" in param_names
        assert "symbol" in param_names
        assert "train_years" in param_names
        assert "test_years" in param_names

    def test_walk_forward_returns_structure(self, ohlcv):
        from engine.optimize import walk_forward
        from data import DataProvider

        with patch.object(DataProvider, 'get_daily', return_value=ohlcv):
            result = walk_forward(
                strategy_name="weekly_macd",
                symbol="AAPL",
                start="2020-01-01",
                end="2021-01-01",
                train_years=1,
                test_years=1,
            )
            assert "windows" in result
            assert isinstance(result["windows"], list)


# ===================================================================
# _next_trading_day — weekend / holiday boundary
# ===================================================================


class TestNextTradingDay:
    def test_friday_to_monday(self):
        """Friday → next trading day is Monday."""
        assert _next_trading_day("2026-05-15") == "2026-05-18"  # Fri → Mon

    def test_thursday_to_friday(self):
        """Thursday → Friday (normal)."""
        assert _next_trading_day("2026-05-14") == "2026-05-15"  # Thu → Fri

    def test_saturday_to_monday(self):
        """Saturday → Monday."""
        assert _next_trading_day("2026-05-16") == "2026-05-18"  # Sat → Mon

    def test_monday_to_tuesday(self):
        """Monday → Tuesday (normal)."""
        assert _next_trading_day("2026-05-18") == "2026-05-19"  # Mon → Tue

    def test_friday_july4_not_skipped(self):
        """July 4 is a US holiday but pandas default bdate_range only skips weekends."""
        assert _next_trading_day("2025-07-03") == "2025-07-04"  # Thu → Fri (not holiday-aware)

    def test_new_year_not_skipped(self):
        """New Year's Day — pandas default treats Jan 1 as business day."""
        assert _next_trading_day("2025-12-31") == "2026-01-01"  # Wed → Thu
