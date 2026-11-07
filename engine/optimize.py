"""Parameter optimization — grid search & walk-forward validation.

Usage:
    pipenv run python engine/optimize.py -s trend_follower -symbol AAPL
    pipenv run python engine/optimize.py -s weekly_macd -symbol 510300
    pipenv run python engine/optimize.py -s enhanced_macd --walk-forward
"""

import argparse
import itertools
import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

if __name__ == "__main__":
    from utils.bootstrap import setup_runtime
    setup_runtime()

from utils.font import setup_chinese_font
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from data import DataProvider
from strategy import STRATEGY_MAP as _STRATEGY_MAP
from engine.trader import BacktestEngine

# ---------------------------------------------------------------------------
def _next_trading_day(date_str: str) -> str:
    """Return the next business day after *date_str* (weekends only; US holidays not skipped)."""
    next_day = pd.Timestamp(date_str) + pd.Timedelta(days=1)
    return pd.bdate_range(start=next_day, periods=1)[0].strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Grid / params access
# ---------------------------------------------------------------------------

def _get_params_grid(strategy_name: str) -> dict:
    """Return the param-grid dict declared on the strategy's params class."""
    strategy = _STRATEGY_MAP[strategy_name]()
    params_cls = type(strategy.params)
    return getattr(params_cls, "grid", {})

def _get_params_cls(strategy_name: str):
    strategy = _STRATEGY_MAP[strategy_name]()
    return type(strategy.params)

# Backward-compatible module-level lookups (computed from strategy classes)
PARAM_GRIDS = {
    name: _get_params_grid(name) for name in _STRATEGY_MAP
}
_PARAMS_CLASS = {
    name: _get_params_cls(name) for name in _STRATEGY_MAP
}


# ---------------------------------------------------------------------------
# Optimization result
# ---------------------------------------------------------------------------

@dataclass
class OptResult:
    params: dict
    total_return: float = 0
    cagr: float = 0
    sharpe: float = 0
    max_dd: float = 0
    win_rate: float = 0
    trades: int = 0
    score: float = 0  # composite


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------


def grid_search(
    strategy_name: str,
    symbol: str,
    start: str = "2018-01-01",
    end: Optional[str] = None,
    initial_capital: float = 10000,
    metric: str = "sharpe",
    top_n: int = 10,
) -> list[OptResult]:
    """Exhaustive grid search over the strategy's declared param grid."""

    if strategy_name not in _STRATEGY_MAP:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    strategy_cls = _STRATEGY_MAP[strategy_name]
    params_cls = _get_params_cls(strategy_name)
    grid = _get_params_grid(strategy_name)
    if not grid:
        raise ValueError(f"No param grid defined for {strategy_name}")

    if end is None:
        end = date.today().isoformat()

    print(f"获取 {symbol} 数据 ...")
    provider = DataProvider()
    df = provider.get_daily(symbol, start=start, end=end)
    if df is None or df.empty:
        raise RuntimeError(f"无法获取 {symbol} 数据")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    keys = list(grid.keys())
    combinations = list(itertools.product(*grid.values()))
    total = len(combinations)
    print(f"策略: {strategy_name}  标的: {symbol}  参数组合: {total}")
    print(f"优化指标: {metric}  数据: {len(df)} 根K线\n")

    results = []
    for combo in tqdm(combinations, desc="搜索中", unit="组"):
        params = dict(zip(keys, combo))
        # Skip invalid combos
        if "short_ma" in params and "long_ma" in params:
            if params["short_ma"] >= params["long_ma"]:
                continue

        try:
            strategy = strategy_cls(**params)
            df_sig = strategy.calculate_indicators(df)
            engine = BacktestEngine(initial_capital=initial_capital)
            bench = engine.run(strategy, df_sig)
            r = engine.get_result(bench)
        except Exception:
            continue

        opt = OptResult(
            params=params,
            total_return=r.total_return_pct,
            cagr=r.cagr_pct,
            sharpe=r.sharpe_ratio,
            max_dd=r.max_drawdown_pct,
            win_rate=r.win_rate_pct,
            trades=r.total_trades,
        )
        # Composite score: penalize low trade count
        opt.score = _compute_score(opt, metric)
        results.append(opt)

    results.sort(key=lambda x: x.score, reverse=True)
    return results[:top_n]


def _compute_score(r: OptResult, metric: str) -> float:
    """Composite score — reward high sharpe/return, penalize low trades."""
    if r.trades < 3:
        return -999
    if metric == "sharpe":
        return r.sharpe
    elif metric == "cagr":
        return r.cagr
    elif metric == "total_return":
        return r.total_return
    # composite: sharpe + return bonus, penalize drawdown
    return r.sharpe * 2 + (r.cagr / 100) - abs(r.max_dd / 100)


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------


def walk_forward(
    strategy_name: str,
    symbol: str,
    start: str = "2015-01-01",
    end: Optional[str] = None,
    train_years: int = 3,
    test_years: int = 1,
    initial_capital: float = 10000,
    metric: str = "sharpe",
) -> dict:
    """Rolling walk-forward optimization.

    For each window:
      1. Optimize params on [window_start, window_start + train_years]
      2. Run best params on (train_end + 1d, train_end + test_years]
         — test window starts one day after training to prevent data leakage.
      3. Slide forward by test_years. Capital carries forward across windows
         so the OOS equity curve is naturally continuous.
    """

    if end is None:
        end = date.today().isoformat()

    end_dt = pd.Timestamp(end)
    window_start = pd.Timestamp(start)
    all_oos_trades = []
    all_oos_equity = []
    windows = []
    current_capital = initial_capital

    while window_start + pd.DateOffset(years=train_years + test_years) <= end_dt:
        train_start = window_start.strftime("%Y-%m-%d")
        train_end = (window_start + pd.DateOffset(years=train_years)).strftime("%Y-%m-%d")
        test_start = _next_trading_day(train_end)
        test_end = (window_start + pd.DateOffset(years=train_years + test_years)).strftime("%Y-%m-%d")

        print(f"\n{'=' * 60}")
        print(f"  训练: {train_start} ~ {train_end}  测试: {test_start} ~ {test_end}")
        print(f"{'=' * 60}")

        # Optimize on training window
        best = grid_search(
            strategy_name, symbol,
            start=train_start, end=train_end,
            initial_capital=current_capital,
            metric=metric, top_n=1,
        )
        if not best:
            print("  ! 训练窗口无有效结果，跳过")
            window_start += pd.DateOffset(years=test_years)
            continue

        best_params = best[0].params
        print(f"  最优参数: {best_params}")
        print(f"  训练集Sharpe: {best[0].sharpe:.2f}  收益: {best[0].total_return:+.1f}%")

        # Run on test window (carry-forward capital from previous window)
        strategy_cls = _STRATEGY_MAP[strategy_name]
        try:
            strategy = strategy_cls(**best_params)
            provider = DataProvider()
            df = provider.get_daily(symbol, start=test_start, end=test_end)
            if df is None or df.empty:
                window_start += pd.DateOffset(years=test_years)
                continue
            df = df.dropna(subset=["Open", "High", "Low", "Close"])
            df_sig = strategy.calculate_indicators(df)
            engine = BacktestEngine(initial_capital=current_capital)
            bench = engine.run(strategy, df_sig)
            r = engine.get_result(bench)
        except Exception:
            window_start += pd.DateOffset(years=test_years)
            continue

        print(f"  测试集: 收益 {r.total_return_pct:+.1f}%  Sharpe {r.sharpe_ratio:.2f}  "
              f"回撤 {r.max_drawdown_pct:.1f}%  交易 {r.total_trades} 笔")
        all_oos_trades.extend(engine.trades)
        all_oos_equity.extend(engine.equity_history)
        windows.append({
            "train_start": train_start, "train_end": train_end,
            "test_start": test_start, "test_end": test_end,
            "best_params": best_params,
            "test_return": r.total_return_pct, "test_sharpe": r.sharpe_ratio,
            "test_dd": r.max_drawdown_pct, "trades": r.total_trades,
        })
        current_capital = r.final_equity
        window_start += pd.DateOffset(years=test_years)

    # Aggregate OOS performance
    if not windows:
        print("\n无有效窗口")
        return {"windows": []}

    # Equity curve is naturally continuous (capital carries forward across windows)
    eq_df = pd.DataFrame(all_oos_equity, columns=["date", "equity"]).set_index("date")
    eq_df = eq_df.sort_index()
    curve = eq_df["equity"]
    rets = curve.pct_change(fill_method=None).dropna()
    total_ret = (curve.iloc[-1] / initial_capital - 1) * 100
    sharpe = np.sqrt(252) * rets.mean() / rets.std() if rets.std() > 0 else 0
    rolling_max = curve.expanding().max()
    max_dd = ((curve - rolling_max) / rolling_max * 100).min()

    print(f"\n{'=' * 60}")
    print(f"  Walk-forward 汇总 ({len(windows)} 个窗口)")
    print(f"{'=' * 60}")
    print(f"  OOS 总收益: {total_ret:+.1f}%")
    print(f"  OOS Sharpe: {sharpe:.2f}")
    print(f"  OOS 最大回撤: {max_dd:.1f}%")
    print(f"  OOS 总交易: {len(all_oos_trades)} 笔")
    print()

    # Print parameter stability
    print("  各窗口最优参数:")
    print(f"  {'训练期':<24} {'参数':<60} {'测试收益':>10}")
    print(f"  {'─' * 94}")
    for w in windows:
        param_str = ", ".join(f"{k}={v}" for k, v in w["best_params"].items())
        print(f"  {w['train_start']}~{w['train_end']:<10}  {param_str:<60}  {w['test_return']:>+9.1f}%")

    return {"windows": windows, "total_return": total_ret, "sharpe": sharpe, "max_dd": max_dd}


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------


def print_grid_results(results: list[OptResult], strategy_name: str):
    print(f"\n{'=' * 90}")
    print(f"  最优参数 — {strategy_name}")
    print(f"{'=' * 90}")
    header = f"  {'排名':<4} {'Sharpe':>7} {'收益':>8} {'CAGR':>7} {'回撤':>7} {'胜率':>6} {'交易':>5}  参数"
    print(header)
    print(f"  {'─' * 88}")
    for rank, r in enumerate(results, 1):
        params_str = ", ".join(f"{k}={v}" for k, v in r.params.items())
        print(f"  {rank:<4} {r.sharpe:>7.2f} {r.total_return:>+7.1f}% {r.cagr:>+6.1f}% "
              f"{r.max_dd:>6.1f}% {r.win_rate:>5.1f}% {r.trades:>5}  {params_str}")
    print()


def plot_heatmap(results: list[OptResult], strategy_name: str, x_key: str, y_key: str):
    """2D heatmap for two-parameter interaction."""
    try:
        setup_chinese_font()
    except Exception:
        pass

    # Pivot
    rows = []
    for r in results:
        row = {**r.params, "sharpe": r.sharpe, "cagr": r.cagr}
        rows.append(row)
    piv = pd.DataFrame(rows).pivot_table(index=y_key, columns=x_key, values="sharpe", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(piv.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(f"{strategy_name} — Sharpe Heatmap")
    plt.colorbar(im, ax=ax, label="Sharpe")
    plt.tight_layout()
    path = f"charts/heatmap_{strategy_name}_{x_key}_{y_key}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    print(f"热力图已保存: {path}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="策略参数优化")
    parser.add_argument("--strategy", "-s", required=True, help="策略名称")
    parser.add_argument("--symbol", default="AAPL", help="标的 (默认 AAPL)")
    parser.add_argument("--start", default="2018-01-01", help="数据起始日")
    parser.add_argument("--end", default=None, help="数据截止日")
    parser.add_argument("--metric", default="sharpe", choices=["sharpe", "cagr", "total_return", "composite"])
    parser.add_argument("--top", type=int, default=10, help="显示前N个结果")
    parser.add_argument("--walk-forward", action="store_true", help="启用 Walk-forward 验证")
    parser.add_argument("--heatmap", action="store_true", help="绘制二维参数热力图")
    args = parser.parse_args()

    if args.walk_forward:
        walk_forward(
            strategy_name=args.strategy,
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            metric=args.metric,
        )
        return

    results = grid_search(
        strategy_name=args.strategy,
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        metric=args.metric,
        top_n=args.top,
    )
    print_grid_results(results, args.strategy)

    if args.heatmap and results:
        grid = _get_params_grid(args.strategy)
        keys = list(grid.keys())
        if len(keys) >= 2:
            plot_heatmap(results, args.strategy, keys[0], keys[1])


if __name__ == "__main__":
    main()
