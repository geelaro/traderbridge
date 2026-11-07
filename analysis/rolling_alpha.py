"""Rolling-window Jensen's α — does the strategy generate real alpha,
or is the equity curve just factor exposure?

**Breaking change (2026-05-31)**: this module used to report the difference
between a strategy and its buy-and-hold benchmark (active return). It now
runs a Newey-West-corrected OLS regression against a factor model
(:class:`analysis.factor_returns.FactorReturns`) and reports the **Jensen
intercept α** instead, plus its t-statistic and the model R².

If you need the old behaviour (strategy_return − benchmark_return), the
analogue is ``r.total_return_pct − r.buy_hold_return_pct`` on any
`BacktestResult`.

Usage:
    pipenv run python analysis/rolling_alpha.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

if __name__ == "__main__":
    from utils.bootstrap import setup_runtime
    setup_runtime()

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from engine.portfolio import Leg, PortfolioBacktest
from analysis.factor_returns import FactorReturns
from analysis.factor_attribution import FactorAttribution, TRADING_DAYS

logger = logging.getLogger(__name__)


def run(
    strategy: str = "weekly_macd_kdj",
    symbols: list = None,
    start: str = "2015-01-01",
    end: str = "2024-06-01",
    window_days: int = TRADING_DAYS,
    initial_capital: float = 100000,
    factor_mode: str = "full",
    save_chart: bool = True,
) -> pd.DataFrame:
    """Run rolling-window Jensen alpha analysis on a portfolio backtest.

    Parameters
    ----------
    strategy : str
        Strategy name (key of STRATEGY_MAP).
    symbols : list[str] | None
        List of ticker symbols (default: US tech watchlist).
    start, end : str
        Backtest date range. The factor model imposes its own inception
        floor (2013-08 for mode='full', 2000-06 for mode='ff3').
    window_days : int
        Rolling window size in trading days (default 252 ≈ 1 year).
    initial_capital : float
        Starting capital for the PortfolioBacktest.
    factor_mode : str
        ``"full"`` 6-factor (MKT/SMB/HML/MOM/QMJ/BAB) or ``"ff3"``
        (MKT/SMB/HML only).
    save_chart : bool
        If True, write rolling-α chart to reports/rolling_alpha.png.

    Returns
    -------
    pd.DataFrame
        Indexed by window-end date with columns:
        alpha_daily, alpha_annual, alpha_tstat, r_squared.
    """
    if symbols is None:
        symbols = ["AAPL", "NVDA", "TSLA", "GOOG", "AMZN", "MU", "INTC", "ORCL", "QQQ", "SPY"]

    # 1. Portfolio backtest → single equity curve
    legs = [Leg(sym, strategy) for sym in symbols]
    bt = PortfolioBacktest(
        legs=legs,
        initial_capital=initial_capital,
        allocation="equal",
    )
    result = bt.run(start=start, end=end)
    equity = result.equity_curve
    if equity is None or equity.empty:
        print("No equity curve — backtest produced no data")
        return pd.DataFrame()

    # 2. Load factor returns aligned to the backtest period
    factors = FactorReturns(mode=factor_mode).load(start, end)
    if factors.empty:
        print("Factor data unavailable — cannot run attribution")
        return pd.DataFrame()

    # 3. Rolling regression
    attr = FactorAttribution(equity, factors)
    try:
        rolling = attr.rolling_alpha(window_days=window_days)
    except ValueError as e:
        print(f"Rolling alpha failed: {e}")
        return pd.DataFrame()

    # 4. Print head/tail summary
    if not rolling.empty:
        print(f"\n滚动 Jensen α (window={window_days} 个交易日, factor mode={factor_mode})")
        print(f"窗口数: {len(rolling)}   最早: {rolling.index[0].date()}   最新: {rolling.index[-1].date()}\n")
        # Sample 5 evenly-spaced rows for terminal preview
        sample_n = min(8, len(rolling))
        sample = rolling.iloc[np.linspace(0, len(rolling) - 1, sample_n).astype(int)]
        formatted = sample.copy()
        formatted["alpha_annual"] = formatted["alpha_annual"].map("{:+.2%}".format)
        formatted["alpha_tstat"] = formatted["alpha_tstat"].map("{:+.2f}".format)
        formatted["r_squared"] = formatted["r_squared"].map("{:.3f}".format)
        formatted = formatted[["alpha_annual", "alpha_tstat", "r_squared"]]
        print(formatted.to_string())

    # 5. Plot
    if save_chart and not rolling.empty:
        _plot(rolling, strategy, window_days)

    return rolling


def _plot(rolling: pd.DataFrame, strategy: str, window_days: int) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Top: rolling annualised alpha
    ax = axes[0]
    alpha_pct = rolling["alpha_annual"] * 100
    colors = ["#2ca02c" if a >= 0 else "#d62728" for a in alpha_pct]
    ax.bar(rolling.index, alpha_pct, color=colors, width=10)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("年化 α (%)")
    ax.set_title(
        f"{strategy} 滚动 Jensen α (window={window_days}d)",
        fontsize=11,
    )
    # Trend line
    if len(alpha_pct) >= 3:
        x = np.arange(len(alpha_pct))
        z = np.polyfit(x, alpha_pct.values, 1)
        ax.plot(rolling.index, np.poly1d(z)(x), "b--", linewidth=1,
                label=f"趋势 (slope={z[0]:.3f}%/window)")
        ax.legend(fontsize=8)

    # Bottom: t-stat — anything outside ±2 is statistically significant
    ax = axes[1]
    ax.plot(rolling.index, rolling["alpha_tstat"], color="#1f77b4", linewidth=1.2)
    ax.axhline(2, color="gray", linestyle="--", linewidth=0.5, label="±2 显著阈")
    ax.axhline(-2, color="gray", linestyle="--", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("α t-stat")
    ax.set_xlabel("Date")
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = Path("reports/rolling_alpha.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120)
    print(f"\n图表已保存: {path}")


if __name__ == "__main__":
    run()
