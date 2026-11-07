"""Experiment: how much alpha is the MACD death-cross exit eating?

Compares four exit strategies on the same KDJ-golden-cross entry:
    A. default weekly_macd_kdj          — MACD death cross exits
    B. + ATR trailing stop (5×ATR)      — death cross OR ATR stop
    C. slower MACD (slow=52)            — pushes death-cross trigger out
    D. NEVER exit (alpha ceiling)       — exits only on backtest close-out

Run with:
    pipenv run python scripts/exit_experiment.py
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.bootstrap import setup_runtime
setup_runtime()

import numpy as np
import pandas as pd

from engine.portfolio import Leg, PortfolioBacktest
from strategy.macd_kdj import MACDKDJStrategy
from data import DataProvider


SYMBOLS = ["AAPL", "NVDA", "TSLA", "GOOG", "AMZN", "MU", "INTC", "ORCL", "QQQ", "SPY", "SMH", "MSFT"]
START, END = "2018-01-01", "2026-05-29"


def run_portfolio(strategy_kwargs: dict, label: str, no_exit: bool = False):
    """Run an equal-weight portfolio backtest on the watchlist."""
    legs = [Leg(s, "weekly_macd_kdj", params=strategy_kwargs) for s in SYMBOLS]
    bt = PortfolioBacktest(legs=legs, initial_capital=100000, allocation="equal")

    if no_exit:
        # Monkey-patch check_exit to never exit (alpha-ceiling test)
        orig = MACDKDJStrategy.check_exit
        MACDKDJStrategy.check_exit = lambda self, *a, **kw: (False, "")
        try:
            result = bt.run(start=START, end=END)
        finally:
            MACDKDJStrategy.check_exit = orig
    else:
        result = bt.run(start=START, end=END)

    avg_hold = (
        sum(t.hold_days or 0 for t in result.closed_trades) / max(len(result.closed_trades), 1)
    )
    return {
        "label": label,
        "sharpe": result.sharpe_ratio,
        "cagr": result.cagr_pct,
        "mdd": result.max_drawdown_pct,
        "trades": result.total_trades,
        "wr": result.win_rate_pct,
        "avg_hold": avg_hold,
    }


def buy_and_hold_eq_weight():
    """Equal-weight buy-and-hold benchmark."""
    provider = DataProvider()
    closes = {sym: provider.get_daily(sym, start=START, end=END)["Close"] for sym in SYMBOLS}
    prices = pd.DataFrame(closes).sort_index().ffill().dropna()
    n = len(prices.columns)
    shares = (100000 / n) / prices.iloc[0]
    eq = (prices * shares).sum(axis=1)
    rets = eq.pct_change(fill_method=None).dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = ((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1) * 100
    sharpe = float(np.sqrt(252) * rets.mean() / rets.std())
    mdd = float(((eq - eq.expanding().max()) / eq.expanding().max() * 100).min())
    return {
        "label": "Equal-weight B&H (ceiling)",
        "sharpe": sharpe, "cagr": cagr, "mdd": mdd,
        "trades": 0, "wr": 0, "avg_hold": (eq.index[-1] - eq.index[0]).days,
    }


def main():
    print(f"\nExit-rule experiment on weekly_macd_kdj entry ({len(SYMBOLS)} symbols, {START} ~ {END})")
    print("=" * 110)
    print(f"  {'strategy':<46s} {'CAGR':>8s} {'Sharpe':>7s} {'MaxDD':>8s} "
          f"{'trades':>7s} {'wr%':>5s} {'avg_hold':>9s}")
    print("-" * 110)

    rows = []
    rows.append(run_portfolio({}, "A. 默认 (MACD死叉出场)"))
    rows.append(run_portfolio(
        {"use_atr_stop": True, "trail_atr_mult": 5.0},
        "B. + ATR止损 trail=5",
    ))
    rows.append(run_portfolio(
        {"macd_slow": 52},
        "C. MACD 慢一倍 (slow=52)",
    ))
    rows.append(run_portfolio({}, "D. KDJ入场 + 不主动出场 (alpha 上限)", no_exit=True))
    rows.append(buy_and_hold_eq_weight())

    for r in rows:
        print(f"  {r['label']:<46s} {r['cagr']:>+7.1f}% {r['sharpe']:>7.2f} "
              f"{r['mdd']:>+7.1f}% {r['trades']:>7d} {r['wr']:>4.0f}% {r['avg_hold']:>8.0f}d")
    print("=" * 110)


if __name__ == "__main__":
    main()
