"""Diagnose turtle_trading per-symbol performance on the current watchlist.

What we want to see:
- How many trades per symbol (zero = trend_filter too strict)
- Win rate
- CAGR vs buy-and-hold (does it beat or lag?)
- MaxDD
- Average holding period (whipsaws?)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.bootstrap import setup_runtime
setup_runtime()

import numpy as np

from data import DataProvider
from engine.trader import BacktestEngine
from strategy.turtle_trading import TurtleTrading


SYMBOLS = ["AAPL", "NVDA", "TSLA", "GOOG", "AMZN", "MU", "INTC", "ORCL", "QQQ", "SPY", "SMH"]


def run_for(strategy_cls, name, **kwargs):
    print(f"\n{'='*110}")
    print(f"  {name}")
    print(f"{'='*110}")
    print(f"  {'symbol':<8s} {'trades':>6s} {'wr':>6s} {'CAGR':>8s} {'B&H':>8s} {'excess':>8s} {'MaxDD':>8s} {'PF':>6s} {'avg_hold':>8s}")
    print("  " + "-" * 76)

    provider = DataProvider()
    totals = {"trades": 0, "wins": 0, "excess_sum": 0, "n": 0}

    for sym in SYMBOLS:
        df = provider.get_daily(sym, start="2018-01-01", end="2024-12-31")
        if df is None or df.empty or len(df) < 200:
            print(f"  {sym:<8s}  no data")
            continue
        strat = strategy_cls(**kwargs)
        df_sig = strat.calculate_indicators(df)
        engine = BacktestEngine(initial_capital=10000)
        bench = engine.run(strat, df_sig)
        r = engine.get_result(bench)
        excess = r.cagr_pct - (r.buy_hold_return_pct / 7)  # crude annualisation
        avg_hold = np.mean([t.holding_days for t in r.trades]) if r.trades else 0
        wr_str = f"{r.win_rate_pct:>5.0f}%" if r.trades else "  n/a"
        print(
            f"  {sym:<8s} {r.total_trades:>6d} {wr_str} "
            f"{r.cagr_pct:>+7.1f}% {r.buy_hold_return_pct/7:>+7.1f}% "
            f"{excess:>+7.1f}% {r.max_drawdown_pct:>+7.1f}% "
            f"{r.profit_factor:>6.2f} {avg_hold:>7.1f}d"
        )
        totals["trades"] += r.total_trades
        totals["wins"] += r.winning_trades
        totals["excess_sum"] += excess
        totals["n"] += 1

    print("  " + "-" * 76)
    if totals["n"] > 0:
        avg_excess = totals["excess_sum"] / totals["n"]
        avg_wr = totals["wins"] / totals["trades"] * 100 if totals["trades"] else 0
        print(
            f"  {'AVG':<8s} {totals['trades']:>6d} {avg_wr:>5.0f}% "
            f"{'':>8s} {'':>8s} {avg_excess:>+7.1f}%"
        )


def main():
    # Current default config
    run_for(TurtleTrading, "Current defaults (short=20 long=50 channel=20 trail=3.0 trend_filter=True)")

    # Off trend filter
    run_for(TurtleTrading, "trend_filter=OFF", trend_filter=False)

    # Shorter channel
    run_for(TurtleTrading, "channel_period=10 (shorter, more entries)", channel_period=10)

    # Tighter trail
    run_for(TurtleTrading, "trail_atr_mult=2.0 (tighter stops)", trail_atr_mult=2.0)

    # Combo: off filter + shorter channel
    run_for(
        TurtleTrading,
        "trend_filter=OFF + channel=10",
        trend_filter=False, channel_period=10,
    )


if __name__ == "__main__":
    main()
