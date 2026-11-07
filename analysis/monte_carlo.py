"""Monte Carlo simulation — shuffle trade sequences to test drawdown sensitivity.

Usage:
    pipenv run python analysis/monte_carlo.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

if __name__ == "__main__":
    from utils.bootstrap import setup_runtime
    setup_runtime()

import numpy as np
import matplotlib.pyplot as plt

from engine.portfolio import PortfolioBacktest, Leg


def run(
    symbols: list = None,
    strategy: str = "weekly_macd_kdj",
    start: str = "2018-01-01",
    end: str = "2026-05-19",
    initial_capital: float = 100000,
    n_sims: int = 2000,
    seed: int = 42,
) -> dict:
    """Run Monte Carlo drawdown simulation by shuffling trade sequences.

    Parameters
    ----------
    symbols : list[str] | None
        List of ticker symbols (default: US tech watchlist).
    strategy : str
        Strategy name applied to all symbols.
    start, end : str
        Backtest date range (YYYY-MM-DD).
    initial_capital : float
        Portfolio initial capital.
    n_sims : int
        Number of Monte Carlo permutations.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys: trades, win_rate, avg_pnl, pnl_std, max_dds, end_equities,
    expected_return, percentiles, dd_probs
    """
    if symbols is None:
        symbols = ["AAPL", "NVDA", "TSLA", "GOOG", "AMZN", "MU", "INTC", "ORCL", "QQQ", "SPY"]

    legs = [Leg(sym, strategy) for sym in symbols]
    pf = PortfolioBacktest(legs, initial_capital=initial_capital)
    result = pf.run(start=start, end=end)

    # Use trade PnL as % of initial capital (realistic portfolio drawdown)
    initial_cap = result.initial_capital
    pnl_pcts = np.array([
        t.pnl / initial_cap * 100 for t in result.trades if t.pnl is not None
    ])
    if len(pnl_pcts) == 0:
        print("无交易数据")
        return {"trades": 0, "error": "no trades"}

    print(f"交易总数: {len(pnl_pcts)}")
    win_rate = (pnl_pcts > 0).sum() / len(pnl_pcts) * 100
    print(f"胜率: {win_rate:.1f}%")
    print(f"平均单笔盈亏: {pnl_pcts.mean():+.2f}% (占初始资金)")
    print(f"单笔标准差: {pnl_pcts.std():.2f}%")

    # Monte Carlo: shuffle trade order, track drawdowns
    if n_sims is None:
        n_sims = 2000
    max_dds = []
    end_equities = []

    rng = np.random.default_rng(seed)
    for _ in range(n_sims):
        shuffled = rng.permutation(pnl_pcts)
        equity = 100.0
        peak = 100.0
        max_dd = 0.0
        for pnl in shuffled:
            equity *= (1 + pnl / 100)
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd
        max_dds.append(max_dd)
        end_equities.append(equity)

    max_dds = np.array(max_dds)
    end_equities = np.array(end_equities)
    final_return = (end_equities.mean() - 100)

    print(f"\nMonte Carlo ({n_sims} 次打乱):")
    print(f"  预期最终收益: {final_return:+.1f}%")
    print(f"  最大回撤分布:")
    for pct in [5, 25, 50, 75, 95]:
        d = np.percentile(max_dds, pct)
        print(f"    P{pct:02d}: {d:.1f}%")

    # Probability of hitting various drawdown levels
    print(f"\n  回撤概率:")
    for threshold in [20, 30, 40, 50, 60]:
        prob = (max_dds >= threshold).sum() / n_sims * 100
        print(f"    回撤 ≥ {threshold}%:  {prob:.1f}%")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(max_dds, bins=40, color="#d62728", edgecolor="white", alpha=0.8)
    ax.axvline(max_dds.mean(), color="blue", linestyle="--", linewidth=1.5,
               label=f"均值 {max_dds.mean():.1f}%")
    ax.axvline(np.percentile(max_dds, 95), color="orange", linestyle="--", linewidth=1.5,
               label=f"P95 {np.percentile(max_dds, 95):.1f}%")
    ax.set_xlabel("最大回撤 %")
    ax.set_ylabel("频次")
    ax.set_title(f"Monte Carlo 最大回撤分布 ({n_sims}次打乱交易序列)", fontsize=11)
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = Path("reports/monte_carlo.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120)
    print(f"\n图表已保存: {path}")

    percentiles = {f"P{pct:02d}": float(np.percentile(max_dds, pct))
                   for pct in [5, 25, 50, 75, 95]}
    dd_probs = {}
    for threshold in [20, 30, 40, 50, 60]:
        dd_probs[threshold] = float((max_dds >= threshold).sum() / n_sims * 100)

    return {
        "trades": len(pnl_pcts),
        "win_rate": float(win_rate),
        "avg_pnl": float(pnl_pcts.mean()),
        "pnl_std": float(pnl_pcts.std()),
        "expected_return": float(final_return),
        "max_dds": max_dds.tolist(),
        "end_equities": end_equities.tolist(),
        "percentiles": percentiles,
        "dd_probs": dd_probs,
    }


if __name__ == "__main__":
    run()
