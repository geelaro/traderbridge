"""One-shot: compare factor attribution across strategies on the current watchlist."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.bootstrap import setup_runtime
setup_runtime()

from engine.portfolio import Leg, PortfolioBacktest
from analysis.factor_returns import FactorReturns
from analysis.factor_attribution import FactorAttribution


SYMBOLS = ["AAPL", "NVDA", "TSLA", "GOOG", "AMZN", "MU", "INTC", "ORCL", "QQQ", "SPY", "SMH"]
STRATEGIES = ["weekly_macd_kdj", "turtle_trading", "spy_ma_breakout", "rsi2_mean_reversion"]


def main():
    print("Loading factor returns (cached)...")
    factors = FactorReturns(mode="full").load("2018-01-01", "2024-12-31")
    print(f"  OK: {len(factors)} bars")

    print()
    header = (
        f"  {'strategy':<22s} {'Sharpe':>7s} {'CAGR':>8s} {'MaxDD':>8s}  "
        f"{'R2':>5s}  {'alpha_y':>8s}  {'a_t':>6s}  significant_betas"
    )
    print("=" * 120)
    print(header)
    print("=" * 120)

    for strat in STRATEGIES:
        legs = [Leg(s, strat) for s in SYMBOLS]
        bt = PortfolioBacktest(legs=legs, initial_capital=100000, allocation="equal")
        try:
            result = bt.run(start="2018-01-01", end="2024-12-31")
        except Exception as e:
            print(f"  {strat:<22s}  backtest failed: {e}")
            continue

        if result.equity_curve.empty or len(result.equity_curve) < 30:
            print(f"  {strat:<22s}  no equity data")
            continue

        try:
            attr = FactorAttribution(result.equity_curve, factors)
            res = attr.regress()
        except Exception as e:
            print(f"  {strat:<22s}  attribution failed: {e}")
            continue

        sig_betas = [
            f"{n}={res.betas[n]:+.2f}(t={res.beta_tstats[n]:+.1f})"
            for n in res.factor_names if abs(res.beta_tstats[n]) >= 2
        ]
        sig_str = ", ".join(sig_betas[:4]) if sig_betas else "-none-"

        print(
            f"  {strat:<22s} {result.sharpe_ratio:>7.2f} {result.cagr_pct:>+7.1f}% "
            f"{result.max_drawdown_pct:>7.1f}%  {res.r_squared:>5.2f}  "
            f"{res.alpha_annual * 100:>+7.1f}%  {res.alpha_tstat:>+6.2f}  {sig_str}"
        )

    print("=" * 120)
    print()
    print("Verdict legend:")
    print("  |a_t| > 2 AND R2 < 0.7   -> real alpha")
    print("  |a_t| > 2 AND R2 > 0.85  -> suspicious (alpha + high R2 = factor mix)")
    print("  |a_t| < 2 AND R2 > 0.85  -> no alpha (factor exposure only)")
    print("  |a_t| < 2 AND R2 < 0.7   -> weak signal (insufficient sample or unstable)")


if __name__ == "__main__":
    main()
