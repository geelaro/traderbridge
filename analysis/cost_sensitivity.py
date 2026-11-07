"""Cost sensitivity analysis — grid-scan commission × slippage.

Usage:
    pipenv run python analysis/cost_sensitivity.py
    pipenv run python analysis/cost_sensitivity.py --strategy weekly_macd_kdj --symbol AAPL
"""

import argparse
import itertools
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    from utils.bootstrap import setup_runtime
    setup_runtime()

from utils.font import setup_chinese_font
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from engine.trader import run_backtest
from strategy import STRATEGY_MAP

# ---------------------------------------------------------------------------
# Default cost grids
# ---------------------------------------------------------------------------

COMMISSION_GRID = [0.0001, 0.0003, 0.001]   # 1bp, 3bp, 10bp
SLIPPAGE_GRID = [0.0001, 0.0005, 0.001, 0.002, 0.005]  # 1bp ~ 50bp

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pivot(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Pivot long-format grid result into a matrix (rows=commission, cols=slippage)."""
    return df.pivot(index="commission", columns="slippage", values=column)


# ---------------------------------------------------------------------------
# Feasibility rating
# ---------------------------------------------------------------------------

def rate_feasibility(df: pd.DataFrame) -> dict:
    """Grade strategy robustness to trading costs.

    Returns dict with keys: grade, summary, details.

    Rules (per commission tier, across all slippage levels):
        A — profitable at 10bp commission
        B — profitable at 3bp but not 10bp
        C — profitable at 1bp only
        D — unprofitable even at 1bp
    """
    tiers = sorted(df["commission"].unique())
    min_return_by_tier = {}
    for comm in tiers:
        subset = df[df["commission"] == comm]
        min_return_by_tier[comm] = subset["return_pct"].min()

    grade = "D"
    if min_return_by_tier.get(0.0001, -999) > 0:
        grade = "C"
    if min_return_by_tier.get(0.0003, -999) > 0:
        grade = "B"
    if min_return_by_tier.get(0.001, -999) > 0:
        grade = "A"

    # Sharpe modifier
    median_sharpe = df["sharpe"].median()
    if grade in ("A", "B") and median_sharpe >= 1.0:
        grade += "+"
    elif grade in ("C", "D") and median_sharpe < 0.5:
        grade += "-"

    tier_labels = {0.0001: "1bp", 0.0003: "3bp", 0.001: "10bp"}
    details = []
    for comm in tiers:
        min_r = min_return_by_tier[comm]
        status = "✓ 正收益" if min_r > 0 else "✗ 亏损"
        details.append(f"  {tier_labels.get(comm, comm)}: min return {min_r:+.1f}%  {status}")

    descriptions = {
        "A+": "极强 — 10bp佣金下全部滑点档位仍盈利，Sharpe≥1.0",
        "A": "强 — 10bp佣金下全部滑点档位仍盈利",
        "B+": "良好 — 3bp佣金下全部盈利，10bp出现亏损，Sharpe≥1.0",
        "B": "尚可 — 3bp佣金下全部盈利，10bp出现亏损",
        "C": "脆弱 — 仅1bp佣金下全部盈利，成本敏感",
        "C-": "很脆弱 — 仅最低成本盈利，且Sharpe<0.5",
        "D": "不可行 — 即使最低成本也出现亏损",
        "D-": "极差 — 全网格亏损，Sharpe<0.5",
    }

    return {
        "grade": grade,
        "summary": descriptions.get(grade, grade),
        "details": "\n".join(details),
    }


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(df: pd.DataFrame, strategy: str, symbol: str,
                 save_path: Optional[str] = None):
    """Plot dual heatmap: return_pct + sharpe over commission × slippage grid."""
    setup_chinese_font()

    metrics = [
        ("return_pct", "总收益率 (%)", "RdYlGn"),
        ("sharpe", "夏普比率", "YlOrBr"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    for ax, (col, title, cmap_name) in zip(axes, metrics):
        mat = _pivot(df, col)
        comm_labels = [f"{c*10000:.0f}bp" for c in mat.index]
        slip_labels = [f"{s*10000:.0f}bp" for s in mat.columns]

        im = ax.imshow(mat.values, aspect="auto", cmap=cmap_name, origin="lower")

        # Annotate cells
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat.iloc[i, j]
                text_color = "white" if col == "return_pct" and val < mat.values.mean() else "black"
                if col == "return_pct":
                    label = f"{val:.1f}%"
                else:
                    label = f"{val:.2f}"
                ax.text(j, i, label, ha="center", va="center", fontsize=10,
                        color=text_color, fontweight="bold")

        ax.set_xticks(range(len(slip_labels)))
        ax.set_xticklabels(slip_labels, fontsize=9)
        ax.set_yticks(range(len(comm_labels)))
        ax.set_yticklabels(comm_labels, fontsize=9)
        ax.set_xlabel("滑点 (Slippage)", fontsize=11)
        ax.set_ylabel("佣金 (Commission)", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")

        cbar = plt.colorbar(im, ax=ax, shrink=0.85)
        cbar.ax.tick_params(labelsize=8)

    rating = rate_feasibility(df)
    fig.suptitle(f"{strategy} / {symbol}  成本敏感性\n实盘评级: {rating['grade']} — {rating['summary']}",
                 fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"图表已保存: {save_path}")
    plt.close()
    return fig


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

def generate_report(df: pd.DataFrame, strategy: str, symbol: str,
                    output_dir: Optional[str] = None):
    """Generate CSV + PNG report under reports/."""
    out = Path(output_dir or PROJECT_ROOT / "reports")
    out.mkdir(parents=True, exist_ok=True)

    stem = f"cost_sensitivity_{strategy}_{symbol}"

    # CSV
    csv_path = out / f"{stem}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV 已保存: {csv_path}")

    # PNG
    png_path = out / f"{stem}.png"
    plot_heatmap(df, strategy, symbol, save_path=str(png_path))

    # Rating
    rating = rate_feasibility(df)
    print(f"\n实盘可行性评级: {rating['grade']}")
    print(f"  {rating['summary']}")
    print(rating["details"])

    return str(csv_path), str(png_path), rating


# ---------------------------------------------------------------------------
# Grid scanner
# ---------------------------------------------------------------------------

def run(
    strategy: str = "weekly_macd_kdj",
    symbol: str = "AAPL",
    start: str = "2020-01-01",
    end: Optional[str] = None,
    commission_grid: Optional[list] = None,
    slippage_grid: Optional[list] = None,
    initial_capital: float = 10000,
    sizing_mode: str = "fixed_capital",
    risk_per_trade: float = 0.005,
    risk_atr_mult: float = 2.0,
) -> pd.DataFrame:
    """Scan the commission × slippage grid and return a metrics DataFrame.

    Parameters
    ----------
    strategy : str
        Strategy name (key of STRATEGY_MAP).
    symbol : str
        Ticker symbol.
    start : str
        Start date (YYYY-MM-DD).
    end : str | None
        End date (default: today).
    commission_grid : list[float] | None
        Commission rates to scan (default: 1bp, 3bp, 10bp).
    slippage_grid : list[float] | None
        Slippage rates to scan (default: 1bp ~ 50bp).
    initial_capital : float
        Initial capital for each backtest run.

    Returns
    -------
    pd.DataFrame
        Columns: commission, slippage, return_pct, sharpe, max_dd_pct, trades
    """
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    if commission_grid is None:
        commission_grid = COMMISSION_GRID
    if slippage_grid is None:
        slippage_grid = SLIPPAGE_GRID

    strategy_cls = STRATEGY_MAP.get(strategy)
    if strategy_cls is None:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(STRATEGY_MAP)}")

    rows = []
    total = len(commission_grid) * len(slippage_grid)
    n = 0

    for comm, slip in itertools.product(commission_grid, slippage_grid):
        n += 1
        label = f"{strategy}/{symbol} [{n}/{total}]"
        print(f"\n{'='*50}")
        print(f"  {label}  comm={comm:.4f}  slip={slip:.4f}")
        print(f"{'='*50}")

        result, _ = run_backtest(
            symbol=symbol,
            start=start,
            end=end,
            initial_capital=initial_capital,
            strategy_cls=strategy_cls,
            commission_rate=comm,
            slippage_pct=slip,
            sizing_mode=sizing_mode,
            risk_per_trade=risk_per_trade,
            risk_atr_mult=risk_atr_mult,
        )

        rows.append({
            "commission": comm,
            "slippage": slip,
            "return_pct": round(result.total_return_pct, 2),
            "sharpe": round(result.sharpe_ratio, 2),
            "max_dd_pct": round(result.max_drawdown_pct, 2),
            "trades": result.total_trades,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cost sensitivity grid scan")
    parser.add_argument("--strategy", default="weekly_macd_kdj", help="Strategy name")
    parser.add_argument("--symbol", default="AAPL", help="Ticker symbol")
    parser.add_argument("--start", default="2020-01-01", help="Start date")
    parser.add_argument("--end", default=None, help="End date (default: today)")
    parser.add_argument("--csv", default=None, help="Output CSV path (overrides auto report)")
    parser.add_argument("--capital", type=float, default=10000, help="Initial capital")
    parser.add_argument("--sizing-mode", default="fixed_capital",
                        choices=["fixed_capital", "risk_budget"], help="Position sizing mode")
    parser.add_argument("--risk-per-trade", type=float, default=0.005,
                        help="Risk fraction per trade (risk_budget mode, default 0.5%%)")
    parser.add_argument("--risk-atr-mult", type=float, default=2.0,
                        help="ATR stop multiplier (risk_budget mode)")
    parser.add_argument("--no-report", action="store_true", help="Skip report generation")
    args = parser.parse_args()

    df = run(
        strategy=args.strategy,
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        initial_capital=args.capital,
        sizing_mode=args.sizing_mode,
        risk_per_trade=args.risk_per_trade,
        risk_atr_mult=args.risk_atr_mult,
    )

    print("\n" + "=" * 60)
    print("  成本敏感性结果")
    print("=" * 60)
    print(df.to_string(index=False))

    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\n结果已保存至: {path}")
    elif not args.no_report:
        generate_report(df, args.strategy, args.symbol)


if __name__ == "__main__":
    main()
