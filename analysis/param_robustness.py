"""Parameter robustness — neighborhood perturbation analysis.

Perturb optimal parameters by ±10% / ±20% one-at-a-time, then measure OOS
performance degradation.  Summarises the distribution (median, IQR, lower
quantile) and emits a ""robust"" / ""fragile"" / ""overfit"" conclusion.

Usage:
    pipenv run python analysis/param_robustness.py -s weekly_macd_kdj -symbol AAPL
    pipenv run python analysis/param_robustness.py -s enhanced_macd --start 2019-01-01
"""

import argparse
import itertools
import sys
from copy import deepcopy
from datetime import date, datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    from utils.bootstrap import setup_runtime
    setup_runtime()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from engine.trader import run_backtest
from engine.optimize import grid_search, PARAM_GRIDS
from strategy import STRATEGY_MAP

# ---------------------------------------------------------------------------
# Perturbation engine
# ---------------------------------------------------------------------------

PERTURB_LEVELS = [0.9, 1.1, 0.8, 1.2]  # -10%, +10%, -20%, +20%


def _is_int_param(param_name: str, param_grid: dict) -> bool:
    """Infer whether *param_name* expects an integer value."""
    if param_name in param_grid:
        values = param_grid[param_name]
        if values and all(isinstance(v, (int, np.integer)) for v in values):
            return True
    return False


def perturb_params(
    base: dict,
    param_grid: dict,
    levels: Optional[list] = None,
) -> list[dict]:
    """Generate one-at-a-time perturbations of *base* parameters.

    For every parameter in *base*, apply each multiplier in *levels* while
    holding all other params fixed.  Integer parameters are rounded and
    clamped to >= 1 — the final value MUST differ from the base value,
    otherwise the perturbation is skipped.

    Returns a list of param dicts (excluding the base itself).
    """
    if levels is None:
        levels = PERTURB_LEVELS

    variants = []
    seen = set()

    for key, base_val in base.items():
        if key not in param_grid:
            continue
        is_int = _is_int_param(key, param_grid)
        for mult in levels:
            if is_int:
                new_val = int(round(base_val * mult))
                if new_val == base_val:
                    new_val = base_val - 1 if mult < 1 else base_val + 1
                new_val = max(1, new_val)
            else:
                new_val = round(base_val * mult, 4)
                if abs(new_val - base_val) < 1e-8:
                    continue

            variant = deepcopy(base)
            variant[key] = new_val
            # Deduplicate via frozen param tuple
            sig = tuple(sorted(variant.items()))
            if sig not in seen:
                seen.add(sig)
                variants.append(variant)

    return variants


# ---------------------------------------------------------------------------
# OOS backtest runner
# ---------------------------------------------------------------------------

def _backtest_oos(
    strategy_name: str,
    symbol: str,
    params: dict,
    start: str,
    end: str,
    initial_capital: float = 10000,
    sizing_mode: str = "fixed_capital",
    risk_per_trade: float = 0.005,
    risk_atr_mult: float = 2.0,
) -> dict:
    """Run a single backtest on the OOS period and return key metrics."""
    strategy_cls = STRATEGY_MAP[strategy_name]
    result, _ = run_backtest(
        symbol=symbol,
        start=start,
        end=end,
        initial_capital=initial_capital,
        strategy_cls=strategy_cls,
        sizing_mode=sizing_mode,
        risk_per_trade=risk_per_trade,
        risk_atr_mult=risk_atr_mult,
        **params,
    )
    return {
        "return_pct": round(result.total_return_pct, 2),
        "sharpe": round(result.sharpe_ratio, 2),
        "max_dd_pct": round(result.max_drawdown_pct, 2),
        "trades": result.total_trades,
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run(
    strategy: str,
    symbol: str,
    start: str = "2018-01-01",
    end: Optional[str] = None,
    base_params: Optional[dict] = None,
    oos_split: float = 0.30,
    initial_capital: float = 10000,
    sizing_mode: str = "fixed_capital",
    risk_per_trade: float = 0.005,
    risk_atr_mult: float = 2.0,
) -> dict:
    """Run parameter robustness analysis.

    Parameters
    ----------
    strategy : str
        Strategy name.
    symbol : str
        Ticker symbol.
    start, end : str
        Full date range.  The last *oos_split* fraction is held out for OOS.
    base_params : dict | None
        Optimal parameters.  If None, they are found via grid_search on the
        IS (first 1 - oos_split) portion.
    oos_split : float
        Fraction of data reserved for OOS testing (default 0.30 = 30%).
    initial_capital : float
        Capital per backtest run.

    Returns
    -------
    dict with keys: base_params, oos_results_df, stats, conclusion, rating
    """
    if end is None:
        end = date.today().isoformat()

    full_start = pd.Timestamp(start)
    full_end = pd.Timestamp(end)
    total_days = (full_end - full_start).days
    split_date = full_start + pd.Timedelta(days=int(total_days * (1 - oos_split)))

    is_end_str = split_date.strftime("%Y-%m-%d")
    oos_start_str = (split_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    param_grid = PARAM_GRIDS.get(strategy, {})
    if not param_grid:
        raise ValueError(f"No param grid defined for {strategy}")

    # --- 1. Find optimal params if not provided ---
    if base_params is None:
        print(f"IS 优化: {start} ~ {is_end_str}")
        top = grid_search(
            strategy_name=strategy,
            symbol=symbol,
            start=start,
            end=is_end_str,
            initial_capital=initial_capital,
            metric="sharpe",
            top_n=1,
        )
        if not top:
            raise RuntimeError("IS 优化无有效结果")
        base_params = top[0].params
        print(f"最优参数: {base_params}")
        print(f"  IS Sharpe: {top[0].sharpe:.2f}  IS 收益: {top[0].total_return:+.1f}%")
    else:
        print(f"使用指定参数: {base_params}")

    # --- 2. Base OOS ---
    print(f"\nOOS 测试区间: {oos_start_str} ~ {end}")
    base_oos = _backtest_oos(strategy, symbol, base_params, oos_start_str, end,
                             initial_capital, sizing_mode, risk_per_trade, risk_atr_mult)
    print(f"最优参数 OOS: 收益 {base_oos['return_pct']:+.1f}%  "
          f"Sharpe {base_oos['sharpe']:.2f}  回撤 {base_oos['max_dd_pct']:.1f}%  "
          f"交易 {base_oos['trades']} 笔")

    # --- 3. Perturb & test ---
    variants = perturb_params(base_params, param_grid)
    print(f"\n邻域采样: {len(variants)} 个扰动点")

    rows = []
    for i, v in enumerate(variants):
        changed = {k: v[k] for k in v if v[k] != base_params.get(k)}
        metrics = _backtest_oos(strategy, symbol, v, oos_start_str, end,
                               initial_capital, sizing_mode, risk_per_trade, risk_atr_mult)
        label = ", ".join(f"{k}: {base_params[k]}→{v[k]}" for k in changed)
        rows.append({
            "label": label,
            **{f"p_{k}": v[k] for k in param_grid},
            **metrics,
        })

    df = pd.DataFrame(rows)

    # --- 4. Statistics ---
    stats = _compute_stats(df, base_oos)
    conclusion = _make_conclusion(stats, base_oos, base_params, df)

    return {
        "strategy": strategy,
        "symbol": symbol,
        "base_params": base_params,
        "base_oos": base_oos,
        "oos_results": df,
        "stats": stats,
        "conclusion": conclusion["text"],
        "rating": conclusion["rating"],
    }


def _compute_stats(df: pd.DataFrame, base_oos: dict) -> dict:
    """Compute distribution statistics for OOS metrics."""
    metrics = ["return_pct", "sharpe", "max_dd_pct", "trades"]
    stats = {}
    for col in metrics:
        series = df[col]
        base_val = base_oos[col]
        degrad = 0
        if abs(base_val) > 0.1:
            degrad = round((base_val - series.median()) / abs(base_val) * 100, 1)
        stats[col] = {
            "median": round(series.median(), 2),
            "q25": round(series.quantile(0.25), 2),
            "q75": round(series.quantile(0.75), 2),
            "q05": round(series.quantile(0.05), 2),
            "q95": round(series.quantile(0.95), 2),
            "min": round(series.min(), 2),
            "max": round(series.max(), 2),
            "base": base_val,
            "degradation_pct": degrad,
        }
    return stats


def _make_conclusion(stats: dict, base_oos: dict, base_params: dict, df: pd.DataFrame) -> dict:
    """Generate a human-readable robustness + viability conclusion."""
    ret_base = base_oos["return_pct"]
    ret_median = stats["return_pct"]["median"]
    ret_q05 = stats["return_pct"]["q05"]
    ret_degrad = stats["return_pct"]["degradation_pct"]

    sharpe_base = base_oos["sharpe"]
    sharpe_median = stats["sharpe"]["median"]
    trades_base = base_oos["trades"]

    # Count what fraction of perturbations are still profitable
    n_profitable = (df["return_pct"] > 0).sum()
    n_total = len(df)
    profit_ratio = n_profitable / n_total if n_total > 0 else 0

    # Find most sensitive parameter
    param_cols = [c for c in df.columns if c.startswith("p_")]
    sensitivities = {}
    for col in param_cols:
        name = col[2:]
        if name not in base_params:
            continue
        vals = df.groupby(col)["return_pct"].mean()
        if len(vals) >= 2:
            sensitivities[name] = round(vals.max() - vals.min(), 2)
    most_sensitive = max(sensitivities, key=sensitivities.get) if sensitivities else "N/A"

    # ---- Robustness rating ----
    if profit_ratio >= 0.9 and ret_degrad < 20 and ret_q05 > 0:
        robustness = "ROBUST"
        robustness_label = "鲁棒 — 参数邻域内表现稳定，90%+扰动仍盈利"
    elif profit_ratio >= 0.7 and ret_q05 > 0:
        robustness = "STABLE"
        robustness_label = "稳定 — 多数扰动保持盈利，下分位仍为正"
    elif profit_ratio >= 0.5:
        robustness = "SENSITIVE"
        robustness_label = "敏感 — 半数以上扰动盈利，但边缘组合出现亏损"
    else:
        robustness = "OVERFIT"
        robustness_label = "过拟合 — 多数扰动导致亏损，参数高度敏感"

    # ---- Strategy viability ----
    if ret_base <= 0:
        viability = "NEGATIVE"
        viability_label = "策略亏损，无论参数如何调整都无法盈利"
    elif trades_base < 3:
        viability = "NO_SIGNAL"
        viability_label = "交易次数不足3笔，统计无意义，信号过于稀疏"
    elif ret_base < 5:
        viability = "WEAK"
        viability_label = f"OOS收益仅{ret_base:+.1f}%，策略未捕获显著alpha，建议换策略或扩大参数搜索空间"
    elif trades_base < 5:
        viability = "MARGINAL"
        viability_label = f"OOS收益{ret_base:+.1f}%但仅{trades_base}笔交易，样本偏小，需更长回测期验证"
    elif sharpe_base < 0.5 and ret_base < 10:
        viability = "MARGINAL"
        viability_label = f"OOS收益{ret_base:+.1f}%但Sharpe仅{sharpe_base:.2f}，风险调整后吸引力不足"
    else:
        viability = "VIABLE"
        viability_label = f"OOS收益{ret_base:+.1f}%，Sharpe {sharpe_base:.2f}，具有实盘跟踪价值"

    # ---- Combined verdict ----
    if robustness == "OVERFIT" or viability in ("NEGATIVE", "NO_SIGNAL"):
        verdict = "✗ 不建议使用"
    elif viability == "WEAK" and robustness in ("ROBUST", "STABLE"):
        verdict = "△ 参数稳定但策略乏力，换策略优先于调参数"
    elif viability == "WEAK":
        verdict = "✗ 参数敏感且收益微弱，双重不利"
    elif viability == "MARGINAL":
        verdict = "△ 可观察，需更长回测期或更优策略替代"
    else:
        verdict = "✓ 可纳入实盘候选"

    # Sensitivity per parameter
    param_lines = []
    for name, spread in sorted(sensitivities.items(), key=lambda x: -x[1]):
        param_lines.append(f"  {name}: Δ{spread:+.1f}%")

    lines = [
        f"鲁棒性: {robustness}  |  策略质量: {viability}  |  结论: {verdict}",
        f"  {robustness_label}",
        f"  {viability_label}",
        "",
        f"最优参数: {base_params}",
        f"OOS 基准: 收益 {ret_base:+.1f}%  Sharpe {sharpe_base:.2f}  交易 {trades_base} 笔",
        "",
        f"邻域分布 ({n_total} 个扰动点):",
        f"  收益中位数: {ret_median:+.1f}%  (最优→中位衰减 {ret_degrad:+.1f}%)",
        f"  收益 IQR:    [{stats['return_pct']['q25']:+.1f}%, {stats['return_pct']['q75']:+.1f}%]",
        f"  收益 90%区间: [{stats['return_pct']['q05']:+.1f}%, {stats['return_pct']['q95']:+.1f}%]",
        f"  5%下分位:    {ret_q05:+.1f}%",
        f"  Sharpe中位数: {sharpe_median:.2f}",
        f"  盈利比例:     {profit_ratio:.0%} ({n_profitable}/{n_total})",
        "",
        "参数敏感度 (邻域收益极差):",
        *param_lines,
        "",
        f"最敏感参数: {most_sensitive}",
    ]

    return {"text": "\n".join(lines), "rating": robustness, "viability": viability, "verdict": verdict}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

from utils.font import setup_chinese_font


def plot(results: dict, save_path: Optional[str] = None):
    """Box-plot of OOS return distribution by perturbed parameter."""
    setup_chinese_font()

    df = results["oos_results"]
    param_cols = [c for c in df.columns if c.startswith("p_")]
    base_params = results["base_params"]
    base_return = results["base_oos"]["return_pct"]

    fig, axes = plt.subplots(1, len(param_cols), figsize=(5 * len(param_cols), 5),
                             squeeze=False)
    axes = axes[0]

    for ax, col in zip(axes, param_cols):
        name = col[2:]
        base_val = base_params.get(name)

        # Group by parameter value
        groups = df.groupby(col)["return_pct"].apply(list).to_dict()
        labels = [str(k) for k in sorted(groups.keys())]
        data = [groups[k] for k in sorted(groups.keys())]

        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)

        # Highlight base value
        if base_val is not None:
            base_label = str(base_val)
            if base_label in labels:
                idx = labels.index(base_label)
                bp["boxes"][idx].set_facecolor("#ffeb3b")
                bp["boxes"][idx].set_edgecolor("#f9a825")
                bp["boxes"][idx].set_linewidth(2)

        ax.axhline(y=base_return, color="green", linestyle="--", linewidth=1, alpha=0.7,
                   label=f"最优 OOS ({base_return:+.1f}%)")
        ax.axhline(y=0, color="red", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_title(f"{name}", fontsize=12, fontweight="bold")
        ax.set_ylabel("OOS 收益率 (%)")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"{results['strategy']} / {results['symbol']}  参数鲁棒性\n"
                 f"评级: {results['rating']}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"图表已保存: {save_path}")
    plt.close()
    return fig


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_report(results: dict, output_dir: Optional[str] = None):
    """Save CSV + PNG report under reports/."""
    out = Path(output_dir or PROJECT_ROOT / "reports")
    out.mkdir(parents=True, exist_ok=True)

    stem = f"param_robustness_{results['strategy']}_{results['symbol']}"

    # CSV
    csv_path = out / f"{stem}.csv"
    results["oos_results"].to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV 已保存: {csv_path}")

    # PNG
    png_path = out / f"{stem}.png"
    plot(results, save_path=str(png_path))

    # Print conclusion
    print(f"\n{results['conclusion']}")

    return str(csv_path), str(png_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="参数鲁棒性分析 — 邻域扰动")
    parser.add_argument("--strategy", "-s", required=True, help="策略名称")
    parser.add_argument("--symbol", default="AAPL", help="标的")
    parser.add_argument("--start", default="2018-01-01", help="数据起始日")
    parser.add_argument("--end", default=None, help="数据截止日")
    parser.add_argument("--oos-split", type=float, default=0.30, help="OOS 占比 (默认 0.30)")
    parser.add_argument("--capital", type=float, default=10000, help="初始资金")
    parser.add_argument("--sizing-mode", default="fixed_capital",
                        choices=["fixed_capital", "risk_budget"], help="仓位模式")
    parser.add_argument("--risk-per-trade", type=float, default=0.005,
                        help="单笔风险比例 (risk_budget 模式)")
    parser.add_argument("--risk-atr-mult", type=float, default=2.0,
                        help="ATR止损倍数 (risk_budget 模式)")
    parser.add_argument("--params", default=None, help="手动指定最优参数 (JSON: '{\"a\":1,\"b\":2}')")
    args = parser.parse_args()

    base_params = None
    if args.params:
        import json
        base_params = json.loads(args.params)

    results = run(
        strategy=args.strategy,
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        base_params=base_params,
        oos_split=args.oos_split,
        initial_capital=args.capital,
        sizing_mode=args.sizing_mode,
        risk_per_trade=args.risk_per_trade,
        risk_atr_mult=args.risk_atr_mult,
    )

    generate_report(results)


if __name__ == "__main__":
    main()
