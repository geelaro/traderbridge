"""Watchlist diversification audit.

ROADMAP.md 阶段八 W-1 的落地工具。回答: 当前 watchlist 的 N 个标的里,
真正独立的敞口有几个? 行业集中在哪? 引入跨资产分散标的能换来多少
VaR/HHI 改善?

Why this tool exists
---------------------
组合层面的分散化收益预期大于继续堆策略指标 (见 ROADMAP.md 阶段八)。
watchlist.toml 目前 13 个标的里 8 个是 utils/sectors.DEFAULT_SECTORS
标记的 "Technology"。这类分析需要重复做 (标的增删后复核), inline
脚本不可复现 —— 参照 scripts/strategy_fit_audit.py 的做法固化成工具。

Usage
-----
    pipenv run python scripts/portfolio_diversification_audit.py
    pipenv run python scripts/portfolio_diversification_audit.py --lookback-years 5
    pipenv run python scripts/portfolio_diversification_audit.py --candidates TLT,GLD,XLE,IWM

Output
------
1. Correlation / Effective Bets — watchlist 真实独立敞口数 + 最高相关对 + 聚类
2. Concentration — 行业 HHI + sector 暴露
3. What-If — 两档分散化方案 (保守 / 适度) 的 VaR/HHI/有效N 变化预演
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import utils  # noqa: F401
import pandas as pd

from data import DataProvider
from utils import load_toml
from utils.sectors import DEFAULT_SECTORS
from analysis.correlation_analysis import correlation_summary
from analysis.concentration import concentration_summary
from analysis.what_if import apply_rebalance, compare_portfolios


DEFAULT_CANDIDATES = ["TLT", "GLD", "XLE", "IWM"]  # 长债 / 黄金 / 能源 / 小盘


def fetch_prices(symbols: list[str], provider: DataProvider, years: int) -> pd.DataFrame:
    end = pd.Timestamp.today()
    start = (end - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    series = {}
    for sym in symbols:
        try:
            df = provider.get_daily(sym, start=start, end=end_str)
        except Exception as exc:
            print(f"  ! {sym} 拉取失败: {exc}")
            continue
        if df is None or df.empty or "Close" not in df.columns:
            print(f"  ! {sym} 无数据")
            continue
        series[sym] = df["Close"]
    if not series:
        return pd.DataFrame()
    return pd.concat(series, axis=1).sort_index()


def print_correlation_section(prices: pd.DataFrame, weights: dict) -> None:
    print(f"\n{'=' * 72}")
    print("  1. Correlation / Effective Bets")
    print(f"{'=' * 72}")

    summary = correlation_summary(prices, weights=weights, cluster_distance=0.3)
    eb = summary.get("effective_bets", {})
    if not eb:
        print("  数据不足 (需要 >= 30 个共同交易日), 跳过")
        return

    print(f"  持仓数 (n_symbols):        {eb['n_symbols']}")
    print(f"  有效独立敞口 (effective_n): {eb['effective_n']:.2f}")
    print(f"  集中度比 (effective/n):    {eb['concentration_ratio']:.2%}")
    print(f"  最大特征值占比:            {eb['top_eigenvalue_pct']:.1f}%")

    max_pair = summary.get("max_pair")
    if max_pair:
        a, b = max_pair["symbols"]
        print(f"\n  最高相关对: {a} <-> {b}   corr = {max_pair['correlation']:.3f}")

    clusters = summary.get("clusters", {})
    if clusters.get("clusters"):
        print(f"\n  相关性聚类 (distance_threshold=0.3, 即 |corr|>0.7 归一类):")
        for cid, syms in sorted(clusters["clusters"].items()):
            tag = "  <- 疑似重复下注" if len(syms) > 1 else ""
            print(f"    簇 {cid}: {', '.join(syms)}{tag}")


def print_concentration_section(weights: dict, corr_matrix: pd.DataFrame) -> dict:
    print(f"\n{'=' * 72}")
    print("  2. Concentration — 行业集中度")
    print(f"{'=' * 72}")

    summary = concentration_summary(
        weights, sector_map=DEFAULT_SECTORS, correlation_matrix=corr_matrix,
    )
    print(f"  持仓数:          {summary['n_holdings']}")
    print(f"  符号级 HHI:      {summary['hhi']:.0f}  ({summary['hhi_label']})")
    print(f"  有效N (等权换算): {summary['effective_n']:.2f}")
    if "correlation_hhi" in summary:
        print(f"  相关性调整HHI:   {summary['correlation_hhi']:.0f}  <- 比符号级HHI更真实")

    sh = summary.get("sector_hhi")
    if sh is not None:
        print(f"\n  行业 HHI:        {sh:.0f}  ({summary['sector_hhi_label']})")
        print(f"  行业暴露分布:")
        for sector, w in summary["sector_exposure"].items():
            bar = "#" * int(w * 40)
            print(f"    {sector:<20s} {w:>6.1%}  {bar}")
    return summary


def print_whatif_section(prices: pd.DataFrame, weights: dict, candidates: list[str]) -> None:
    print(f"\n{'=' * 72}")
    print("  3. What-If — 分散化方案预演")
    print(f"{'=' * 72}")

    tech_syms = [s for s in weights if DEFAULT_SECTORS.get(s.upper()) == "Technology"]
    semis = [s for s in ["NVDA", "MU", "INTC", "SMH"] if s in weights]
    available_candidates = [c for c in candidates if c in prices.columns]
    if not available_candidates:
        print(f"  分散化候选 {candidates} 均无价格数据, 跳过 What-If")
        return
    n_cand = len(available_candidates)

    scenarios = {
        "保守 (半仓半导体簇 -> 分散候选)": {
            **{s: -weights[s] * 0.5 for s in semis},
            **{c: sum(weights[s] * 0.5 for s in semis) / n_cand for c in available_candidates},
        },
        "适度 (全部Technology -20% -> 分散候选)": {
            **{s: -weights[s] * 0.2 for s in tech_syms},
            **{c: sum(weights[s] * 0.2 for s in tech_syms) / n_cand for c in available_candidates},
        },
    }

    for name, deltas in scenarios.items():
        after_weights = apply_rebalance(weights, deltas)
        cmp = compare_portfolios(
            prices, weights, after_weights, sector_map=DEFAULT_SECTORS,
        )
        print(f"\n  场景: {name}")
        print(f"    调整: {', '.join(f'{s} {d:+.1%}' for s, d in deltas.items() if abs(d) > 1e-6)}")
        print(f"    {cmp['summary_text']}")
        b, a = cmp["before"], cmp["after"]
        print(f"    VaR(95%,1d):   {b['var_pct']:.2f}% -> {a['var_pct']:.2f}%")
        print(f"    行业HHI:       {b.get('sector_hhi', 0):.0f} -> {a.get('sector_hhi', 0):.0f}")
        print(f"    有效N:         {b['effective_n']:.2f} -> {a['effective_n']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Watchlist diversification audit")
    parser.add_argument("--config", type=str, default="watchlist.toml")
    parser.add_argument("--lookback-years", type=int, default=3)
    parser.add_argument("--candidates", type=str, default=",".join(DEFAULT_CANDIDATES),
                        help="逗号分隔的分散化候选标的 (跨行业/跨资产 ETF)")
    args = parser.parse_args()

    import os
    os.chdir(Path(__file__).parent.parent)

    config = load_toml(args.config)
    symbols = [item["symbol"] for item in config.get("watchlist", [])]
    candidates = [c.strip().upper() for c in args.candidates.split(",") if c.strip()]

    print(f"\n  Watchlist 分散化审计 — {len(symbols)} 个标的: {', '.join(symbols)}")
    print(f"  回溯: {args.lookback_years} 年   分散化候选: {', '.join(candidates)}")

    provider = DataProvider()
    all_symbols = list(dict.fromkeys(symbols + candidates))  # de-dup, preserve order
    prices = fetch_prices(all_symbols, provider, args.lookback_years)

    if prices.empty:
        print("\n  无可用价格数据, 终止")
        return

    watchlist_prices = prices[[s for s in symbols if s in prices.columns]]
    missing = set(symbols) - set(watchlist_prices.columns)
    if missing:
        print(f"\n  ! 以下标的无价格数据, 已从分析中剔除: {', '.join(sorted(missing))}")

    weights = {s: 1.0 / len(watchlist_prices.columns) for s in watchlist_prices.columns}

    print_correlation_section(watchlist_prices, weights)
    corr_matrix = watchlist_prices.pct_change(fill_method=None).dropna().corr()
    print_concentration_section(weights, corr_matrix)
    print_whatif_section(prices, weights, candidates)

    print(f"\n{'=' * 72}\n")


if __name__ == "__main__":
    main()
