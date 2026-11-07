"""Daily scanner — fetch latest data, run strategies, output today's signals.

Usage:  pipenv run python daily.py          # scan today
        pipenv run python daily.py --date 2026-05-10  # scan specific day
        pipenv run python daily.py --history          # show recent signals
"""

import argparse
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

# Runtime bootstrap — MUST run before matplotlib / dotenv-aware imports.
from utils.bootstrap import setup_runtime
setup_runtime()

import pandas as pd

from data import DataProvider
from data.cache import CacheManager
from analysis.risk_monitor import compute_symbol_sentiment
from strategy import SIGNAL_LABEL
from utils import get_logger, load_toml
from utils.notify import Notifier
from utils.signal_scanner import SignalScanner

logger = get_logger("daily")


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str = "watchlist.toml") -> dict:
    return load_toml(path)


# ---------------------------------------------------------------------------
# Core scanner
# ---------------------------------------------------------------------------

def scan_day(
    config: dict,
    target_date: str = None,
    provider: DataProvider = None,
    cache: CacheManager = None,
) -> list[dict]:
    """Run all strategies across all watchlist symbols for *target_date*.

    Returns list of signal dicts.
    """
    if target_date is None:
        target_date = date.today().isoformat()

    if provider is None:
        provider = DataProvider()
    if cache is None:
        cache = CacheManager()

    lookback = config.get("scanner", {}).get("lookback_years", 3)
    scanner = SignalScanner(provider, cache=cache, lookback_years=lookback)
    results = scanner.scan(config, target_date=target_date)

    # --- data quality checks + display (presentation layer) ------------------
    scanned_symbols = set()
    results_by_symbol: dict[str, list[dict]] = {}
    for result in results:
        results_by_symbol.setdefault(result["symbol"], []).append(result)

    for r in results:
        sym = r["symbol"]
        if sym in scanned_symbols:
            continue
        scanned_symbols.add(sym)

        df = provider.get_daily(sym, start=(
            pd.Timestamp(target_date) - pd.DateOffset(years=lookback)
        ).strftime("%Y-%m-%d"), end=target_date)
        if df is None or df.empty:
            for signal in results_by_symbol[sym]:
                signal["symbol_sentiment"] = None
            continue

        sentiment = compute_symbol_sentiment(df)
        for signal in results_by_symbol[sym]:
            signal["symbol_sentiment"] = sentiment

        latest = df.index[-1]
        age_days = (pd.Timestamp(target_date) - latest).days
        if age_days > 5:
            logger.warning("数据陈旧: %s 最新K线 %s (%d天前)", sym, latest.date(), age_days)
            print(f"  ! {sym} 数据陈旧: 最新 {latest.date()} ({age_days}天前)")
        if len(df) < 50:
            logger.warning("数据不足: %s 仅 %d 根K线", sym, len(df))
            print(f"  ! {sym} K线不足: 仅 {len(df)} 根")
        if (df["Close"] <= 0).any():
            logger.warning("异常价格: %s 存在零/负收盘价", sym)
            print(f"  ! {sym} 异常价格: 存在零/负值")

    # --- display signals grouped by symbol ---
    symbol_groups: dict[str, list[dict]] = {}
    for r in results:
        symbol_groups.setdefault(r["symbol"], []).append(r)

    active_by_symbol = {}
    for item in config.get("watchlist", []):
        active_by_symbol[item["symbol"]] = item.get("active", "")

    for sym, group in symbol_groups.items():
        active_strat = active_by_symbol.get(sym, "")
        bar_date = group[0]["bar_date"] if group else ""
        name = group[0]["name"] if group else sym

        signal_lines = []
        for r in group:
            label = SIGNAL_LABEL.get(r["signal"], str(r["signal"]))
            if r["signal"] != 0:
                is_active = r["strategy"] == active_strat if isinstance(active_strat, str) else r["strategy"] == "ensemble"
                tag = " ★" if is_active else "  "
                signal_lines.append(
                    f"  {r['strategy']:<20s}  {label:<5s}  "
                    f"价格: {r['price']:.2f}  ATR: {r['atr']:.2f}{tag}"
                )

        if signal_lines:
            print(f"\n{'─' * 60}")
            print(f"  {sym}  {name}  ({bar_date})")
            print(f"{'─' * 60}")
            for line in signal_lines:
                print(line)

    return results


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]):
    """Print a compact summary — one row per symbol, buy/sell columns with wrapping."""

    grouped: dict[str, dict] = {}
    for r in results:
        sym = r["symbol"]
        if sym not in grouped:
            grouped[sym] = {"name": r["name"], "buy": [], "sell": [], "price": None}
        if r["signal"] == 1:
            grouped[sym]["buy"].append(r["strategy"])
            grouped[sym]["price"] = grouped[sym]["price"] or r["price"]
        elif r["signal"] == -1:
            grouped[sym]["sell"].append(r["strategy"])
            grouped[sym]["price"] = grouped[sym]["price"] or r["price"]

    active = {k: v for k, v in grouped.items() if v["buy"] or v["sell"]}
    if not active:
        print("\n  今日无买入/卖出信号\n")
        return

    buy_total = sum(1 for v in active.values() if v["buy"])
    sell_total = sum(1 for v in active.values() if v["sell"])
    COL_W = 22
    SYM_W = 8

    print(f"\n{'=' * 72}")
    print(f"  每日回溯结果")
    print(f"{'=' * 72}")
    print(f"  {'标的':<{SYM_W}} {'买入信号':<{COL_W}} {'卖出信号':<{COL_W}} {'参考价':>8}")
    print(f"  {'─' * (SYM_W + COL_W * 2 + 10)}")

    for sym in sorted(active.keys()):
        g = active[sym]
        buys = g["buy"] if g["buy"] else ["—"]
        sells = g["sell"] if g["sell"] else ["—"]
        price_str = f"${g['price']:.2f}" if g["price"] else "—"
        max_lines = max(len(buys), len(sells))

        for line_idx in range(max_lines):
            sym_col = sym if line_idx == 0 else ""
            buy_col = buys[line_idx] if line_idx < len(buys) else ""
            sell_col = sells[line_idx] if line_idx < len(sells) else ""
            price_col = price_str if line_idx == 0 else ""
            print(f"  {sym_col:<{SYM_W}} {buy_col:<{COL_W}} {sell_col:<{COL_W}} {price_col:>8}")

    print(f"  {'─' * (SYM_W + COL_W * 2 + 10)}")
    print(f"  买入: {buy_total} 个标的  卖出: {sell_total} 个标的  总计: {len(results)} 个策略-标的组合")
    print()


# ---------------------------------------------------------------------------
# History viewer
# ---------------------------------------------------------------------------

def show_history(cache: CacheManager = None, days: int = 7):
    """Display recent signal history from the cache."""
    if cache is None:
        cache = CacheManager()
    since = (date.today() - timedelta(days=days)).isoformat()
    rows = cache.query_signals(scan_date=since)
    if not rows:
        print(f"近 {days} 天无扫描记录")
        return

    print(f"\n{'=' * 80}")
    print(f"  近 {days} 天信号历史")
    print(f"{'=' * 80}")
    print(f"  {'日期':<12} {'标的':<8} {'策略':<22} {'信号':<6} {'价格':>8}")
    print(f"  {'─' * 56}")
    for r in rows:
        label = SIGNAL_LABEL.get(r["signal"], "?")
        print(f"  {r['scan_date']:<12} {r['symbol']:<8} {r['strategy']:<22} {label:<6} {r['price']:>8.2f}")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="traderbridge daily signal scanner")
    parser.add_argument("--date", type=str, help="Target date (YYYY-MM-DD), default today")
    parser.add_argument("--history", action="store_true", help="Show recent signal history")
    parser.add_argument("--config", type=str, default="watchlist.toml", help="Config file path")
    parser.add_argument("--notify", action="store_true", help="Send signal card via Feishu")
    parser.add_argument("--optimize", action="store_true", help="Walk-forward re-optimize and update params if OOS degraded")
    parser.add_argument("--opt-threshold", type=float, default=0.3,
                        help="Sharpe degradation threshold for auto-update (default 30%%)")
    args = parser.parse_args()

    # Ensure we run from the project root
    os.chdir(Path(__file__).parent)

    config = load_config(args.config)
    target_date = args.date or date.today().isoformat()

    if args.history:
        cache = CacheManager()
        show_history(cache, target_date)
        return

    provider = DataProvider()
    cache = CacheManager()
    results = scan_day(config, target_date=target_date, provider=provider, cache=cache)

    print_summary(results)

    if args.notify and results:
        notifier = Notifier()
        signals_with_signal = [r for r in results if r["signal"] != 0]
        if signals_with_signal:
            notifier.signal_card(signals_with_signal)

    # --- optional: walk-forward re-optimization -------------------------------
    if args.optimize:
        _run_optimize_and_update(config, args, target_date)


def _run_optimize_and_update(config: dict, args, target_date: str):
    """Run walk-forward optimization on active strategies and auto-update
    watchlist.toml if OOS Sharpe has degraded beyond *opt_threshold*."""
    from engine.optimize import walk_forward
    from utils import save_toml

    watchlist = config.get("watchlist", [])
    strategy_params = config.get("strategy", {})
    updated = False

    for item in watchlist:
        active = item.get("active", "")
        if not active or active == "enhanced_macd":
            continue
        if isinstance(active, list):  # ensemble — optimise members individually
            continue
        sym = item["symbol"]
        current_params = strategy_params.get(active, {})

        print(f"\n  滚动优化: {active} @ {sym}")
        try:
            result = walk_forward(active, sym, metric="sharpe")
        except Exception as e:
            logger.warning("优化失败 %s/%s: %s", active, sym, e)
            continue

        if not result.get("windows"):
            continue

        # Aggregate OOS performance
        oos_sharpe = result.get("sharpe", 0)
        # Best params from last window
        last_window = result["windows"][-1]
        new_params = last_window.get("best_params", {})

        if not new_params or new_params == current_params:
            continue

        # Compare: if last window's OOS sharpe < threshold relative to
        # historic average, the params have degraded.
        window_sharpes = [w.get("test_sharpe", 0) for w in result["windows"]]
        avg_sharpe = sum(window_sharpes) / len(window_sharpes) if window_sharpes else 0
        last_sharpe = window_sharpes[-1] if window_sharpes else 0

        if avg_sharpe > 0 and last_sharpe < avg_sharpe * (1 - args.opt_threshold):
            logger.info("参数退化: %s Sharpe %.2f → %.2f (最新), 自动更新", active, avg_sharpe, last_sharpe)
            print(f"  ! {active} 参数退化 (Sharpe {avg_sharpe:.2f} → {last_sharpe:.2f}), 更新为: {new_params}")
            strategy_params[active] = new_params
            updated = True
        else:
            print(f"  {active} OOS Sharpe {oos_sharpe:.2f}, 参数稳定, 无需更新")

    if updated:
        config["strategy"] = strategy_params
        save_toml(args.config, config)
        print(f"\n  已更新 {args.config}")
        logger.info("watchlist.toml 参数已自动更新")


if __name__ == "__main__":
    main()
