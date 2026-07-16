"""Golden-value drift diagnostic — ROADMAP.md 阶段八 CI-1.

Recomputes every strategy's metrics on the exact same fixed-seed dataset
tests/test_golden.py uses, and diffs against the values currently
hardcoded in that file. Prints a reviewable table — it does NOT rewrite
test_golden.py and is NOT a CI gate.

Why this exists
----------------
tests/test_golden.py's values are hand-typed floats. Updating them after
an intentional strategy-calculation change today means eyeballing pytest's
actual-vs-expected diff and re-typing numbers by feel — "盲改" (a blind
edit with no record of what changed or why). This script makes the "what
changed" part mechanical and reviewable; the "why" and the decision to
actually apply it stay human — same philosophy as scripts/strategy_fit_
audit.py.

Usage
-----
    pipenv run python scripts/check_golden_drift.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import utils  # noqa: F401
from utils.bootstrap import setup_runtime
setup_runtime()

from tests.conftest import make_ohlcv
from tests.test_golden import _GOLDEN_RISK_BUDGET
from engine.trader import BacktestEngine
from strategy import (
    EnhancedMACDStrategy, WeeklyMACD_KDJ, TurtleTrading,
    ATRBreakout, DonchianBreakout, DailyMACD_KDJ, WeeklyMACD,
    MACDKDJStrategy,
)

_GOLDEN_FILE = Path(__file__).parent.parent / "tests" / "test_golden.py"

# Mirrors the literal strategy instantiation in each
# test_golden_<name>_fixed_capital function — kept in sync manually,
# same as strategy_fit_audit.py's own strategy list.
_FIXED_CAPITAL_FACTORIES = {
    "test_golden_enhanced_macd_fixed_capital": EnhancedMACDStrategy,
    "test_golden_weekly_macd_kdj_fixed_capital": WeeklyMACD_KDJ,
    "test_golden_turtle_trading_fixed_capital": TurtleTrading,
    "test_golden_atr_breakout_fixed_capital": ATRBreakout,
    "test_golden_donchian_breakout_fixed_capital": DonchianBreakout,
    "test_golden_daily_macd_kdj_fixed_capital": DailyMACD_KDJ,
    "test_golden_weekly_macd_fixed_capital": WeeklyMACD,
    "test_golden_macd_kdj_merged_fixed_capital": lambda: MACDKDJStrategy(freq="D", use_atr_stop=True),
}

_RISK_BUDGET_FACTORIES = {
    "enhanced_macd": EnhancedMACDStrategy,
    "weekly_macd_kdj": WeeklyMACD_KDJ,
    "turtle_trading": TurtleTrading,
    "atr_breakout": ATRBreakout,
    "donchian_breakout": DonchianBreakout,
    "daily_macd_kdj": DailyMACD_KDJ,
    "weekly_macd": WeeklyMACD,
    "macd_kdj": lambda: MACDKDJStrategy(freq="D", use_atr_stop=True),
}

_METRIC_PATTERNS = {
    "trades": re.compile(r"total_trades\s*==\s*(\d+)"),
    "return": re.compile(r"total_return_pct\s*==\s*pytest\.approx\((-?\d+\.?\d*)"),
    "sharpe": re.compile(r"sharpe_ratio\s*==\s*pytest\.approx\((-?\d+\.?\d*)"),
    "maxdd": re.compile(r"max_drawdown_pct\s*==\s*pytest\.approx\((-?\d+\.?\d*)"),
}


def parse_fixed_capital_golden_values(source: str) -> dict:
    """{test_func_name: {trades?, return?, sharpe?, maxdd?}} — parsed via
    line-blocks, not a full Python parser, so it only sees what a diff
    would show: whichever metrics that specific test function actually
    asserts (some, like the merged-strategy test, only check trades+return)."""
    blocks = re.split(r"(?=^def test_golden_\w+_fixed_capital)", source, flags=re.M)
    result = {}
    for block in blocks:
        m = re.match(r"^def (test_golden_\w+_fixed_capital)", block)
        if not m:
            continue
        name = m.group(1)
        # Stop at the next top-level def so later functions' asserts don't leak in.
        body = block.split("\ndef ", 1)[0]
        values = {}
        for metric, pattern in _METRIC_PATTERNS.items():
            mm = pattern.search(body)
            if mm:
                values[metric] = float(mm.group(1))
        result[name] = values
    return result


def recompute(strategy_factory, sizing_mode: str = "fixed_capital") -> dict:
    df = make_ohlcv(n_bars=300, seed=42)
    s = strategy_factory()
    df = s.calculate_indicators(df)
    if sizing_mode == "risk_budget":
        engine = BacktestEngine(initial_capital=10000, sizing_mode="risk_budget",
                                 risk_per_trade=0.01, risk_atr_mult=2.0)
    else:
        engine = BacktestEngine(initial_capital=10000)
    engine.run(s, df)
    r = engine.get_result(df["Close"].pct_change(fill_method=None).dropna())
    return {
        "trades": r.total_trades,
        "return": round(r.total_return_pct, 4),
        "sharpe": round(r.sharpe_ratio, 4),
        "maxdd": round(r.max_drawdown_pct, 4),
    }


def _fmt(v) -> str:
    return "—" if v is None else (f"{v:g}" if isinstance(v, float) else str(v))


def main():
    source = _GOLDEN_FILE.read_text(encoding="utf-8")
    current_fixed = parse_fixed_capital_golden_values(source)

    print(f"\n{'=' * 88}")
    print("  Golden Drift Check — fixed_capital mode")
    print(f"{'=' * 88}")
    print(f"  {'test':<42} {'metric':<8} {'当前值':>10} {'重算值':>10} {'漂移':>6}")
    print(f"  {'-' * 84}")

    any_drift = False
    for test_name, factory in _FIXED_CAPITAL_FACTORIES.items():
        current = current_fixed.get(test_name, {})
        recomputed = recompute(factory)
        for metric in ("trades", "return", "sharpe", "maxdd"):
            if metric not in current:
                continue  # this test function doesn't assert this metric
            cur_v = current[metric]
            new_v = recomputed[metric]
            drift = abs(cur_v - new_v) > 0.01
            any_drift = any_drift or drift
            flag = "⚠ DRIFT" if drift else ""
            print(f"  {test_name:<42} {metric:<8} {_fmt(cur_v):>10} {_fmt(new_v):>10} {flag:>6}")

    print(f"\n{'=' * 88}")
    print("  Golden Drift Check — risk_budget mode")
    print(f"{'=' * 88}")
    print(f"  {'strategy':<20} {'metric':<8} {'当前值':>10} {'重算值':>10} {'漂移':>6}")
    print(f"  {'-' * 84}")

    for strat_name, factory in _RISK_BUDGET_FACTORIES.items():
        current = _GOLDEN_RISK_BUDGET[strat_name]
        recomputed = recompute(factory, sizing_mode="risk_budget")
        pairs = [
            ("trades", current["trades"], recomputed["trades"]),
            ("return", current["return"], recomputed["return"]),
            ("sharpe", current["sharpe"], recomputed["sharpe"]),
            ("maxdd", current["maxdd"], recomputed["maxdd"]),
        ]
        for metric, cur_v, new_v in pairs:
            drift = abs(cur_v - new_v) > 0.01
            any_drift = any_drift or drift
            flag = "⚠ DRIFT" if drift else ""
            print(f"  {strat_name:<20} {metric:<8} {_fmt(cur_v):>10} {_fmt(new_v):>10} {flag:>6}")

    print(f"\n{'=' * 88}")
    if any_drift:
        print("  结论: 检测到漂移 — 如果是预期内的策略逻辑改动, 手动更新")
        print("  tests/test_golden.py 里对应数值, 并在 commit message 里写清楚为什么。")
        print("  如果不是预期改动, 说明有代码改动意外影响了策略计算, 需要排查。")
    else:
        print("  结论: 无漂移 — tests/test_golden.py 的数值和当前代码一致。")
    print(f"{'=' * 88}\n")


if __name__ == "__main__":
    main()
