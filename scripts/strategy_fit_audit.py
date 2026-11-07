"""Strategy-symbol fit audit.

For each symbol, run each candidate strategy and compare against Buy & Hold.
Output a recommendation per symbol (which strategy gives the best risk-adjusted
or absolute return), and flag symbols where the current watchlist.toml active
strategy disagrees with the audit recommendation.

Why this tool exists
--------------------
- watchlist.toml started with `active = "weekly_macd_kdj"` for all 12 symbols
- 2026-06-07 audit found weekly_macd_kdj is double-loss on MU/INTC
- 2026-06-09 split data fix (IVW/SOXX/sector ETFs) invalidated earlier audit;
  re-run found weekly_macd Sharpe-dominates on 4/12 long-history symbols
- Future split fixes, new symbols, parameter changes all need re-audit. Inline
  ad-hoc scripts are not reproducible. This tool fixes that.

Usage
-----
    pipenv run python scripts/strategy_fit_audit.py
    pipenv run python scripts/strategy_fit_audit.py --symbols MU,INTC,ORCL
    pipenv run python scripts/strategy_fit_audit.py --strategies all
    pipenv run python scripts/strategy_fit_audit.py --sort cagr
    pipenv run python scripts/strategy_fit_audit.py --lookback-years 10

Output
------
- Per-symbol table: each candidate strategy's CAGR / Sharpe / MaxDD + delta vs B&H
- Summary table: recommended active strategy per symbol + comparison with
  watchlist.toml current setting
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.bootstrap import setup_runtime
setup_runtime()
import numpy as np
import pandas as pd
import toml

from data import DataProvider
from engine.trader import BacktestEngine
from strategy import STRATEGY_MAP


DEFAULT_CANDIDATES = ["weekly_macd", "weekly_macd_kdj"]


@dataclass
class StrategyAudit:
    symbol: str
    strategy: str
    cagr: float
    sharpe: float
    max_dd: float
    total_trades: int
    bh_cagr: float
    bh_sharpe: float
    bh_max_dd: float

    @property
    def sharpe_delta(self) -> float:
        return self.sharpe - self.bh_sharpe

    @property
    def cagr_delta(self) -> float:
        return self.cagr - self.bh_cagr

    @property
    def dd_delta(self) -> float:
        """Positive = strategy DD shallower than B&H (improvement)."""
        return self.max_dd - self.bh_max_dd


def _bh_metrics(close: pd.Series) -> tuple[float, float, float]:
    rets = close.pct_change().dropna()
    yrs = (close.index[-1] - close.index[0]).days / 365.25
    cagr = ((close.iloc[-1] / close.iloc[0]) ** (1 / yrs) - 1) * 100 if yrs > 0 else 0.0
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    eq = (1 + rets).cumprod()
    dd = float((eq / eq.cummax() - 1).min()) * 100
    return cagr, sharpe, dd


def run_strategy(strategy_name: str, df: pd.DataFrame):
    """Backtest *strategy_name* on *df*. Return BacktestResult or None on failure."""
    StratCls = STRATEGY_MAP.get(strategy_name)
    if StratCls is None:
        return None
    try:
        st = StratCls()
    except Exception:
        return None
    df_weekly = df.resample("W-FRI").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna()
    try:
        ds = st.calculate_indicators(df.copy(), df_weekly=df_weekly).copy()
    except TypeError:
        ds = st.calculate_indicators(df.copy()).copy()
    except Exception:
        return None
    try:
        eng = BacktestEngine(initial_capital=100000)
        bench = eng.run(st, ds)
        return eng.get_result(bench)
    except Exception:
        return None


def audit_symbol(
    symbol: str,
    provider: DataProvider,
    strategies: Iterable[str],
    lookback_years: Optional[float] = None,
) -> list[StrategyAudit]:
    """Run all *strategies* against *symbol*. Returns list of audits."""
    end = pd.Timestamp.now().normalize().strftime("%Y-%m-%d")
    if lookback_years and lookback_years < 99:
        start = (pd.Timestamp(end) - pd.DateOffset(years=int(lookback_years))).strftime("%Y-%m-%d")
    else:
        start = "2006-01-01"
    df = provider.get_daily(symbol, start=start, end=end)
    if df is None or df.empty or len(df) < 60:
        return []
    bh_cagr, bh_sharpe, bh_dd = _bh_metrics(df["Close"])
    audits = []
    for sname in strategies:
        r = run_strategy(sname, df)
        if r is None:
            continue
        audits.append(StrategyAudit(
            symbol=symbol, strategy=sname,
            cagr=r.cagr_pct, sharpe=r.sharpe_ratio, max_dd=r.max_drawdown_pct,
            total_trades=r.total_trades,
            bh_cagr=bh_cagr, bh_sharpe=bh_sharpe, bh_max_dd=bh_dd,
        ))
    return audits


def best_audit(audits: list[StrategyAudit], sort_by: str = "sharpe") -> Optional[StrategyAudit]:
    """Pick best strategy from *audits* by *sort_by* ('sharpe' or 'cagr')."""
    if not audits:
        return None
    key = (lambda a: a.sharpe) if sort_by == "sharpe" else (lambda a: a.cagr)
    return max(audits, key=key)


def load_watchlist_active(config_path: str = "watchlist.toml") -> dict[str, str]:
    """Return {symbol: active_strategy} from watchlist.toml. Empty if missing."""
    path = Path(config_path)
    if not path.is_file():
        return {}
    cfg = toml.load(path)
    result = {}
    for w in cfg.get("watchlist", []):
        sym = w.get("symbol", "").upper()
        active = w.get("active", "")
        if isinstance(active, str) and active:
            result[sym] = active
    return result


def format_per_symbol(audits: list[StrategyAudit], current_active: Optional[str]) -> str:
    """Format the per-symbol detail table."""
    if not audits:
        return "  (no audits — insufficient data?)"
    sym = audits[0].symbol
    yrs = "?"
    # Header
    lines = []
    bh = audits[0]
    lines.append(f"== {sym} (B&H CAGR {bh.bh_cagr:.2f}% / Sharpe {bh.bh_sharpe:.2f} / DD {bh.bh_max_dd:.1f}%) ==")
    lines.append(f"{'strategy':<22s} {'CAGR':>7s} {'Sharpe':>7s} {'MaxDD':>8s} {'CAGRΔ':>7s} {'SharpeΔ':>8s} {'DDΔ':>7s} {'trades':>6s}")
    lines.append("-" * 80)
    for a in audits:
        marker = "  <- active" if current_active == a.strategy else ""
        lines.append(
            f"{a.strategy:<22s} {a.cagr:>6.2f}% {a.sharpe:>7.2f} {a.max_dd:>7.1f}% "
            f"{a.cagr_delta:>+6.2f} {a.sharpe_delta:>+7.2f} {a.dd_delta:>+6.1f} "
            f"{a.total_trades:>6d}{marker}"
        )
    return "\n".join(lines)


def format_summary(
    by_symbol: dict[str, list[StrategyAudit]],
    current: dict[str, str],
    sort_by: str,
) -> str:
    """Recommendation table — current vs audit-recommended active."""
    lines = []
    lines.append(f"\n== 推荐汇总 (按 {sort_by} 选最优) ==")
    lines.append(f"{'symbol':<6s} {'current':<22s} {'recommended':<22s} {'flag':<10s} {'notes':<40s}")
    lines.append("-" * 110)
    n_changed = 0
    for sym, audits in by_symbol.items():
        best = best_audit(audits, sort_by)
        if best is None:
            lines.append(f"{sym:<6s}  (no data)")
            continue
        cur = current.get(sym, "?")
        if cur != best.strategy:
            n_changed += 1
            flag = "[CHANGE]"
        else:
            flag = "[same]"
        notes = f"Sharpe {best.sharpe:.2f} (BH {best.bh_sharpe:.2f}, Δ{best.sharpe_delta:+.2f})"
        lines.append(f"{sym:<6s} {cur:<22s} {best.strategy:<22s} {flag:<10s} {notes:<40s}")
    lines.append("-" * 110)
    lines.append(f"建议改动: {n_changed} 个标的的 active 跟审计推荐不一致")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", type=str, default="",
                   help="逗号分隔标的列表 (默认从 watchlist.toml 读)")
    p.add_argument("--strategies", type=str, default=",".join(DEFAULT_CANDIDATES),
                   help=f"逗号分隔策略列表, 'all' 跑全部 (默认 {','.join(DEFAULT_CANDIDATES)})")
    p.add_argument("--lookback-years", type=float, default=99,
                   help="回测年数, 99 = 全历史 (默认 99)")
    p.add_argument("--sort", choices=["sharpe", "cagr"], default="sharpe",
                   help="推荐策略排序依据 (默认 sharpe)")
    p.add_argument("--config", default="watchlist.toml",
                   help="watchlist 配置路径 (默认 watchlist.toml)")
    args = p.parse_args()

    # Resolve symbols
    current = load_watchlist_active(args.config)
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = list(current.keys())
    if not symbols:
        print("没有标的 — 指定 --symbols 或确保 watchlist.toml 存在")
        return 1

    # Resolve strategies
    if args.strategies == "all":
        strategies = sorted(STRATEGY_MAP.keys())
    else:
        strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    if not strategies:
        print("没有策略 — 检查 --strategies 参数")
        return 1

    provider = DataProvider()
    by_symbol: dict[str, list[StrategyAudit]] = {}
    for sym in symbols:
        audits = audit_symbol(sym, provider, strategies, lookback_years=args.lookback_years)
        by_symbol[sym] = audits
        print()
        print(format_per_symbol(audits, current.get(sym)))

    print(format_summary(by_symbol, current, args.sort))
    return 0


if __name__ == "__main__":
    sys.exit(main())
