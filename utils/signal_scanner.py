"""Shared signal-scanning engine — used by daily.py and live_trader.py.

Extracted from the ~70% duplicated logic that both entry points previously
reimplemented: fetch bar data per symbol, run each strategy on it, and
collect the last-bar signal/price/ATR/indicators snapshot.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Dict, List, Optional

import pandas as pd

from data import DataProvider
from data.cache import CacheManager
from strategy import STRATEGY_MAP
from utils import get_logger

logger = get_logger("scanner")


class SignalScanner:
    """Run configured strategies across watchlist symbols for a target date.

    Parameters
    ----------
    provider : DataProvider
    cache : CacheManager, optional
        If provided, signals are persisted to the signal_history table.
    lookback_years : int
        How many years of bar data to fetch before scanning.
    """

    def __init__(
        self,
        provider: DataProvider,
        cache: Optional[CacheManager] = None,
        lookback_years: int = 3,
        weekly_anchor: str = "W-FRI",
    ):
        self.provider = provider
        self.cache = cache
        self.lookback_years = lookback_years
        self.weekly_anchor = weekly_anchor
        self._data_cache: Dict[str, pd.DataFrame] = {}  # symbol → daily df
        self._weekly_cache: Dict[str, pd.DataFrame] = {}  # symbol → weekly df
        self._proxy_cache: Optional[pd.DataFrame] = None  # shared proxy for ensemble

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(
        self,
        config: dict,
        target_date: Optional[str] = None,
        orphan_positions: Optional[List[dict]] = None,
        monitors: bool = True,
    ) -> List[dict]:
        """Run strategies on every watchlist (and optional orphan) symbol.

        Returns
        -------
        list[dict]
            Each dict: symbol, name, strategy, signal, price, atr,
            bar_date, indicators, orphan (bool).
        """
        if target_date is None:
            target_date = date.today().isoformat()

        start = (
            pd.Timestamp(target_date) - pd.DateOffset(years=self.lookback_years)
        ).strftime("%Y-%m-%d")

        results: List[dict] = []
        self._data_cache.clear()
        self._weekly_cache.clear()

        # --- watchlist symbols -------------------------------------------------
        for item in config.get("watchlist", []):
            results.extend(
                self._scan_symbol(item, start, target_date, active_first=monitors)
            )

        # --- orphan positions --------------------------------------------------
        if orphan_positions:
            for op in orphan_positions:
                if op["symbol"] in {r["symbol"] for r in results}:
                    continue  # already scanned via watchlist
                item = {
                    "symbol": op["symbol"],
                    "name": op.get("name", op["symbol"]),
                    "active": op["strategy"],
                    "_orphan": True,
                }
                results.extend(
                    self._scan_symbol(item, start, target_date, active_first=False)
                )

        self._data_cache.clear()
        self._weekly_cache.clear()
        self._proxy_cache = None
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _scan_symbol(
        self,
        item: dict,
        start: str,
        target_date: str,
        active_first: bool = True,
    ) -> List[dict]:
        """Scan one symbol across its active + monitor strategies."""
        symbol = item["symbol"]
        name = item.get("name", symbol)
        is_orphan = item.get("_orphan", False)
        active_strat = item.get("active", "")
        monitor_list = item.get("monitor", []) if active_first else []

        # ── ensemble mode: active is a list of strategy names ─────────
        if isinstance(active_strat, list) and active_strat:
            return self._scan_ensemble(
                symbol, name, active_strat, item, start, target_date, is_orphan,
            )

        strategy_names = [active_strat] + monitor_list if active_strat else monitor_list
        if not strategy_names:
            return []

        df = self._fetch_data(symbol, start, target_date)
        if df is None or df.empty:
            return []

        df_weekly = self._fetch_weekly(symbol, start, target_date)

        if target_date not in df.index.strftime("%Y-%m-%d"):
            logger.warning("[%s] %s 无 K 线数据, 使用最新 bar %s",
                           symbol, target_date, df.index[-1].strftime("%Y-%m-%d"))
            bar_date = df.index[-1].strftime("%Y-%m-%d")
        else:
            bar_date = target_date

        results: List[dict] = []
        for strat_name in strategy_names:
            if strat_name not in STRATEGY_MAP:
                # Common pitfall: user wrote `active = "ensemble"` instead of
                # `active = ["turtle_trading", "rsi2_mean_reversion"]`.
                # STRATEGY_MAP has no ensemble entry by design, so this would
                # silently produce zero signals. Warn loudly.
                if "ensemble" in str(strat_name).lower():
                    logger.warning(
                        "[%s] active=%r is not in STRATEGY_MAP — to use the "
                        "ensemble strategy set `active = [\"member_a\", \"member_b\"]` "
                        "(list of member strategy names) in watchlist.toml",
                        symbol, strat_name,
                    )
                continue
            strat_params = item.get("params", {})
            strategy = STRATEGY_MAP[strat_name](**strat_params)
            try:
                df_sig = strategy.calculate_indicators(df, df_weekly=df_weekly)
            except Exception:
                logger.exception("策略计算失败: %s %s", symbol, strat_name)
                continue

            last_idx = -1
            signal = int(df_sig["Signal"].iloc[last_idx])
            price = float(df_sig["Close"].iloc[last_idx])
            atr = float(df_sig["ATR"].iloc[last_idx]) if "ATR" in df_sig.columns else 0

            indicators: Dict[str, float] = {}
            for col in df_sig.columns:
                if col not in ("Open", "High", "Low", "Close", "Volume", "Signal"):
                    val = df_sig[col].iloc[last_idx]
                    if isinstance(val, (float, int)) and not pd.isna(val):
                        indicators[col] = round(float(val), 4)

            # Persist to cache if available
            if self.cache is not None:
                self.cache.save_signal(
                    scan_date=target_date,
                    symbol=symbol,
                    strategy=strat_name,
                    bar_date=bar_date,
                    signal=signal,
                    price=price,
                    atr=atr,
                    indicators=json.dumps(indicators, ensure_ascii=False),
                )

            results.append({
                "symbol": symbol,
                "name": name,
                "strategy": strat_name,
                "signal": signal,
                "price": price,
                "atr": atr,
                "bar_date": bar_date,
                "indicators": indicators,
                "orphan": is_orphan,
            })

        return results

    def _scan_ensemble(
        self, symbol: str, name: str, members: list, item: dict,
        start: str, target_date: str, is_orphan: bool,
    ) -> List[dict]:
        """Build a StrategyEnsemble from *members* and return one signal dict."""
        df = self._fetch_data(symbol, start, target_date)
        if df is None or df.empty:
            return []

        # Lazy-fetch proxy data once per scan pass
        if self._proxy_cache is None:
            self._proxy_cache = self.provider.get_daily("SPY", start=start, end=target_date)

        from strategy.ensemble import StrategyEnsemble
        from strategy import STRATEGY_MAP as _SMAP

        tags = item.get("regime_tags") or ["trend"] * len(members)
        if len(tags) != len(members):
            logger.warning("regime_tags 长度与 members 不匹配: %s", symbol)
            return []

        member_params = item.get("params", {}).get("members", {})
        pairs = []
        for strat_name, tag in zip(members, tags):
            cls = _SMAP.get(strat_name)
            if cls is None:
                logger.warning("未知策略: %s", strat_name)
                return []
            mp = member_params.get(strat_name, {})
            pairs.append((cls(**mp), tag))

        ep = item.get("ensemble_params") or {}
        ensemble = StrategyEnsemble(
            members=pairs, proxy_df=self._proxy_cache, **ep,
        )
        df_sig = ensemble.calculate_indicators(df)

        idx = -1
        signal = int(df_sig["Signal"].iloc[idx])
        price = float(df_sig["Close"].iloc[idx])
        atr = float(df_sig["ATR"].iloc[idx]) if "ATR" in df_sig.columns else 0

        bar_date = (
            target_date if target_date in df_sig.index.strftime("%Y-%m-%d")
            else df_sig.index[-1].strftime("%Y-%m-%d")
        )

        indicators: Dict[str, float] = {}
        for col in df_sig.columns:
            if col not in ("Open", "High", "Low", "Close", "Volume", "Signal"):
                val = df_sig[col].iloc[idx]
                if isinstance(val, (float, int)) and not pd.isna(val):
                    indicators[col] = round(float(val), 4)

        if self.cache is not None:
            self.cache.save_signal(
                scan_date=target_date, symbol=symbol, strategy="ensemble",
                bar_date=bar_date, signal=signal, price=price, atr=atr,
                indicators=json.dumps(indicators, ensure_ascii=False),
            )

        return [{
            "symbol": symbol, "name": name, "strategy": "ensemble",
            "signal": signal, "price": price, "atr": atr,
            "bar_date": bar_date, "indicators": indicators, "orphan": is_orphan,
        }]

    def _fetch_data(
        self, symbol: str, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        """Fetch daily OHLCV for *symbol*, caching per-symbol within a scan pass."""
        if symbol not in self._data_cache:
            self._data_cache[symbol] = self.provider.get_daily(symbol, start=start, end=end)
        df = self._data_cache[symbol]
        if df is None or df.empty:
            return None
        return df

    def _fetch_weekly(
        self, symbol: str, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        """Fetch weekly OHLCV for *symbol* (resampled from daily, cached in-scan).

        Uses ``self.weekly_anchor`` for the resample weekday (default ``W-FRI``
        for US markets; pass ``W-WED`` or ``W-THU`` for Asian markets).
        """
        if symbol not in self._weekly_cache:
            daily = self._fetch_data(symbol, start, end)
            if daily is not None and not daily.empty:
                self._weekly_cache[symbol] = (
                    daily.resample(self.weekly_anchor)
                    .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
                    .dropna()
                )
            else:
                self._weekly_cache[symbol] = pd.DataFrame()
        df = self._weekly_cache[symbol]
        if df is None or df.empty:
            return None
        return df


# ---------------------------------------------------------------------------
# Helper: merge strategy params from watchlist.toml into scan item
# ---------------------------------------------------------------------------

def enrich_scan_items(config: dict) -> List[dict]:
    """Return watchlist items with strategy params baked in."""
    items: List[dict] = []
    for item in config.get("watchlist", []):
        active = item.get("active", "")
        merged = dict(item)
        if isinstance(active, list):
            # Ensemble mode — collect per-member params
            merged["params"] = {"members": {}}
            for name in active:
                merged["params"]["members"][name] = config.get("strategy", {}).get(name, {})
        else:
            merged["params"] = config.get("strategy", {}).get(active, {})
        items.append(merged)
    return items
