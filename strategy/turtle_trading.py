"""Turtle Trading — adapted from "大哥2.2" strategy.

Long:  recursive SMA short > SMA long AND close >= Donchian upper (prev bar)
Short: recursive SMA short < SMA long AND close <= Donchian lower (prev bar)
Exit:  Chandelier trailing stop with entry-time ATR (fixed), tracked on High/Low.

Key differences from vanilla Turtle:
- SMA uses recursive seeding (seed = first close), matching 大哥2.2 exactly.
- Donchian channel built on **Close** (not High/Low).
- Entry triggered against **previous** bar's channel value.
- Stop-loss ATR is **frozen at entry** (not updated daily).
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseStrategy, StrategyParams, register


# -- helpers ---------------------------------------------------------------


def _recursive_sma(series: pd.Series, period: int) -> pd.Series:
    """Recursive SMA seeded from the first valid close (大哥2.2 formula).

    ``sma[t] = (sma[t-1] * (n-1) + close[t]) / n``
    """
    sma = pd.Series(np.nan, index=series.index, dtype=float)
    first = series.first_valid_index()
    if first is None:
        return sma
    sma.loc[first] = series.loc[first]
    prev = series.loc[first]
    for idx in range(series.index.get_loc(first) + 1, len(series)):
        prev = (prev * (period - 1) + series.iloc[idx]) / period
        sma.iloc[idx] = prev
    return sma


# -- params -----------------------------------------------------------------


@dataclass(frozen=True)
class TurtleTradingParams(StrategyParams):
    short_period: int = 20
    long_period: int = 50
    channel_period: int = 20
    atr_period: int = 14
    trail_atr_mult: float = 3.0
    risk_per_trade: float = 0.02
    max_position_pct: float = 0.95
    trend_filter: bool = True
    trend_filter_bars: int = 60

    grid = {
        "short_period": [10, 20, 30],
        "long_period": [40, 50, 60],
        "channel_period": [15, 20, 30, 40],
        "trail_atr_mult": [2.0, 3.0, 4.0],
    }

    def validate(self):
        if not (self.short_period < self.long_period):
            raise ValueError("short_period must be < long_period")
        if not (self.channel_period >= 10):
            raise ValueError("channel_period must be >= 10")
        if not (self.trail_atr_mult > 0):
            raise ValueError("trail_atr_mult must be > 0")
        if not (0 < self.risk_per_trade <= 1):
            raise ValueError("validation failed")


# -- strategy --------------------------------------------------------------


@register("turtle_trading")
class TurtleTrading(BaseStrategy):
    """大哥2.2-style channel breakout with recursive SMA trend filter.

    Long:  close >= upper_channel[-1]  AND  SMA_short > SMA_long
    Short: close <= lower_channel[-1]  AND  SMA_short < SMA_long
    Exit:  Chandelier stop (entry-time ATR, High/Low tracking).
    """

    regime = "trend"
    long_only = False

    params: TurtleTradingParams

    def __init__(self, **kwargs):
        super().__init__(TurtleTradingParams(**kwargs))
        # Per-entry cache for incremental high/low tracking and frozen ATR.
        # Reset whenever check_exit sees a new entry_date.
        self._cur_entry_date = None
        self._cur_entry_atr = 0.0
        self._cur_highest_high = -float('inf')
        self._cur_lowest_low = float('inf')

    @property
    def min_bars(self) -> int:
        p = self.params
        return max(p.long_period, p.channel_period, p.atr_period) + 5

    # -- indicators ---------------------------------------------------------

    def calculate_indicators(
        self, df: pd.DataFrame, df_weekly: pd.DataFrame = None,
    ) -> pd.DataFrame:
        df = df.copy()
        p = self.params

        # Recursive SMA (大哥2.2 formula)
        df["SMA_short"] = _recursive_sma(df["Close"], p.short_period)
        df["SMA_long"] = _recursive_sma(df["Close"], p.long_period)

        # Donchian Channel on Close (大哥2.2: high_line / low_line)
        df["Donchian_upper"] = df["Close"].rolling(p.channel_period).max()
        df["Donchian_lower"] = df["Close"].rolling(p.channel_period).min()

        # ATR (Welles Wilder)
        from .base import compute_atr
        df["ATR"] = compute_atr(df, p.atr_period)

        # Trend filter — suppress shorts when long-term SMA is rising
        trend_up = pd.Series(False, index=df.index)
        if p.trend_filter:
            trend_up = df["SMA_long"] > df["SMA_long"].shift(p.trend_filter_bars)
            trend_dn = df["SMA_long"] < df["SMA_long"].shift(p.trend_filter_bars)

        # Signals — entry vs previous bar's channel (大哥2.2: high_line[-1])
        df["Signal"] = 0
        buy = (
            (df["SMA_short"] > df["SMA_long"])
            & (df["Close"] >= df["Donchian_upper"].shift(1))
        )
        short = (
            (df["SMA_short"] < df["SMA_long"])
            & (df["Close"] <= df["Donchian_lower"].shift(1))
        )
        if p.trend_filter:
            short = short & ~trend_up  # suppress short in uptrend
            buy = buy & ~trend_dn     # suppress long in downtrend
        df.loc[buy, "Signal"] = 1
        df.loc[short, "Signal"] = -1

        return df

    # -- sizing -------------------------------------------------------------

    def position_size(self, capital: float, price: float, atr: float) -> int:
        return self._risk_budget_size(capital, price, atr,
            self.params.risk_per_trade, self.params.trail_atr_mult,
            self.params.max_position_pct)

    # -- exit (大哥2.2 trailing stop) ---------------------------------------

    def check_exit(
        self,
        df: pd.DataFrame,
        i: int,
        entry_price: float,
        highest_since_entry: float,
        lowest_since_entry: Optional[float] = None,
        position: Optional[Dict] = None,
    ) -> Tuple[bool, str]:
        """Chandelier stop with entry-time ATR, tracked on High/Low (大哥2.2).

        Long stop:  close <= max(high[entry:i]) - ATR_entry × trail_atr_mult
        Short stop: close >= min(low[entry:i]) + ATR_entry × trail_atr_mult

        Uses an O(1) per-bar incremental cache (``_cur_*``) rather than
        re-slicing ``df`` on every call. Cache is reset when *entry_date*
        changes, so the backtest engine's per-bar call cadence is required
        for correctness — single-shot test calls still produce the correct
        result for the current bar.
        """
        price = float(df["Close"].iloc[i])
        entry_date = position.get('date') if position else None

        # Detect new entry → reset cache and freeze ATR at entry (大哥2.2: ATR[pos-1])
        if entry_date != self._cur_entry_date:
            self._cur_entry_date = entry_date
            if entry_date is not None and entry_date in df.index:
                pos_idx = df.index.get_loc(entry_date)
                self._cur_entry_atr = float(df["ATR"].iloc[max(0, pos_idx - 1)])
                self._cur_highest_high = float(df["High"].iloc[pos_idx])
                self._cur_lowest_low = float(df["Low"].iloc[pos_idx])
            else:
                self._cur_entry_atr = float(df["ATR"].iloc[i])
                self._cur_highest_high = float(df["High"].iloc[i])
                self._cur_lowest_low = float(df["Low"].iloc[i])

        # Incremental update from the latest bar
        cur_high = float(df["High"].iloc[i])
        cur_low = float(df["Low"].iloc[i])
        if cur_high > self._cur_highest_high:
            self._cur_highest_high = cur_high
        if cur_low < self._cur_lowest_low:
            self._cur_lowest_low = cur_low

        direction = position.get('direction', 'LONG') if position else 'LONG'
        if direction == 'SHORT':
            cover_level = self._cur_lowest_low + self._cur_entry_atr * self.params.trail_atr_mult
            if price >= cover_level:
                return True, "移动止损(空)"
        else:
            stop_level = self._cur_highest_high - self._cur_entry_atr * self.params.trail_atr_mult
            if price <= stop_level:
                return True, "移动止损"

        return False, ""
