"""TrendFollower — MA + ADX filter + Chandelier trailing stop exit."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseStrategy, StrategyParams, ChandelierTrailingExit, compute_adx, register


@dataclass(frozen=True)
class TrendFollowerParams(StrategyParams):
    short_ma: int = 5
    long_ma: int = 20
    adx_period: int = 14
    adx_threshold: float = 15.0
    atr_period: int = 14
    trail_atr_mult: float = 2.5
    risk_per_trade: float = 0.02
    max_position_pct: float = 0.95

    grid = {
        "short_ma": [10, 20, 30],
        "long_ma": [40, 50, 60],
        "adx_threshold": [15, 20, 25],
        "trail_atr_mult": [2.0, 3.0, 4.0],
    }

    def validate(self):
        if not (self.short_ma < self.long_ma): raise ValueError("validation failed")
        if not (self.trail_atr_mult > 0): raise ValueError("validation failed")


@register("trend_follower")
class TrendFollower(ChandelierTrailingExit, BaseStrategy):
    """Breakout trend-follower with Chandelier trailing stop.

    Entry: MA uptrend + ADX confirms trend + +DI above -DI.
    Exit:  Chandelier trailing stop only.
    """

    regime = "trend"

    params: TrendFollowerParams

    def __init__(self, **kwargs):
        super().__init__(TrendFollowerParams(**kwargs))

    @property
    def min_bars(self) -> int:
        return max(self.params.long_ma, self.params.atr_period,
                   self.params.adx_period) + 5

    # ------------------------------------------------------------------

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        p = self.params

        df["SMA_short"] = df["Close"].rolling(p.short_ma).mean()
        df["SMA_long"] = df["Close"].rolling(p.long_ma).mean()

        # ADX (also sets ATR column)
        compute_adx(df, p.adx_period)

        # Entry signals only
        df["Signal"] = 0
        buy = (
            (df["SMA_short"] > df["SMA_long"])
            & (df["ADX"] > p.adx_threshold)
            & (df["+DI"] > df["-DI"])
        )
        df.loc[buy, "Signal"] = 1
        return df

    # ------------------------------------------------------------------

    def position_size(self, capital: float, price: float, atr: float) -> int:
        return self._risk_budget_size(capital, price, atr,
            self.params.risk_per_trade, self.params.trail_atr_mult,
            self.params.max_position_pct)

    # ------------------------------------------------------------------

    def check_exit(
        self,
        df: pd.DataFrame,
        i: int,
        entry_price: float,
        highest_since_entry: float,
        lowest_since_entry: Optional[float] = None,
        position: Optional[Dict] = None,
    ) -> Tuple[bool, str]:
        return self._chandelier_exit(df, i, highest_since_entry, lowest_since_entry, position)
