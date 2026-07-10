"""ATR Breakout — volatility-normalized breakout strategy.

Entry: close crosses above MA + N*ATR.
Exit:  Chandelier trailing stop.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseStrategy, StrategyParams, ChandelierTrailingExit, compute_atr, register


@dataclass(frozen=True)
class ATRBreakoutParams(StrategyParams):
    ma_period: int = 20
    atr_period: int = 14
    breakout_atr_mult: float = 2.0
    trail_atr_mult: float = 3.0
    risk_per_trade: float = 0.02
    max_position_pct: float = 0.95
    ma_type: str = "ema"

    grid = {
        "ma_period": [15, 20, 30],
        "breakout_atr_mult": [1.5, 2.0, 2.5, 3.0],
        "trail_atr_mult": [2.5, 3.0, 4.0, 5.0],
    }

    def validate(self):
        if not (self.ma_period > 0):
            raise ValueError("ma_period must be > 0")
        if not (self.breakout_atr_mult > 0):
            raise ValueError("breakout_atr_mult must be > 0")
        if not (self.trail_atr_mult >= self.breakout_atr_mult):
            raise ValueError("trail_atr_mult must be >= breakout_atr_mult")
        if not (self.ma_type in ("ema", "sma")):
            raise ValueError("ma_type must be 'ema' or 'sma'")
        if not (0 < self.risk_per_trade <= 1):
            raise ValueError("risk_per_trade must be in (0, 1]")


@register("atr_breakout")
class ATRBreakout(ChandelierTrailingExit, BaseStrategy):
    """Entry: close crosses above MA + N*ATR. Exit: Chandelier trailing stop."""

    regime = "trend"

    params: ATRBreakoutParams

    def __init__(self, **kwargs):
        super().__init__(ATRBreakoutParams(**kwargs))

    @property
    def min_bars(self) -> int:
        return max(self.params.ma_period, self.params.atr_period) + 5

    # ------------------------------------------------------------------

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        p = self.params

        if p.ma_type == "ema":
            df["MA"] = df["Close"].ewm(span=p.ma_period, adjust=False).mean()
        else:
            df["MA"] = df["Close"].rolling(p.ma_period).mean()

        df["ATR"] = compute_atr(df, p.atr_period)
        df["Upper_band"] = df["MA"] + p.breakout_atr_mult * df["ATR"]

        # Entry signals only
        df["Signal"] = 0
        buy = (
            (df["Close"] > df["Upper_band"])
            & (df["Close"].shift(1) <= df["Upper_band"].shift(1))
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
