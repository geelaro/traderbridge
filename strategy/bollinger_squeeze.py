"""Bollinger Squeeze — volatility contraction / expansion strategy.

Entry: BB width is at a low percentile (squeeze) AND close breaks above upper BB.
Exit:  close falls back below mid-BB OR Chandelier trailing stop.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseStrategy, StrategyParams, ChandelierTrailingExit, compute_atr, compute_bollinger


@dataclass(frozen=True)
class BollingerSqueezeParams(StrategyParams):
    bb_period: int = 20
    bb_std: float = 2.0
    squeeze_lookback: int = 125
    squeeze_percentile: float = 10.0
    atr_period: int = 14
    trail_atr_mult: float = 3.0
    risk_per_trade: float = 0.02
    max_position_pct: float = 0.95

    grid = {
        "bb_period": [15, 20, 25],
        "bb_std": [1.5, 2.0, 2.5],
        "squeeze_percentile": [5, 10, 15],
        "trail_atr_mult": [2.0, 3.0, 4.0],
    }

    def validate(self):
        if not (self.squeeze_lookback > self.bb_period):
            raise ValueError("squeeze_lookback must be > bb_period")
        if not (0 < self.squeeze_percentile <= 50):
            raise ValueError("squeeze_percentile must be in (0, 50]")
        if not (self.trail_atr_mult > 0):
            raise ValueError("trail_atr_mult must be positive")
        if not (0 < self.risk_per_trade <= 1):
            raise ValueError("risk_per_trade must be in (0, 1]")


class BollingerSqueeze(ChandelierTrailingExit, BaseStrategy):
    """Wait for volatility contraction (squeeze), then enter on upward breakout."""

    regime = "trend"

    params: BollingerSqueezeParams

    def __init__(self, **kwargs):
        super().__init__(BollingerSqueezeParams(**kwargs))

    @property
    def min_bars(self) -> int:
        p = self.params
        return max(p.bb_period, p.squeeze_lookback, p.atr_period) + 5

    # ------------------------------------------------------------------

    def calculate_indicators(
        self, df: pd.DataFrame, df_weekly: pd.DataFrame = None,
    ) -> pd.DataFrame:
        df = df.copy()
        p = self.params

        df = compute_bollinger(df, p.bb_period, p.bb_std)
        df["ATR"] = compute_atr(df, p.atr_period)

        # Rolling percentile of bandwidth: is current width in the lowest X%?
        df["BB_width_pct"] = (
            df["BB_width"]
            .rolling(p.squeeze_lookback)
            .apply(
                lambda x: (x.iloc[-1] <= x).sum() / len(x) * 100,
                raw=False,
            )
        )
        df["is_squeeze"] = df["BB_width_pct"] <= p.squeeze_percentile

        # ---- Signals ----
        df["Signal"] = 0
        buy = (
            df["is_squeeze"]
            & (df["Close"] > df["BB_upper"])
            & (df["Close"].shift(1) <= df["BB_upper"].shift(1))
        )
        df.loc[buy, "Signal"] = 1

        sell = (
            (df["Close"] < df["BB_mid"])
            & (df["Close"].shift(1) >= df["BB_mid"].shift(1))
        )
        df.loc[sell, "Signal"] = -1

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
        exit_flag, reason = self._chandelier_exit(df, i, highest_since_entry)
        if exit_flag:
            return True, reason

        if int(df["Signal"].iloc[i]) == -1:
            return True, "回归中轨"

        return False, ""
