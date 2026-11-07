"""Bollinger Mean Reversion — counter-trend strategy.

Entry: price at/under lower BB + RSI oversold + RSI turned up.
Exit:  price back to mid-BB (mean reversion) OR ATR stop-loss.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseStrategy, StrategyParams, compute_atr, compute_bollinger, compute_rsi


@dataclass(frozen=True)
class BollingerMeanReversionParams(StrategyParams):
    bb_period: int = 20
    bb_std: float = 2.0
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_turnup: float = 3.0
    oversold_window: int = 5
    atr_period: int = 14
    trail_atr_mult: float = 2.0
    risk_per_trade: float = 0.02
    max_position_pct: float = 0.95

    grid = {
        "bb_period": [15, 20, 25],
        "bb_std": [1.5, 2.0, 2.5],
        "rsi_oversold": [25, 30, 35],
        "rsi_turnup": [2.0, 3.0, 5.0],
        "oversold_window": [3, 5, 7],
        "trail_atr_mult": [1.5, 2.0, 3.0],
    }

    def validate(self):
        if not (self.bb_std > 0): raise ValueError("bb_std must be positive")
        if not (0 < self.rsi_oversold < 50): raise ValueError("rsi_oversold must be in (0, 50)")
        if not (self.rsi_turnup > 0): raise ValueError("rsi_turnup must be positive")
        if not (self.trail_atr_mult > 0): raise ValueError("trail_atr_mult must be positive")
        if not (0 < self.risk_per_trade <= 1): raise ValueError("validation failed")


class BollingerMeanReversion(BaseStrategy):
    """Buy at lower BB when RSI is oversold and turning up; sell at mid-BB."""

    regime = "mean_reversion"

    params: BollingerMeanReversionParams

    def __init__(self, **kwargs):
        super().__init__(BollingerMeanReversionParams(**kwargs))

    @property
    def min_bars(self) -> int:
        return max(self.params.bb_period, self.params.rsi_period,
                   self.params.atr_period) + 10

    # ------------------------------------------------------------------

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        p = self.params

        df = compute_bollinger(df, p.bb_period, p.bb_std)
        df["ATR"] = compute_atr(df, p.atr_period)

        # RSI
        df = compute_rsi(df, p.rsi_period)

        # RSI minimum over lookback window — used to detect turn-up
        df["RSI_low5"] = df["RSI"].rolling(p.oversold_window).min()

        # Oversold zone: price at/under lower BB AND RSI oversold
        df["Oversold_zone"] = (
            (df["Close"] <= df["BB_lower"]) & (df["RSI"] < p.rsi_oversold)
        )

        # ---- Signals ----
        df["Signal"] = 0

        # Buy: RSI confirmed turning up AND oversold zone was active
        # within the last *oversold_window* bars (decoupled from current bar).
        buy = (
            ((df["RSI"] - df["RSI_low5"]) >= p.rsi_turnup)
            & (df["Oversold_zone"].rolling(p.oversold_window).max() > 0)
        )
        df.loc[buy, "Signal"] = 1

        sell = (
            (df["Close"] >= df["BB_mid"])
            & (df["Close"].shift(1) < df["BB_mid"].shift(1))
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
        price = float(df["Close"].iloc[i])
        atr = float(df["ATR"].iloc[i])
        p = self.params

        stop_loss = entry_price - atr * p.trail_atr_mult
        if price <= stop_loss:
            return True, "止损"

        # Primary exit handled by Signal == -1 (mid-BB cross)
        if int(df["Signal"].iloc[i]) == -1:
            return True, "均值回归"

        return False, ""
