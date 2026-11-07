"""Donchian Breakout — Turtle-style channel breakout strategy.

Entry: close breaks above the N-bar Donchian upper channel.
Exit:  Chandelier trailing stop OR price falls back below channel midpoint.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd

from .base import BaseStrategy, StrategyParams, ChandelierTrailingExit, compute_atr, register


@dataclass(frozen=True)
class DonchianBreakoutParams(StrategyParams):
    channel_period: int = 20
    atr_period: int = 14
    trail_atr_mult: float = 3.0
    risk_per_trade: float = 0.02
    max_position_pct: float = 0.95
    volume_confirm: bool = True

    grid = {
        "channel_period": [15, 20, 30, 40],
        "trail_atr_mult": [2.0, 3.0, 4.0],
    }

    def validate(self):
        if not (self.channel_period >= 10): raise ValueError("channel_period must be >= 10")
        if not (self.trail_atr_mult > 0): raise ValueError("trail_atr_mult must be positive")
        if not (0 < self.risk_per_trade <= 1): raise ValueError("validation failed")


@register("donchian_breakout")
class DonchianBreakout(ChandelierTrailingExit, BaseStrategy):
    """Long: close > Donchian upper. Short: close < Donchian lower.
    Exit: trailing stop or price crosses back beyond channel mid."""

    regime = "trend"
    long_only = False

    params: DonchianBreakoutParams

    def __init__(self, **kwargs):
        super().__init__(DonchianBreakoutParams(**kwargs))

    @property
    def min_bars(self) -> int:
        return max(self.params.channel_period, self.params.atr_period, 20) + 5

    # ------------------------------------------------------------------

    def calculate_indicators(
        self, df: pd.DataFrame, df_weekly: pd.DataFrame = None,
    ) -> pd.DataFrame:
        df = df.copy()
        p = self.params

        # Donchian Channel
        df["Donchian_upper"] = df["High"].rolling(p.channel_period).max().shift(1)
        df["Donchian_lower"] = df["Low"].rolling(p.channel_period).min().shift(1)
        df["Donchian_mid"] = (df["Donchian_upper"] + df["Donchian_lower"]) / 2

        df["ATR"] = compute_atr(df, p.atr_period)

        # Volume MA for confirmation
        df["Volume_MA"] = df["Volume"].rolling(20).mean()

        # Signals — exit is handled in check_exit
        df["Signal"] = 0
        buy = df["Close"] > df["Donchian_upper"]
        short = df["Close"] < df["Donchian_lower"]
        if p.volume_confirm:
            vol_ok = df["Volume"] > df["Volume_MA"]
            buy = buy & vol_ok
            short = short & vol_ok
        df.loc[buy, "Signal"] = 1
        df.loc[short, "Signal"] = -1

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
        exit_flag, reason = self._chandelier_exit(df, i, highest_since_entry,
                                                   lowest_since_entry, position)
        if exit_flag:
            return True, reason

        price = float(df["Close"].iloc[i])
        mid = float(df["Donchian_mid"].iloc[i])
        if position and position.get('direction') == 'SHORT':
            if price >= mid:
                return True, "突破通道中线(空)"
        else:
            if price <= mid:
                return True, "跌破通道中线"

        return False, ""
