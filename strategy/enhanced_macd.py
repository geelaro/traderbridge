"""Enhanced MACD — dual-MA + MACD + RSI filter + ATR stop-loss / take-profit."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .base import (
    BaseStrategy,
    StrategyParams,
    compute_atr,
    compute_macd,
    compute_rsi,
)


@dataclass(frozen=True)
class EnhancedMACDParams(StrategyParams):
    short_ma: int = 20
    long_ma: int = 50
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    atr_period: int = 14
    trail_atr_mult: float = 2.0
    take_profit_mult: float = 4.0
    risk_per_trade: float = 0.02
    max_position_pct: float = 0.95

    grid = {
        "short_ma": [10, 20, 30],
        "long_ma": [40, 50, 60],
        "trail_atr_mult": [1.5, 2.0, 3.0],
        "take_profit_mult": [3.0, 4.0, 5.0],
    }
    volume_ma_period: int = 20

    def validate(self):
        if not (self.short_ma < self.long_ma): raise ValueError("short_ma must be < long_ma")
        if not (0 < self.risk_per_trade <= 1): raise ValueError("validation failed")


class EnhancedMACDStrategy(BaseStrategy):
    """Dual-MA + MACD with RSI filter, ATR stop-loss, volume confirmation.

    Entry: MA uptrend AND MACD histogram turns positive AND RSI within range
           AND volume confirms.
    Exit:  ATR stop-loss / take-profit, OR MACD histogram turns negative, OR
           MA death cross.
    """

    regime = "trend"

    params: EnhancedMACDParams

    def __init__(self, **kwargs):
        super().__init__(EnhancedMACDParams(**kwargs))

    @property
    def min_bars(self) -> int:
        return max(self.params.long_ma, self.params.atr_period,
                   self.params.rsi_period, self.params.volume_ma_period) + 5

    # ------------------------------------------------------------------

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        p = self.params

        # Moving averages
        df["SMA_short"] = df["Close"].rolling(p.short_ma).mean()
        df["SMA_long"] = df["Close"].rolling(p.long_ma).mean()

        # MACD
        df = compute_macd(df, p.macd_fast, p.macd_slow, p.macd_signal)

        # RSI
        df = compute_rsi(df, p.rsi_period)

        # ATR
        df["ATR"] = compute_atr(df, p.atr_period)

        # Volume MA
        df["Volume_MA"] = df["Volume"].rolling(p.volume_ma_period).mean()
        df["Volume_ratio"] = df["Volume"] / df["Volume_MA"]

        # ---- Signals ----
        df["Signal"] = 0

        buy = (
            (df["SMA_short"] > df["SMA_long"])
            & (df["MACD_hist"] > 0)
            & (df["MACD_hist"].shift(1) <= 0)
            & df["RSI"].between(p.rsi_oversold, p.rsi_overbought)
            & (df["Volume_ratio"] > 0.8)
        )
        df.loc[buy, "Signal"] = 1

        macd_sell = (df["MACD_hist"] < 0) & (df["MACD_hist"].shift(1) >= 0)
        ma_sell = (
            (df["SMA_short"] < df["SMA_long"])
            & (df["SMA_short"].shift(1) >= df["SMA_long"].shift(1))
        )
        df.loc[macd_sell | ma_sell, "Signal"] = -1

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

        trail_stop = highest_since_entry - atr * p.trail_atr_mult
        take_profit = entry_price + atr * p.take_profit_mult

        if price <= trail_stop:
            return True, "移动止损"
        if price >= take_profit:
            return True, "止盈"
        if int(df["Signal"].iloc[i]) == -1:
            return True, "卖出信号"
        return False, ""
