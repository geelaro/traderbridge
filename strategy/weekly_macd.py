"""WeeklyMACD — single MACD crossover on weekly bars."""

from dataclasses import dataclass

import pandas as pd

from .base import BaseStrategy, StrategyParams, compute_atr, compute_macd, resample_weekly, register


@dataclass(frozen=True)
class WeeklyMACDParams(StrategyParams):
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    max_position_pct: float = 0.95

    grid = {
        "macd_fast": [8, 12, 16],
        "macd_slow": [21, 26, 31],
        "macd_signal": [7, 9, 11],
    }


@register("weekly_macd")
class WeeklyMACD(BaseStrategy):
    """MACD golden/death cross on weekly bars. Low trade frequency.

    Entry: MACD crosses above signal line (golden cross) on weekly close.
    Exit:  MACD crosses below signal line (death cross).
    No stop-loss, no take-profit.
    """

    regime = "trend"

    params: WeeklyMACDParams

    def __init__(self, **kwargs):
        super().__init__(WeeklyMACDParams(**kwargs))

    @property
    def min_bars(self) -> int:
        return max(self.params.macd_slow, self.params.macd_signal) + 5

    # ------------------------------------------------------------------

    def calculate_indicators(
        self, df: pd.DataFrame, df_weekly: pd.DataFrame = None,
    ) -> pd.DataFrame:
        p = self.params
        weekly = resample_weekly(df)
        weekly = compute_macd(weekly, p.macd_fast, p.macd_slow, p.macd_signal)
        weekly["ATR"] = compute_atr(weekly, 14)

        weekly["Signal"] = 0
        golden = (
            (weekly["MACD"] > weekly["MACD_signal"])
            & (weekly["MACD"].shift(1) <= weekly["MACD_signal"].shift(1))
        )
        death = (
            (weekly["MACD"] < weekly["MACD_signal"])
            & (weekly["MACD"].shift(1) >= weekly["MACD_signal"].shift(1))
        )
        weekly.loc[golden, "Signal"] = 1
        weekly.loc[death, "Signal"] = -1
        return weekly

    # ------------------------------------------------------------------

    def position_size(self, capital: float, price: float, atr: float) -> int:
        if price <= 0:
            return 0
        return int(capital * self.params.max_position_pct / price)
