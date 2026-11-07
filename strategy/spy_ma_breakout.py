"""SPY MA filter + N-day high breakout + ATR take-profit + MA stop-loss.

Long-only by design.

- Macro gate:  SPY > SPY_MA (block all trades when SPY is below MA)
- Entry:       Close > MA  AND  Close == N-day high
- Exit:
    Take-profit:  Close >= entry + ATR × N
    Stop-loss:    Close < MA (trend invalidated)
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd

from .base import BaseStrategy, StrategyParams, compute_atr, register
from utils import get_logger as _get_logger

_spy_logger = _get_logger("strategy.spy_ma_breakout")

# Module-level SPY filter cache
_spy_filter_cache: Optional[pd.Series] = None
_spy_filter_ma_period: int = 200


@dataclass(frozen=True)
class SPYMABreakoutParams(StrategyParams):
    ma_period: int = 200
    high_period: int = 20
    atr_period: int = 14
    take_profit_atr_mult: float = 4.0
    risk_per_trade: float = 0.02
    max_position_pct: float = 0.95

    grid = {
        "ma_period": [60, 120, 200],
        "high_period": [10, 20, 30],
        "take_profit_atr_mult": [2.0, 3.0, 4.0, 5.0],
    }

    def validate(self):
        if self.ma_period < 20:
            raise ValueError("ma_period must be >= 20")
        if self.high_period < 5:
            raise ValueError("high_period must be >= 5")
        if self.take_profit_atr_mult <= 0:
            raise ValueError("take_profit_atr_mult must be > 0")


@register("spy_ma_breakout")
class SPYMABreakout(BaseStrategy):
    """SPY macro filter + MA trend + N-day breakout + stop-loss + take-profit (long-only).

    Entry
    -----
    SPY filter:   SPY > SPY_MA            (broad market uptrend gate)
    Long:         Close > MA  AND  Close == N-day high

    Exit
    ----
    Take-profit:  Close >= entry + ATR × N
    Stop-loss:    Close < MA (trend broken)
    """

    regime = "trend"
    long_only = True

    params: SPYMABreakoutParams

    def __init__(self, spy_df: pd.DataFrame = None, **kwargs):
        global _spy_filter_cache, _spy_filter_ma_period
        super().__init__(SPYMABreakoutParams(**kwargs))
        self._spy_filter = None
        if spy_df is not None and not spy_df.empty:
            spy_ma = spy_df["Close"].rolling(self.params.ma_period).mean()
            self._spy_filter = spy_df["Close"] > spy_ma
        elif _spy_filter_cache is not None and _spy_filter_ma_period == self.params.ma_period:
            self._spy_filter = _spy_filter_cache
        else:
            try:
                from data import DataProvider
                spy = DataProvider().get_daily("SPY", start="2015-01-01")
                if spy is not None and not spy.empty:
                    spy_ma = spy["Close"].rolling(self.params.ma_period).mean()
                    self._spy_filter = spy["Close"] > spy_ma
                    _spy_filter_cache = self._spy_filter
                    _spy_filter_ma_period = self.params.ma_period
            except Exception:
                _spy_logger.warning("SPY 数据获取失败，宏观过滤器不可用 — 策略将无过滤运行")

    @property
    def min_bars(self) -> int:
        return self.params.ma_period + self.params.high_period + 5

    # -- indicators ---------------------------------------------------------

    def calculate_indicators(
        self, df: pd.DataFrame, df_weekly: pd.DataFrame = None,
    ) -> pd.DataFrame:
        df = df.copy()
        p = self.params

        df["MA"] = df["Close"].rolling(p.ma_period).mean()
        df["N_day_high"] = df["Close"].rolling(p.high_period).max()
        df["ATR"] = compute_atr(df, p.atr_period)

        df["Signal"] = 0
        buy = (df["Close"] > df["MA"]) & (df["Close"] == df["N_day_high"])
        df.loc[buy, "Signal"] = 1

        # SPY macro filter — suppress all when SPY is below MA
        if self._spy_filter is not None:
            aligned = self._spy_filter.reindex(df.index, method="ffill").fillna(False)
            df.loc[~aligned, "Signal"] = 0

        return df

    # -- sizing -------------------------------------------------------------

    def position_size(self, capital: float, price: float, atr: float) -> int:
        return self._risk_budget_size(capital, price, atr,
            self.params.risk_per_trade, 2.0,
            self.params.max_position_pct)

    # -- exit: take-profit + MA stop-loss -----------------------------------

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
        ma = float(df["MA"].iloc[i])

        if price >= entry_price + atr * self.params.take_profit_atr_mult:
            return True, "动态止盈"
        if price < ma:
            return True, "MA止损"
        return False, ""
