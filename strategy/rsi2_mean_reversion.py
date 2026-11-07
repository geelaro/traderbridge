"""RSI2 ETF Mean Reversion — Larry Connors' short-term reversal strategy.

Entry: RSI(2) < 5  +  3 consecutive down days  +  Price > MA200.
Exit:  RSI(2) > 70  OR  Price > MA5  OR  hold ≥ 5 days.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseStrategy, StrategyParams, compute_rsi, register


@dataclass(frozen=True)
class RSI2MeanReversionParams(StrategyParams):
    rsi_period: int = 2
    rsi_entry: float = 5.0
    rsi_exit: float = 70.0
    ma_trend: int = 200       # long-term trend MA
    ma_exit: int = 5          # short-term exit MA
    max_hold_days: int = 5    # time stop
    risk_per_trade: float = 0.02
    max_position_pct: float = 0.95
    use_spy_filter: bool = False  # requires spy_df passed to __init__

    grid = {
        "rsi_entry": [3.0, 5.0, 10.0],
        "rsi_exit": [60.0, 70.0, 80.0],
        "max_hold_days": [3, 5, 7],
        "risk_per_trade": [0.01, 0.015, 0.02],
    }

    def validate(self):
        if self.rsi_period < 1:
            raise ValueError("rsi_period must be >= 1")
        if self.rsi_entry >= self.rsi_exit:
            raise ValueError("rsi_entry must be < rsi_exit")
        if not (0 < self.risk_per_trade <= 1):
            raise ValueError("risk_per_trade must be in (0, 1]")


@register("rsi2_mean_reversion")
class RSI2MeanReversion(BaseStrategy):
    """Larry Connors RSI2 mean-reversion for index ETFs.

    Buy when an ETF in a long-term uptrend has a short-term panic sell-off.
    Sell when the bounce materialises or the trade stalls.

    Parameters
    ----------
    spy_df : pd.DataFrame, optional
        SPY OHLCV for the broad-market filter.  If not provided the filter
        is auto-fetched once (or skipped when ``use_spy_filter=False``).
    """

    regime = "mean_reversion"
    long_only = True

    params: RSI2MeanReversionParams

    def __init__(self, spy_df: pd.DataFrame = None, **kwargs):
        super().__init__(RSI2MeanReversionParams(**kwargs))
        self._spy_filter = None
        p = self.params
        if not p.use_spy_filter:
            return
        if spy_df is not None and not spy_df.empty:
            spy_ma = spy_df["Close"].rolling(p.ma_trend).mean()
            self._spy_filter = spy_df["Close"] > spy_ma

    @property
    def min_bars(self) -> int:
        return self.params.ma_trend + 5

    # -- indicators ----------------------------------------------------------

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        p = self.params

        df["MA200"] = df["Close"].rolling(p.ma_trend).mean()
        df["MA5"] = df["Close"].rolling(p.ma_exit).mean()

        # ATR for unified risk sizing
        from .base import compute_atr
        df["ATR"] = compute_atr(df, 14)

        # RSI(2)
        df = compute_rsi(df, p.rsi_period)

        # ---- Entry ----
        df["Signal"] = 0

        # 3 consecutive down days
        down3 = (
            (df["Close"] < df["Close"].shift(1))
            & (df["Close"].shift(1) < df["Close"].shift(2))
            & (df["Close"].shift(2) < df["Close"].shift(3))
        )

        buy = (
            (df["RSI"] < p.rsi_entry)
            & down3
            & (df["Close"] > df["MA200"])
        )
        df.loc[buy, "Signal"] = 1

        # SPY macro filter
        if self._spy_filter is not None:
            aligned = self._spy_filter.reindex(df.index, method="ffill").fillna(True)
            df.loc[~aligned, "Signal"] = 0

        return df

    # -- sizing --------------------------------------------------------------

    def position_size(self, capital: float, price: float, atr: float) -> int:
        return self._risk_budget_size(capital, price, atr,
            self.params.risk_per_trade, 2.0, self.params.max_position_pct)

    # -- exit ----------------------------------------------------------------

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
        rsi = float(df["RSI"].iloc[i])
        ma5 = float(df["MA5"].iloc[i])

        # Time stop
        if position and "date" in position:
            entry_date = position["date"]
            if entry_date in df.index:
                days_held = i - df.index.get_loc(entry_date)
                if days_held >= self.params.max_hold_days:
                    return True, "超时退出"

        if rsi > self.params.rsi_exit:
            return True, "RSI超买退出"
        if price > ma5:
            return True, "均线回归退出"

        return False, ""
