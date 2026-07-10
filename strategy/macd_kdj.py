"""MACDKDJStrategy — unified MACD+KDJ signal strategy (D/W freq, optional ATR stop).

Replaces the duplicated ``weekly_macd_kdj`` and ``daily_macd_kdj`` modules.
Backward-compat aliases ``WeeklyMACD_KDJ`` and ``DailyMACD_KDJ`` are provided.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd

from .base import (
    BaseStrategy,
    ChandelierTrailingExit,
    StrategyParams,
    compute_atr,
    compute_kdj,
    compute_macd,
    register,
    resample_weekly,
)


# -- unified params -----------------------------------------------------------


@dataclass(frozen=True)
class MACDKDJParams(StrategyParams):
    freq: str = "W"
    use_atr_stop: bool = False
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    kdj_n: int = 9
    kdj_k: int = 3
    kdj_d: int = 3
    atr_period: int = 14
    trail_atr_mult: float = 3.0
    risk_per_trade: float = 0.02
    max_position_pct: float = 0.95

    grid = {
        "macd_fast": [8, 12, 16],
        "macd_slow": [21, 26, 31],
        "macd_signal": [7, 9, 11],
        "kdj_n": [7, 9, 14],
        "kdj_k": [2, 3, 5],
        "kdj_d": [2, 3, 5],
        "trail_atr_mult": [2.0, 3.0, 4.0],
    }

    def validate(self):
        if not (self.macd_fast < self.macd_slow):
            raise ValueError("macd_fast must be < macd_slow")
        if not (self.freq in ("W", "D")):
            raise ValueError("freq must be 'W' or 'D'")


# -- backward-compat param classes (preserve grid for optimizer) --------------


@dataclass(frozen=True)
class WeeklyMACDKDJParams(MACDKDJParams):
    freq: str = "W"
    use_atr_stop: bool = False

    grid = {"kdj_n": [7, 9, 14], "kdj_k": [2, 3, 5], "kdj_d": [2, 3, 5]}


@dataclass(frozen=True)
class DailyMACDKDJParams(MACDKDJParams):
    freq: str = "D"
    use_atr_stop: bool = True

    grid = {
        "macd_fast": [8, 12, 16],
        "macd_slow": [21, 26, 31],
        "macd_signal": [7, 9, 11],
        "kdj_n": [7, 9, 14],
        "kdj_k": [2, 3, 5],
        "kdj_d": [2, 3, 5],
        "trail_atr_mult": [2.0, 3.0, 4.0],
    }


# -- unified strategy ---------------------------------------------------------


@register("macd_kdj")
class MACDKDJStrategy(ChandelierTrailingExit, BaseStrategy):
    """KDJ golden cross entry + MACD death cross exit.

    Parameters
    ----------
    freq : "W" | "D"
        Resample to weekly bars before computing indicators.
    use_atr_stop : bool
        Enable ATR Chandelier trailing stop + risk-budget position sizing.
    """

    regime = None

    params: MACDKDJParams

    def __init__(self, params: Optional[MACDKDJParams] = None, **kwargs):
        if params is not None:
            super().__init__(params)
        else:
            super().__init__(MACDKDJParams(**kwargs))

    @property
    def min_bars(self) -> int:
        p = self.params
        n = max(p.macd_slow, p.macd_signal, p.kdj_n)
        if p.use_atr_stop:
            n = max(n, p.atr_period)
        return n + 5

    # -- indicators -----------------------------------------------------------

    def calculate_indicators(self, df: pd.DataFrame, df_weekly: pd.DataFrame = None) -> pd.DataFrame:
        p = self.params
        if p.freq == "W":
            if df_weekly is not None and not df_weekly.empty:
                # MTF mode: compute indicators on weekly data, then map
                # signals back to daily index for precise entry timing.
                work = df_weekly.copy()
                work = compute_macd(work, p.macd_fast, p.macd_slow, p.macd_signal)
                work = compute_kdj(work, p.kdj_n, p.kdj_k, p.kdj_d)
                work["ATR"] = compute_atr(work, p.atr_period)
                work["Signal"] = 0
                golden = (work["K"] > work["D"]) & (work["K"].shift(1) <= work["D"].shift(1))
                death = (work["MACD"] < work["MACD_signal"]) & (work["MACD"].shift(1) >= work["MACD_signal"].shift(1))
                work.loc[golden, "Signal"] = 1
                work.loc[death, "Signal"] = -1

                # Map weekly signals back to daily index (forward-fill)
                daily = df.copy()
                daily["Signal"] = 0
                for col in ["K", "D", "J", "MACD", "MACD_signal", "MACD_hist", "ATR"]:
                    if col in work.columns:
                        daily[col] = work[col].reindex(daily.index, method="ffill")
                daily["Signal"] = work["Signal"].reindex(daily.index, method="ffill").fillna(0).astype(int)
                daily["ATR"] = daily["ATR"].ffill().fillna(0)
                return daily
            else:
                df = resample_weekly(df)

        df = df.copy()
        df = compute_macd(df, p.macd_fast, p.macd_slow, p.macd_signal)
        df = compute_kdj(df, p.kdj_n, p.kdj_k, p.kdj_d)
        df["ATR"] = compute_atr(df, p.atr_period)

        df["Signal"] = 0
        golden_kdj = (
            (df["K"] > df["D"])
            & (df["K"].shift(1) <= df["D"].shift(1))
        )
        death_macd = (
            (df["MACD"] < df["MACD_signal"])
            & (df["MACD"].shift(1) >= df["MACD_signal"].shift(1))
        )
        df.loc[golden_kdj, "Signal"] = 1
        df.loc[death_macd, "Signal"] = -1
        return df

    # -- sizing ---------------------------------------------------------------

    def position_size(self, capital: float, price: float, atr: float) -> int:
        if self.params.use_atr_stop:
            return self._risk_budget_size(
                capital, price, atr,
                self.params.risk_per_trade,
                self.params.trail_atr_mult,
                self.params.max_position_pct,
            )
        if price <= 0:
            return 0
        return int(capital * self.params.max_position_pct / price)

    # -- exit -----------------------------------------------------------------

    def check_exit(
        self,
        df: pd.DataFrame,
        i: int,
        entry_price: float,
        highest_since_entry: float,
        lowest_since_entry: Optional[float] = None,
        position: Optional[Dict] = None,
    ) -> Tuple[bool, str]:
        if self.params.use_atr_stop:
            exit_flag, reason = self._chandelier_exit(
                df, i, highest_since_entry, lowest_since_entry, position
            )
            if exit_flag:
                return True, reason
        if int(df["Signal"].iloc[i]) == -1:
            return True, "MACD死叉" if self.params.use_atr_stop else "卖出信号"
        return False, ""


# -- backward-compat aliases --------------------------------------------------


@register("weekly_macd_kdj")
class WeeklyMACD_KDJ(MACDKDJStrategy):
    """Backward-compat. Use MACDKDJStrategy(freq="W", use_atr_stop=False)."""

    regime = "trend"

    params: WeeklyMACDKDJParams

    def __init__(self, **kwargs):
        super().__init__(WeeklyMACDKDJParams(**kwargs))


@register("daily_macd_kdj")
class DailyMACD_KDJ(MACDKDJStrategy):
    """Backward-compat. Use MACDKDJStrategy(freq="D", use_atr_stop=True)."""

    params: DailyMACDKDJParams

    def __init__(self, **kwargs):
        super().__init__(DailyMACDKDJParams(**kwargs))
