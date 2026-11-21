"""Strategy interface — every strategy must implement this.

Design
------
- `calculate_indicators(df)` → DataFrame with indicators + 'Signal' column
- `check_exit(...)` → (exit_flag, reason) — exit logic lives in the strategy
- `position_size(...)` → int — sizing logic lives in the strategy
- `min_bars` property — minimum bars before first signal

The backtest engine never inspects strategy internals — it acts solely on
public methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Type

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Strategy registry — populated by the @register decorator on subclasses
# ---------------------------------------------------------------------------

_STRATEGY_REGISTRY: dict[str, type] = {}


def register(name: str) -> Callable[[type], type]:
    """Decorator: register a strategy under ``name`` in the global map.

    Replaces the old manual ``STRATEGY_MAP[name] = SomeStrategy`` in
    ``strategy/__init__.py``.  Forgetting to register meant the
    optimizer / scanner couldn't find the strategy.

    Usage::

        @register("trend_follower")
        class TrendFollower(BaseStrategy):
            ...

    The registry is read via ``strategy.STRATEGY_MAP`` (which re-exports
    this module-level dict).  Duplicate names raise ValueError at import
    time so name collisions can't go silent.
    """
    def _wrap(cls: type) -> type:
        if name in _STRATEGY_REGISTRY and _STRATEGY_REGISTRY[name] is not cls:
            raise ValueError(
                f"Strategy name '{name}' already registered to "
                f"{_STRATEGY_REGISTRY[name].__name__}; cannot redefine on "
                f"{cls.__name__}"
            )
        _STRATEGY_REGISTRY[name] = cls
        cls._strategy_name = name  # type: ignore[attr-defined]
        return cls
    return _wrap


def get_strategy_map() -> dict[str, type]:
    """Live view of the registry — kept as a function so callers always
    see late-registered strategies."""
    return _STRATEGY_REGISTRY


# ---------------------------------------------------------------------------
# Parameters — frozen dataclass, validated on construction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StrategyParams:
    """Immutable parameter bag.  Subclass per strategy with typed defaults."""

    def validate(self):
        """Override to add parameter sanity checks.  Called by __post_init__."""
        pass

    def __post_init__(self):
        self.validate()

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# Base strategy
# ---------------------------------------------------------------------------


class BaseStrategy(ABC):
    """Every strategy MUST subclass this and implement the abstract methods.

    Subclass contract
    -----------------
    1. `calculate_indicators(df)` — return `df` with indicator columns AND
       a `'Signal'` column (1 = bullish, -1 = bearish, 0 = neutral).
       The action depends on the current position and `long_only`.
    2. `min_bars` — how many bars must elapse before the first signal fires.

    Optional overrides (sensible defaults provided):
    3. `check_exit(...)` — exit conditions (default: opposite signal)
    4. `position_size(...)` — position sizing (default: 95 % of capital)
    5. `entry_signal(...)` — entry trigger (default: Signal == 1)

    Optional class attributes:
    6. `regime` — "trend" | "mean_reversion" | None (mixed/unrestricted)
    """

    params: StrategyParams  # set by __init_subclass__ or constructor
    regime: Optional[str] = None  # "trend", "mean_reversion", or None for mixed
    long_only: bool = True  # set False to enable short entries

    def __init__(self, params: Optional[StrategyParams] = None):
        if params is not None:
            self.params = params

    # -- mandatory -----------------------------------------------------------

    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame, df_weekly: pd.DataFrame = None) -> pd.DataFrame:
        """Return *df* with indicator columns and a `'Signal'` column added.

        Signal semantics
        ----------------
          1  = long entry   -1  = short entry   0  = neutral

        *df_weekly* (optional) — pre-resampled weekly OHLCV for multi-timeframe
        strategies.  When provided the strategy can compute weekly indicators
        and map them back to the daily index for precise entry timing.

        The returned DataFrame may have a different index (e.g. weekly) from
        the input — the engine handles alignment.

        Must always include an `'ATR'` column (for position sizing).
        """
        ...

    @property
    @abstractmethod
    def min_bars(self) -> int:
        """Minimum rows needed before `calculate_indicators` produces valid data."""
        ...

    # -- optional (default implementations) ----------------------------------

    def position_size(self, capital: float, price: float, atr: float) -> int:
        """Default: invest 95 % of available capital in one trade."""
        if price <= 0:
            return 0
        return int(capital * 0.95 / price)

    @staticmethod
    def _risk_budget_size(
        capital: float,
        price: float,
        atr: float,
        risk_pct: float,
        stop_atr_mult: float,
        max_pct: float,
    ) -> int:
        """Risk-budget position sizing used by most strategies.

        Shares = (capital × risk_pct) / (atr × stop_atr_mult), capped at
        (capital × max_pct / price).
        """
        if pd.isna(atr) or atr <= 0 or price <= 0:
            return 0
        risk_dollar = capital * risk_pct
        stop_distance = atr * stop_atr_mult
        if stop_distance <= 0:
            return 0
        shares = int(risk_dollar / stop_distance)
        max_shares = int(capital * max_pct / price)
        return max(0, min(shares, max_shares))

    def entry_signal(self, df: pd.DataFrame, i: int) -> int:
        """Return direction signal at bar *i*: 1=long, -1=short, 0=neutral.

        Default: read from ``Signal`` column. Override for custom logic.
        """
        return int(df["Signal"].iloc[i])

    def check_exit(
        self,
        df: pd.DataFrame,
        i: int,
        entry_price: float,
        highest_since_entry: float,
        lowest_since_entry: Optional[float] = None,
        position: Optional[Dict] = None,
    ) -> Tuple[bool, str]:
        """Return (exit_now, reason).

        *lowest_since_entry* is used by short positions (cover when price
        rises above the trailing cover level).  The engine passes it together
        with *highest_since_entry* — strategies can ignore one depending on
        the position direction in ``position.get('direction')``.
        """
        direction = (position or {}).get("direction", "LONG")
        exit_signal = 1 if direction == "SHORT" else -1
        if int(df["Signal"].iloc[i]) == exit_signal:
            return True, "空头平仓信号" if direction == "SHORT" else "卖出信号"
        return False, ""


# ---------------------------------------------------------------------------
# Chandelier trailing stop Mixin
# ---------------------------------------------------------------------------


class ChandelierTrailingExit:
    """Mixin providing Chandelier trailing-stop exit for long & short positions.

    The host strategy must declare ``params.trail_atr_mult``.
    Use by inheriting *before* ``BaseStrategy``.

    Call ``self._chandelier_exit(df, i, highest, lowest, position)``
    in ``check_exit()`` — it auto-picks the correct stop direction.
    """

    def _chandelier_exit(self, df, i, highest_since_entry,
                         lowest_since_entry=None, position=None):
        """Return (exit_now, reason). Direction-aware: chooses long or short stop."""
        if position and position.get('direction') == 'SHORT':
            return self._chandelier_cover(df, i, lowest_since_entry)
        return self._chandelier_stop(df, i, highest_since_entry)

    def _chandelier_stop(self, df, i, highest_since_entry):
        """Long exit: price <= highest - trail_atr_mult × ATR."""
        price = float(df["Close"].iloc[i])
        atr = float(df["ATR"].iloc[i])
        stop = highest_since_entry - self.params.trail_atr_mult * atr
        if price <= stop:
            return True, "移动止损"
        return False, ""

    def _chandelier_cover(self, df, i, lowest_since_entry):
        """Short cover: price >= lowest + trail_atr_mult × ATR."""
        if lowest_since_entry is None:
            return False, ""
        price = float(df["Close"].iloc[i])
        atr = float(df["ATR"].iloc[i])
        cover = lowest_since_entry + self.params.trail_atr_mult * atr
        if price >= cover:
            return True, "移动止损(空)"
        return False, ""


# ---------------------------------------------------------------------------
# Built-in helpers that subclasses can reuse
# ---------------------------------------------------------------------------


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Welles Wilder ATR via EMA smoothing."""
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def compute_macd(
    df: pd.DataFrame, fast: int, slow: int, signal: int
) -> pd.DataFrame:
    """Add MACD / MACD_signal / MACD_hist columns to *df*."""
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    return df


def compute_kdj(
    df: pd.DataFrame, n: int = 9, k_period: int = 3, d_period: int = 3
) -> pd.DataFrame:
    """Add K / D / J columns to *df* (vectorised via ewm)."""
    low_n = df["Low"].rolling(n).min()
    high_n = df["High"].rolling(n).max()
    rsv = (df["Close"] - low_n) / (high_n - low_n).replace(0, np.nan) * 100

    # Forward-fill gaps (e.g. zero-range bars), then seed initial bar at 50
    rsv = rsv.ffill().fillna(50.0)

    df["K"] = rsv.ewm(alpha=1 / k_period, adjust=False).mean()
    df["D"] = df["K"].ewm(alpha=1 / d_period, adjust=False).mean()
    df["J"] = 3 * df["K"] - 2 * df["D"]
    return df


def compute_bollinger(
    df: pd.DataFrame, period: int = 20, std_mult: float = 2.0
) -> pd.DataFrame:
    """Add BB_mid / BB_upper / BB_lower / BB_width columns to *df*."""
    df["BB_mid"] = df["Close"].rolling(period).mean()
    bb_std = df["Close"].rolling(period).std()
    df["BB_upper"] = df["BB_mid"] + std_mult * bb_std
    df["BB_lower"] = df["BB_mid"] - std_mult * bb_std
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_mid"].replace(0, np.nan)
    return df


def resample_weekly(df: pd.DataFrame, weekday: str = "FRI") -> pd.DataFrame:
    """Resample daily OHLCV to weekly (default Friday close).

    Parameters
    ----------
    weekday : str
        Weekday anchor for resample, e.g. ``"FRI"`` (US), ``"WED"`` (some Asian markets).
    """
    return (
        df.resample(f"W-{weekday}")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna()
    )


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add RSI column to *df* using Wilder EMA smoothing."""
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add ADX / +DI / -DI / ATR columns to *df* (Wilder smoothing via EMA).

    Includes ATR so callers don't need a separate ``compute_atr()`` call.
    """
    high_diff = df["High"].diff()
    low_diff = -df["Low"].diff()
    plus_dm = pd.Series(
        (high_diff > low_diff) & (high_diff > 0), index=df.index
    ).astype(float) * high_diff
    plus_dm = plus_dm.clip(lower=0)
    minus_dm = pd.Series(
        (low_diff > high_diff) & (low_diff > 0), index=df.index
    ).astype(float) * low_diff
    minus_dm = minus_dm.clip(lower=0)

    atr_s = compute_atr(df, period).replace(0, np.nan).clip(lower=1e-8)
    df["ATR"] = atr_s
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_s
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_s

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["ADX"] = dx.ewm(alpha=1 / period, adjust=False).mean()
    df["+DI"] = plus_di
    df["-DI"] = minus_di
    return df
