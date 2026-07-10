"""StrategyEnsemble — market-regime-aware weighted voting across strategies.

Weights adapt to the current market state (classified from a proxy like SPY):

- TRENDING_UP:   trend × 0.7  +  mean_reversion × 0.3
- TRENDING_DOWN: trend × 0.7  +  mean_reversion × 0.3
- RANGING:       mean_reversion × 0.6  +  trend × 0.4
- HIGH_VOL:      trend × 0.6  +  exit signals priority
- TRANSITIONAL:  equal weight
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseStrategy, StrategyParams, compute_atr
from utils.market_state import MarketStateClassifier, MarketRegime, Volatility


@dataclass(frozen=True)
class EnsembleParams(StrategyParams):
    long_bias_threshold: float = 0.3   # score ≥ +threshold → long
    short_bias_threshold: float = 0.3  # score ≤ -threshold → short
    min_agreement: int = 1             # minimum strategies agreeing for entry

    def validate(self):
        if self.long_bias_threshold < 0:
            raise ValueError("long_bias_threshold must be >= 0")
        if self.short_bias_threshold < 0:
            raise ValueError("short_bias_threshold must be >= 0")


# Default weights per regime
_DEFAULT_WEIGHTS = {
    MarketRegime.TRENDING_UP:   {"trend": 0.7, "mean_reversion": 0.3},
    MarketRegime.TRENDING_DOWN: {"trend": 0.7, "mean_reversion": 0.3},
    MarketRegime.RANGING:       {"trend": 0.4, "mean_reversion": 0.6},
    MarketRegime.TRANSITIONAL:  {"trend": 0.5, "mean_reversion": 0.5},
}


class StrategyEnsemble(BaseStrategy):
    """Combine multiple strategies via regime-weighted voting.

    Parameters
    ----------
    members : list of (strategy, regime_label)
        e.g. ``[(TurtleTrading(), "trend"), (BollingerMeanReversion(), "mean_reversion")]``
    proxy_df : pd.DataFrame
        OHLCV of the market proxy (e.g. SPY) for regime classification.
    weights : dict, optional
        Regime → {regime_label: weight} mapping.  Falls back to defaults.
    """

    regime = None  # mixed — all types
    long_only = False  # ensemble can go both ways

    params: EnsembleParams

    def __init__(
        self,
        members: List[Tuple[BaseStrategy, str]],
        proxy_df: pd.DataFrame,
        weights: Optional[dict] = None,
        member_weights: Optional[List[float]] = None,
        **kwargs,
    ):
        super().__init__(EnsembleParams(**kwargs))
        self._members = members
        self._classifier = MarketStateClassifier(proxy_df)
        self._classifier.calculate()
        self._weights = weights or _DEFAULT_WEIGHTS
        self._member_weights = member_weights  # per-member scalar multipliers
        self._regime_map = {s.regime: r for s, r in members}

    @property
    def min_bars(self) -> int:
        """Worst-case min_bars across members, normalised to the daily index.

        Members declared on weekly bars (e.g. ``freq="W"``) report
        ``min_bars`` in weekly counts, but the ensemble runs on the daily
        index after reindex — convert weekly counts to daily by ×5 so the
        engine waits for genuinely warmed-up signals.
        """
        worst = 0
        for s, _ in self._members:
            mb = s.min_bars
            freq = getattr(getattr(s, "params", None), "freq", "D")
            if freq == "W":
                mb *= 5
            if mb > worst:
                worst = mb
        return worst

    # -- indicators ---------------------------------------------------------

    def calculate_indicators(self, df: pd.DataFrame, df_weekly=None) -> pd.DataFrame:
        """Run each member and combine signals via weighted vote."""
        state = self._classifier.classify()
        weights = self._weights.get(state.regime, {"trend": 0.5, "mean_reversion": 0.5})

        # High-vol: increase trend weight slightly, favour exit
        vol_bonus = 0.0
        if state.volatility == Volatility.HIGH:
            vol_bonus = 0.1

        # Collect each member's signal + full DataFrame
        member_signals: List[pd.Series] = []
        member_regimes: List[str] = []
        member_dfs: List[pd.DataFrame] = []
        for strat, regime_label in self._members:
            try:
                df_sig = strat.calculate_indicators(df, df_weekly=df_weekly)
            except TypeError:
                df_sig = strat.calculate_indicators(df)
            member_signals.append(df_sig["Signal"])
            member_regimes.append(regime_label)
            member_dfs.append(df_sig)

        # Align every member's Signal series to the primary daily index.
        # MTF members (e.g. macd_kdj freq="W") return a weekly index that
        # would otherwise misalign with df.index when summed/compared.
        aligned_signals = [s.reindex(df.index).fillna(0) for s in member_signals]
        n = len(aligned_signals)

        # Per-member weight normalisation: divide by mean so the total
        # regime weight is preserved across members.
        if self._member_weights and len(self._member_weights) >= n and n > 0:
            mw_slice = list(self._member_weights[:n])
            mw_mean = sum(mw_slice) / n
            mw_factor = [m / mw_mean if mw_mean > 0 else 1.0 for m in mw_slice]
        else:
            mw_factor = [1.0] * n

        score = pd.Series(0.0, index=df.index)
        for idx, (sig_series, regime_label) in enumerate(zip(aligned_signals, member_regimes)):
            w = weights.get(regime_label, 0.33) * mw_factor[idx]
            if regime_label == "trend":
                w += vol_bonus
            score = score + sig_series.astype(float) * w

        # Consensus count — how many members agree on direction
        long_votes = sum((s > 0).astype(int) for s in aligned_signals)
        short_votes = sum((s < 0).astype(int) for s in aligned_signals)

        long_mask = (long_votes >= self.params.min_agreement) & (score >= self.params.long_bias_threshold)
        short_mask = (short_votes >= self.params.min_agreement) & (score <= -self.params.short_bias_threshold)

        result = df.copy()
        result["Signal"] = 0
        result.loc[long_mask & ~short_mask, "Signal"] = 1
        result.loc[short_mask & ~long_mask, "Signal"] = -1
        # True conflict — both masks true: arbitrate by score sign
        conflict = long_mask & short_mask
        if conflict.any():
            result.loc[conflict, "Signal"] = np.sign(score[conflict]).astype(int)

        # Carry forward ATR from first member that has it (from cached df_sig)
        for df_sig in member_dfs:
            if "ATR" in df_sig.columns:
                result["ATR"] = self._align_to_daily(df_sig["ATR"], df.index, fill_value=0)
                break
        else:
            result["ATR"] = compute_atr(result, 14)

        return result

    @staticmethod
    def _align_to_daily(series: pd.Series, index: pd.Index, fill_value: float = 0) -> pd.Series:
        """Align a member output series to the ensemble's daily index.

        Weekly members return sparse weekly indices.  Forward-fill their latest
        known value so downstream sizing sees a usable ATR on non-Friday bars.
        """
        return series.reindex(index, method="ffill").fillna(fill_value)

    # -- sizing -------------------------------------------------------------

    def position_size(self, capital: float, price: float, atr: float) -> int:
        return self._risk_budget_size(capital, price, atr, 0.02, 2.0, 0.95)
