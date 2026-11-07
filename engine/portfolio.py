"""Portfolio backtest — multi-symbol, shared capital pool.

Usage:
    pipenv run python engine/portfolio.py
"""

import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Optional, Tuple

if __name__ == "__main__":
    from utils.bootstrap import setup_runtime
    setup_runtime()

from utils.font import setup_chinese_font
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from data import DataProvider
from broker import OrderSide, OrderStatus
from strategy import BaseStrategy, STRATEGY_MAP as _STRATEGY_MAP
from engine.trader import BacktestEngine, BacktestResult
from engine.execution import ExecutionConfig, ExecutionModel, ExecutionTiming


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Leg:
    """One leg of the portfolio — a symbol + strategy (or ensemble of strategies).

    Ensemble mode is triggered when ``members`` is a non-empty list.  In that
    mode ``strategy_name`` is ignored and the ensemble is built from the
    listed member strategy names, each paired with its ``regime_tags`` label.
    """

    symbol: str
    strategy_name: str = ""
    params: dict = field(default_factory=dict)
    members: Optional[List[str]] = None
    regime_tags: Optional[List[str]] = None
    ensemble_params: Optional[dict] = None

    def create_strategy(
        self, proxy_df: Optional["pd.DataFrame"] = None
    ) -> BaseStrategy:
        # ── ensemble mode ───────────────────────────────────────────
        if self.members:
            if proxy_df is None:
                raise ValueError(
                    "proxy_df is required for ensemble legs; "
                    "run() provides it automatically"
                )
            from strategy.ensemble import StrategyEnsemble
            from strategy import STRATEGY_MAP as _SMAP

            # Build member (strategy_instance, regime_label) pairs
            pairs = []
            tags = self.regime_tags or ["trend"] * len(self.members)
            if len(tags) != len(self.members):
                raise ValueError(
                    f"regime_tags length ({len(tags)}) != members length "
                    f"({len(self.members)}) for {self.symbol}"
                )
            member_params = self.params.get("members", {})
            for name, tag in zip(self.members, tags):
                cls = _SMAP.get(name)
                if cls is None:
                    raise ValueError(f"Unknown ensemble member: {name}")
                mp = member_params.get(name, {})
                pairs.append((cls(**mp), tag))

            ep = dict(self.ensemble_params or {})
            mw = ep.pop("member_weights", None)
            return StrategyEnsemble(members=pairs, proxy_df=proxy_df, member_weights=mw, **ep)

        # ── single-strategy mode ────────────────────────────────────
        cls = _STRATEGY_MAP.get(self.strategy_name)
        if cls is None:
            raise ValueError(f"Unknown strategy: {self.strategy_name}")
        return cls(**self.params)


@dataclass
class PortfolioTrade:
    """A single round-trip trade recorded during portfolio backtest."""

    symbol: str
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp] = None
    qty: int = 0
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    reason: str = ""
    hold_days: Optional[int] = None
    direction: str = "LONG"


# ---------------------------------------------------------------------------
# Leg runtime state (replaces dict[str, Any] pattern)
# ---------------------------------------------------------------------------

@dataclass
class LegState:
    leg: Leg
    strategy: object  # BaseStrategy
    df: pd.DataFrame
    position: int = 0
    entry_price: float = 0.0
    highest: float = 0.0
    lowest: float = float('inf')
    capital_allocated: float = 0.0
    pending_order: Optional[object] = None  # ExecutionPlan
    last_price: float = 0.0
    entry_date: Optional[pd.Timestamp] = None

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __setitem__(self, key: str, value):
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)


@dataclass
class PortfolioState:
    """Mutable state threaded through the portfolio backtest loop.

    Replaces the previous pattern where ``cash`` was a local variable in
    ``run()`` that every helper had to return-and-rebind. Helpers mutate
    ``state.cash`` directly, which removes a class of "forgot to update cash"
    bugs and makes the data-flow obvious.
    """

    cash: float
    leg_state: dict = field(default_factory=dict)  # int → LegState
    trades: List["PortfolioTrade"] = field(default_factory=list)
    open_trade_idx: dict = field(default_factory=dict)  # leg_index → trades index
    equity_history: List[Tuple[pd.Timestamp, float]] = field(default_factory=list)
    daily_new_count: int = 0
    prev_date: Optional[pd.Timestamp] = None


# ---------------------------------------------------------------------------
# Portfolio backtest engine
# ---------------------------------------------------------------------------


class PortfolioBacktest:
    """Multi-symbol backtest with shared cash pool.

    Each leg gets an equal fraction of capital by default.
    All legs share a single cash pool — when one exits, cash
    is returned to the pool for other legs to use.
    """

    def __init__(
        self,
        legs: List[Leg],
        initial_capital: float = 100000,
        allocation: str = "equal",
        max_positions: int = 10,
        commission_rate: float = 0.0003,
        slippage_pct: float = 0.0001,
        # --- portfolio risk ---
        max_symbol_weight: float = 0.0,
        max_daily_new_positions: int = 0,
        max_gross_exposure: float = 0.0,
        # --- execution constraints ---
        lot_size: int = 0,
        max_participation_rate: float = 0.0,
        # --- sizing ---
        sizing_mode: str = "fixed_capital",
        risk_per_trade: float = 0.005,
        risk_atr_mult: float = 2.0,
        execution_model: ExecutionModel | None = None,
        execution_timing: str = "next_open",
        # --- enhanced risk ---
        max_sector_weight: float = 0.0,
        sector_map: Optional[dict] = None,
        cooldown_after_stop_days: int = 0,
        # --- short selling ---
        short_margin_ratio: float = 1.5,
        # --- single-symbol cap (parity with live RiskController) ---
        # Defaults to RiskLimits().max_position_pct (0.30) so portfolio
        # backtest semantics match live execution. Pass max_position_pct=0
        # (or any falsy non-None) to disable the cap entirely — e.g. when
        # running a 1-leg portfolio at full cash.
        max_position_pct: Optional[float] = None,
    ):
        self.legs = legs
        self.initial_capital = initial_capital
        self.allocation = allocation
        self.max_positions = max_positions
        self.commission_rate = commission_rate
        self.slippage_pct = slippage_pct
        if execution_model is None:
            execution_model = ExecutionModel(ExecutionConfig(
                timing=ExecutionTiming(execution_timing),
                slippage_pct=slippage_pct,
                commission_rate=commission_rate,
                max_participation_rate=max_participation_rate,
            ))
        self.execution_model = execution_model
        self.max_symbol_weight = max_symbol_weight
        self.max_daily_new_positions = max_daily_new_positions
        self._rejections: list[dict] = []
        self.max_gross_exposure = max_gross_exposure
        self.lot_size = lot_size
        self.max_participation_rate = max_participation_rate
        self.sizing_mode = sizing_mode
        self.risk_per_trade = risk_per_trade
        self.risk_atr_mult = risk_atr_mult
        # Resolve max_position_pct sentinel: None → RiskLimits default (0.30),
        # 0/0.0 → disabled. Imported lazily to avoid top-level cycle.
        if max_position_pct is None:
            from utils.risk import RiskLimits
            max_position_pct = RiskLimits().max_position_pct
        self.max_position_pct = max_position_pct
        self.max_sector_weight = max_sector_weight
        # Without a sector_map, the gate would lump every symbol into
        # "Unknown" — silently turning the per-sector cap into a global
        # exposure cap. Fall back to DEFAULT_SECTORS and warn explicitly so
        # the user notices that the gate is using a generic mapping.
        if max_sector_weight > 0 and not sector_map:
            import logging as _logging
            from utils.sectors import DEFAULT_SECTORS
            _logging.getLogger(__name__).warning(
                "max_sector_weight=%.2f 已启用但未传 sector_map — "
                "回退到 utils.sectors.DEFAULT_SECTORS。未覆盖的标的将归为 'Unknown' "
                "并被合并为同一行业, 这可能不是你想要的。",
                max_sector_weight,
            )
            self.sector_map = dict(DEFAULT_SECTORS)
        else:
            self.sector_map = sector_map or {}
        self.cooldown_after_stop_days = cooldown_after_stop_days
        # Short proceeds inflate ``cash``; this ratio (Reg-T style) is the
        # maintenance margin locked per dollar of short notional, deducted
        # from available_cash so it cannot fund new long entries.
        self.short_margin_ratio = short_margin_ratio
        self._stop_dates: dict[str, Optional[pd.Timestamp]] = {}

    def _available_cash(self, cash: float, leg_state: dict) -> float:
        """Cash available for opening new LONG positions across the portfolio.

        Deducts margin locked by current short legs. Short proceeds were
        booked into ``cash`` at fill time but cannot be re-used to fund longs.
        """
        short_margin = 0.0
        for s in leg_state.values():
            pos = s.get("position", 0)
            if pos < 0:
                short_margin += abs(pos) * s.get("last_price", 0) * self.short_margin_ratio
        return cash - short_margin

    def _reject(self, date_idx, symbol: str, reason: str, detail: str = "") -> None:
        """Record a rejection and return — used in risk gate chains."""
        self._rejections.append({
            "date": date_idx, "symbol": symbol,
            "reason": reason, "detail": detail,
        })

    def _calc_risk_budget_qty(self, capital: float, price: float, atr: float) -> int:
        """Return position size such that stop_distance loss = risk_per_trade % of capital."""
        if price <= 0:
            return 0
        if atr is None or np.isnan(atr) or atr <= 0:
            atr = price * 0.02
        risk_dollar = capital * self.risk_per_trade
        stop_distance = atr * self.risk_atr_mult
        if stop_distance <= 0:
            return 0
        shares = int(risk_dollar / stop_distance)
        max_shares = int(capital / (price * (1 + self.slippage_pct) * (1 + self.commission_rate)))
        return max(0, min(shares, max_shares))

    def run(
        self, start: str = "2018-01-01", end: Optional[str] = None
    ) -> "PortfolioResult":
        if end is None:
            end = date.today().isoformat()

        # Warn if max_position_pct × num_legs leaves cash idle. Below 4 legs
        # at the 0.30 default, the portfolio cannot reach full deployment —
        # caller may want to add legs or raise the cap.
        if self.max_position_pct and len(self.legs) > 0:
            total_deployable = self.max_position_pct * len(self.legs)
            if total_deployable < 1.0:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "max_position_pct=%.2f × num_legs=%d = %.0f%% < 100%% — "
                    "组合无法满仓 (剩 %.0f%% 现金闲置)。考虑增加标的或调高 max_position_pct, "
                    "或显式传 max_position_pct=0 关闭该上限。",
                    self.max_position_pct, len(self.legs),
                    total_deployable * 100, (1 - total_deployable) * 100,
                )

        provider = DataProvider()

        # Fetch proxy data once if any leg is an ensemble
        proxy_df = None
        has_ensemble = any(leg.members for leg in self.legs)
        if has_ensemble:
            proxy_df = provider.get_daily("SPY", start=start, end=end)
            if proxy_df is None or proxy_df.empty:
                print("  ! SPY 无数据，集成代理不可用")
                proxy_df = None

        # Fetch all data & calculate signals upfront
        leg_data = []  # list of (leg, strategy, df_sig)
        for leg in self.legs:
            df = provider.get_daily(leg.symbol, start=start, end=end)
            if df is None or df.empty:
                print(f"  ! {leg.symbol} 无数据，跳过")
                continue
            df = df.dropna(subset=["Open", "High", "Low", "Close"])
            strategy = leg.create_strategy(proxy_df=proxy_df)
            df_sig = strategy.calculate_indicators(df)
            display_name = "ensemble" if leg.members else leg.strategy_name
            leg_data.append((leg, strategy, df_sig))
            print(f"  {leg.symbol:<8s}  {display_name:<20s}  {len(df_sig)} 根K线")

        if not leg_data:
            raise RuntimeError("所有标的均无数据")

        # Build common timeline
        all_dates = set()
        for _, _, df_sig in leg_data:
            all_dates.update(df_sig.index)
        timeline = sorted(all_dates)

        # Per-leg state
        leg_state: dict = {}
        for i, (leg, strategy, df_sig) in enumerate(leg_data):
            leg_state[i] = LegState(leg=leg, strategy=strategy, df=df_sig)

        # Portfolio-level mutable state — replaces a handful of local
        # variables that previously had to be threaded through every helper.
        state = PortfolioState(cash=self.initial_capital, leg_state=leg_state)
        # Back-compat shim for any code still reaching for _leg_state_ref
        # (e.g. private callers); avoid breaking that surface accidentally.
        self._leg_state_ref = state.leg_state

        for date_idx in timeline:
            # Reset daily counter on new date
            if state.prev_date is not None and date_idx.date() != state.prev_date.date():
                state.daily_new_count = 0
            state.prev_date = date_idx

            for i, st in state.leg_state.items():
                df_sig = st["df"]
                if date_idx not in df_sig.index:
                    # Symbol has no bar for this date (cross-market holiday gap).
                    # Position value unchanged on non-trading days.
                    continue

                idx_pos = df_sig.index.get_loc(date_idx)
                if idx_pos < st["strategy"].min_bars:
                    continue

                row = df_sig.iloc[idx_pos]
                price = float(row["Close"])
                atr = float(row["ATR"]) if "ATR" in row.index else 0
                st["last_price"] = price

                # --- pending order fill ---
                if st.get("pending_order") is not None:
                    handled, skip_leg = self._handle_pending_order(
                        state, st, row, date_idx, idx_pos, i,
                    )
                    if handled or skip_leg:
                        continue

                # --- exit / entry signal ---
                if st["position"] > 0:
                    self._check_leg_exit(st, df_sig, idx_pos, price)
                elif st["position"] < 0:
                    self._check_leg_cover(st, df_sig, idx_pos, price)
                else:
                    self._check_leg_entry(
                        state, st, df_sig, idx_pos, row, price, atr, date_idx,
                    )

            total_equity = state.cash + sum(
                s["position"] * s.get("last_price", 0) for s in state.leg_state.values()
            )
            state.equity_history.append((date_idx, total_equity))

        # Close all open positions at end of period
        last_date = timeline[-1] if timeline else pd.Timestamp(end)
        for i, st in state.leg_state.items():
            if st["position"] > 0 and "last_price" in st:
                self._close_trade(state, st, last_date, st["last_price"], "end_of_period", i)
            elif st["position"] < 0 and "last_price" in st:
                self._close_short_trade(state, st, last_date, st["last_price"], "end_of_period", i)

        final_equity = state.cash
        if state.equity_history:
            # Keep result.final_equity and the last equity-curve point on the
            # same liquidation basis used by performance metrics.
            state.equity_history[-1] = (last_date, final_equity)

        return PortfolioResult(
            equity_history=state.equity_history,
            initial_capital=self.initial_capital,
            final_equity=final_equity,
            legs=self.legs,
            trades=state.trades,
            rejections=self._rejections,
        )

    # -- run() sub-methods ----------------------------------------------------

    def _handle_pending_order(
        self, state: PortfolioState, st, row, date_idx, idx_pos, leg_i,
    ):
        """Execute a pending order fill for one leg.

        Returns ``(handled, skip)`` — when handled, the caller should skip
        signal evaluation for this leg on this bar. Cash mutations happen
        inside the apply_* helpers via ``state.cash``.
        """
        pending = st["pending_order"]
        # Only restrict sell fills when closing existing longs (position > 0).
        # Short-entry sells (position <= 0) should not be capped.
        avail = st["position"] if (pending.side == OrderSide.SELL and st["position"] > 0) else None
        fill = self.execution_model.execute_bar(
            pending, row, date_idx, idx_pos, available_qty=avail,
        )
        if pending.side == OrderSide.SELL:
            if st["position"] > 0:
                # Close / reduce long
                if fill.status in (OrderStatus.FILLED, OrderStatus.PARTIAL):
                    self._apply_close_fill(state, st, fill, pending.reason, leg_i)
                    pending.quantity -= fill.filled_qty
                if fill.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED) or st["position"] == 0:
                    st["pending_order"] = None
                return True, True
            else:
                # Open / add to short
                if fill.status in (OrderStatus.FILLED, OrderStatus.PARTIAL):
                    self._apply_short_entry_fill(state, st, fill, leg_i)
                    pending.quantity -= fill.filled_qty
                if fill.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED) or pending.quantity <= 0:
                    st["pending_order"] = None
                return True, True

        if pending.side == OrderSide.BUY:
            if st["position"] < 0:
                # Cover short
                if fill.status in (OrderStatus.FILLED, OrderStatus.PARTIAL):
                    self._apply_cover_fill(state, st, fill, pending.reason, leg_i)
                    pending.quantity -= fill.filled_qty
                if fill.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED) or st["position"] >= 0:
                    st["pending_order"] = None
                return True, False
            else:
                # Open / add to long — must respect available_cash so short
                # margin already locked elsewhere in the portfolio is not
                # double-spent.
                if fill.status in (OrderStatus.FILLED, OrderStatus.PARTIAL):
                    spendable = self._available_cash(state.cash, state.leg_state)
                    if self._apply_open_fill(state, st, fill, leg_i, spendable):
                        pending.quantity -= fill.filled_qty
                    else:
                        self._reject(date_idx, st["leg"].symbol, "成交时现金不足",
                                     f"需${fill.gross_value + fill.commission:,.0f} > 可用${spendable:,.0f}")
                if fill.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED) or pending.quantity <= 0:
                    st["pending_order"] = None
                return True, False

        if fill.status in (OrderStatus.CANCELLED, OrderStatus.REJECTED):
            st["pending_order"] = None
            return True, False
        return False, False

    def _check_leg_exit(self, st, df_sig, idx_pos, price):
        """Check exit signal for a held LONG position."""
        if price > st["highest"]:
            st["highest"] = price
        exit_now, reason = st["strategy"].check_exit(
            df_sig, idx_pos,
            entry_price=st["entry_price"],
            highest_since_entry=st["highest"],
            position={"date": st["entry_date"], "direction": "LONG"},
        )
        if exit_now:
            st["pending_order"] = self.execution_model.make_plan(
                symbol=st["leg"].symbol,
                side=OrderSide.SELL,
                quantity=st["position"],
                created_index=idx_pos,
                reason=reason,
            )

    def _check_leg_cover(self, st, df_sig, idx_pos, price):
        """Check cover signal for an existing SHORT position.

        Cover is an exit — does not touch ``state.daily_new_count``.
        """
        if price < st["lowest"]:
            st["lowest"] = price
        exit_now, reason = st["strategy"].check_exit(
            df_sig, idx_pos,
            entry_price=st["entry_price"],
            highest_since_entry=st.get("highest", 0),
            lowest_since_entry=st["lowest"],
            position={"date": st["entry_date"], "direction": "SHORT"},
        )
        if exit_now:
            st["pending_order"] = self.execution_model.make_plan(
                symbol=st["leg"].symbol,
                side=OrderSide.BUY,
                quantity=abs(st["position"]),
                created_index=idx_pos,
                reason=reason,
            )

    def _check_leg_entry(
        self, state: PortfolioState, st, df_sig, idx_pos, row, price, atr, date_idx,
    ):
        """Check entry signal for a leg with no position.

        Mutates ``state.daily_new_count`` on successful order placement.
        Supports both long (signal=1 → BUY) and short (signal=-1 → SELL) entries.
        """
        sig = st["strategy"].entry_signal(df_sig, idx_pos)
        if (st["position"] != 0
                or st.get("pending_order") is not None
                or idx_pos >= len(df_sig) - 1
                or sig == 0):
            return

        direction = "LONG" if sig == 1 else ("SHORT" if sig == -1 else None)
        if direction is None:
            return
        if direction == "SHORT" and getattr(st["strategy"], "long_only", True):
            return

        sym = st["leg"].symbol

        # Count active positions (absolute — both long and short count)
        active_positions = sum(1 for s in state.leg_state.values() if s.get("position", 0) != 0)
        if active_positions >= self.max_positions:
            self._reject(date_idx, sym, "仓位上限",
                         f"活跃仓位{active_positions}≥{self.max_positions}")
            return

        current_equity = state.cash + sum(
            s["position"] * s.get("last_price", 0) for s in state.leg_state.values()
        )

        # Cooldown after stop
        if self.cooldown_after_stop_days > 0 and sym in self._stop_dates:
            last_stop = self._stop_dates[sym]
            if last_stop is not None:
                days_since = (date_idx - last_stop).days
                if days_since < self.cooldown_after_stop_days:
                    self._reject(date_idx, sym, "冷却期",
                                 f"距止损{days_since}天 (<{self.cooldown_after_stop_days})")
                    return

        # Daily new position cap
        if self.max_daily_new_positions > 0 and state.daily_new_count >= self.max_daily_new_positions:
            self._reject(date_idx, sym, "日开仓上限",
                         f"当日已开{state.daily_new_count}笔 (≥{self.max_daily_new_positions})")
            return

        # LONG entries must size against available_cash so locked short margin
        # is respected. SHORT entries can size against the full cash pool —
        # margin lockup is enforced on subsequent long entries.
        avail_cash = self._available_cash(state.cash, state.leg_state) if direction == "LONG" else state.cash
        alloc = self._allocate(st["leg"], avail_cash, len(state.leg_state), current_equity)
        if alloc <= 0:
            return

        if self.sizing_mode == "risk_budget":
            qty = self._calc_risk_budget_qty(alloc, price, atr)
        else:
            qty = st["strategy"].position_size(alloc, price, atr)

        # Single-symbol cap — parity with live RiskController.calc_position_size.
        # Applied as a hard ceiling on qty (not a reject) so signals on
        # subsequent bars don't churn rejection log entries.
        if self.max_position_pct and current_equity > 0:
            cap_qty = int(current_equity * self.max_position_pct / price)
            qty = min(qty, cap_qty)

        qty = self._apply_execution_constraints(qty, row, df_sig, idx_pos)
        if qty <= 0:
            return

        notional = price * qty

        if self.max_sector_weight > 0:
            sector = self.sector_map.get(sym, "Unknown")
            sector_exposure = sum(
                s["position"] * s.get("last_price", 0)
                for s in state.leg_state.values()
                if self.sector_map.get(s["leg"].symbol, "Unknown") == sector
            )
            if (sector_exposure + notional) / max(current_equity, 1) > self.max_sector_weight:
                self._reject(date_idx, sym, "行业权重",
                             f"{sector}敞口{(sector_exposure + notional) / max(current_equity, 1) * 100:.0f}% > {self.max_sector_weight * 100:.0f}%")
                return

        if self.max_symbol_weight > 0 and current_equity > 0:
            if notional / current_equity > self.max_symbol_weight:
                self._reject(date_idx, sym, "标的上限",
                             f"{sym}占比{notional / current_equity * 100:.0f}% > {self.max_symbol_weight * 100:.0f}%")
                return

        if self.max_gross_exposure > 0 and current_equity > 0:
            existing_exposure = sum(
                s["position"] * s.get("last_price", 0) for s in state.leg_state.values()
            )
            if (existing_exposure + notional) / current_equity > self.max_gross_exposure:
                self._reject(date_idx, sym, "总敞口上限",
                             f"总敞口{(existing_exposure + notional) / current_equity * 100:.0f}% > {self.max_gross_exposure * 100:.0f}%")
                return

        if direction == "LONG":
            actual_price = price * (1 + self.slippage_pct)
            cost = actual_price * qty * (1 + self.commission_rate)
            if cost <= avail_cash and qty > 0:
                st["pending_order"] = self.execution_model.make_plan(
                    symbol=sym, side=OrderSide.BUY, quantity=qty, created_index=idx_pos,
                )
                state.daily_new_count += 1
            elif qty > 0:
                self._reject(date_idx, sym, "可用现金不足",
                             f"需${cost:,.0f} > 可用${avail_cash:,.0f}")
        else:  # SHORT
            if qty > 0:
                st["pending_order"] = self.execution_model.make_plan(
                    symbol=sym, side=OrderSide.SELL, quantity=qty, created_index=idx_pos,
                    reason="short_entry",
                )
                state.daily_new_count += 1

    # -- helpers --------------------------------------------------------------

    def _close_trade(
        self, state: PortfolioState, st, date_idx: pd.Timestamp, price: float,
        reason: str, leg_index: int,
    ) -> None:
        """Close a long position — unified path for signal exits and end-of-period close.

        Applies slippage and commission consistently. Mutates ``state.cash``
        and ``state.trades`` / ``state.open_trade_idx`` in place.
        """
        qty = st["position"]
        exit_price = price * (1 - self.slippage_pct)
        exit_proceeds = exit_price * qty * (1 - self.commission_rate)
        entry_cost = st["capital_allocated"]
        state.cash += exit_proceeds

        if leg_index in state.open_trade_idx:
            t = state.trades[state.open_trade_idx[leg_index]]
            t.exit_time = date_idx
            t.exit_price = exit_price
            t.pnl = exit_proceeds - entry_cost
            t.pnl_pct = (t.pnl / entry_cost * 100) if entry_cost > 0 else 0
            t.reason = reason
            t.hold_days = (date_idx - t.entry_time).days
            del state.open_trade_idx[leg_index]

        st["position"] = 0
        st["capital_allocated"] = 0
        st["entry_price"] = 0
        st["entry_date"] = None
        st["highest"] = 0
        st["lowest"] = float('inf')
        # Track stop-loss exit for cooldown
        if reason.startswith("止损"):
            self._stop_dates[st["leg"].symbol] = date_idx

    def _close_short_trade(
        self, state: PortfolioState, st, date_idx: pd.Timestamp, price: float,
        reason: str, leg_index: int,
    ) -> None:
        """Close a short position — cover at market price."""
        qty = abs(st["position"])
        cover_price = price * (1 + self.slippage_pct)
        cost = cover_price * qty * (1 + self.commission_rate)
        state.cash -= cost

        if leg_index in state.open_trade_idx:
            t = state.trades[state.open_trade_idx[leg_index]]
            t.exit_time = date_idx
            t.exit_price = cover_price
            # Short PnL: proceeds from entry minus cost to cover
            entry_proceeds = t.entry_price * qty * (1 - self.commission_rate)
            t.pnl = entry_proceeds - cost
            t.pnl_pct = (t.pnl / (t.entry_price * qty) * 100) if t.entry_price > 0 else 0
            t.reason = reason
            t.hold_days = (date_idx - t.entry_time).days
            del state.open_trade_idx[leg_index]

        st["position"] = 0
        st["capital_allocated"] = 0
        st["entry_price"] = 0
        st["entry_date"] = None
        st["highest"] = 0
        st["lowest"] = float('inf')
        if reason.startswith("止损"):
            self._stop_dates[st["leg"].symbol] = date_idx

    def _apply_open_fill(
        self, state: PortfolioState, st, fill, leg_index: int, spendable_cash: float,
    ) -> bool:
        """Apply a backtest fill to open or add to a long leg.

        ``spendable_cash`` is the available_cash budget at the moment of fill,
        which is ≤ ``state.cash`` whenever any leg in the portfolio is short
        (margin is locked). Returns False on insufficient funds.
        """
        cost = fill.gross_value + fill.commission
        if cost > spendable_cash or fill.filled_qty <= 0:
            return False

        state.cash -= cost
        old_qty = st["position"]
        st["position"] += fill.filled_qty
        st["highest"] = max(st.get("highest", 0), fill.fill_price)
        st["lowest"] = min(st.get("lowest", float('inf')), fill.fill_price)
        st["capital_allocated"] += cost
        if old_qty > 0:
            st["entry_price"] = (
                st["entry_price"] * old_qty + fill.fill_price * fill.filled_qty
            ) / st["position"]
            if leg_index in state.open_trade_idx:
                t = state.trades[state.open_trade_idx[leg_index]]
                t.qty = st["position"]
                t.entry_price = st["entry_price"]
        else:
            st["entry_price"] = fill.fill_price
            st["entry_date"] = fill.date
            trade = PortfolioTrade(
                symbol=st["leg"].symbol,
                entry_time=fill.date,
                qty=fill.filled_qty,
                entry_price=fill.fill_price,
            )
            state.trades.append(trade)
            state.open_trade_idx[leg_index] = len(state.trades) - 1
        return True

    def _apply_short_entry_fill(
        self, state: PortfolioState, st, fill, leg_index: int,
    ) -> None:
        """SELL fill for opening or adding to a short position.

        Mutates ``state.cash`` (proceeds from short sale added) and
        ``state.trades`` / ``state.open_trade_idx``.
        """
        if fill.filled_qty <= 0:
            return
        old_qty = st["position"]
        st["position"] -= fill.filled_qty  # short: position goes negative
        st["highest"] = max(st.get("highest", 0), fill.fill_price)
        st["lowest"] = min(st.get("lowest", float('inf')), fill.fill_price)
        if old_qty < 0:
            # Adding to existing short — update avg entry price
            old_shares = abs(old_qty)
            st["entry_price"] = (
                st["entry_price"] * old_shares + fill.fill_price * fill.filled_qty
            ) / (old_shares + fill.filled_qty)
            if leg_index in state.open_trade_idx:
                t = state.trades[state.open_trade_idx[leg_index]]
                t.qty = abs(st["position"])
                t.entry_price = st["entry_price"]
        else:
            # New short position
            st["entry_price"] = fill.fill_price
            st["entry_date"] = fill.date
            trade = PortfolioTrade(
                symbol=st["leg"].symbol,
                entry_time=fill.date,
                qty=fill.filled_qty,
                entry_price=fill.fill_price,
                direction="SHORT",
            )
            state.trades.append(trade)
            state.open_trade_idx[leg_index] = len(state.trades) - 1

        # Short sale proceeds increase cash
        state.cash += fill.fill_price * fill.filled_qty * (1 - self.commission_rate)

    def _apply_cover_fill(
        self, state: PortfolioState, st, fill, reason: str, leg_index: int,
    ) -> None:
        """BUY fill for covering a short position."""
        qty = min(fill.filled_qty, abs(st["position"]))
        if qty <= 0:
            return
        cost = fill.fill_price * qty + fill.fill_price * qty * self.commission_rate
        state.cash -= cost
        st["position"] += qty  # negative towards zero

        if leg_index in state.open_trade_idx:
            t = state.trades[state.open_trade_idx[leg_index]]
            if st["position"] == 0:
                # Fully covered — record trade (PnL: entry - exit for shorts)
                t.exit_time = fill.date
                t.exit_price = fill.fill_price
                entry_proceeds = t.entry_price * qty * (1 - self.commission_rate)
                t.pnl = entry_proceeds - cost
                t.pnl_pct = (t.pnl / (t.entry_price * qty) * 100) if t.entry_price > 0 else 0
                t.reason = reason
                t.hold_days = (fill.date - t.entry_time).days
                del state.open_trade_idx[leg_index]
            else:
                # Partial cover — update remaining quantity
                t.qty = abs(st["position"])

        if st["position"] == 0:
            st["capital_allocated"] = 0
            st["entry_price"] = 0
            st["entry_date"] = None
            st["highest"] = 0
            st["lowest"] = float('inf')
        if reason.startswith("止损"):
            self._stop_dates[st["leg"].symbol] = fill.date

    def _apply_close_fill(
        self, state: PortfolioState, st, fill, reason: str, leg_index: int,
    ) -> None:
        """Apply a backtest fill to close or reduce a long leg."""
        qty = min(fill.filled_qty, st["position"])
        if qty <= 0:
            return
        avg_entry_cost = st["capital_allocated"] / st["position"] if st["position"] > 0 else 0
        entry_cost = avg_entry_cost * qty
        exit_proceeds = fill.fill_price * qty - fill.fill_price * qty * self.commission_rate
        state.cash += exit_proceeds
        st["position"] -= qty
        st["capital_allocated"] -= entry_cost

        if leg_index in state.open_trade_idx:
            t = state.trades[state.open_trade_idx[leg_index]]
            if st["position"] == 0:
                t.exit_time = fill.date
                t.exit_price = fill.fill_price
                t.pnl = exit_proceeds - entry_cost
                t.pnl_pct = (t.pnl / entry_cost * 100) if entry_cost > 0 else 0
                t.reason = reason
                t.hold_days = (fill.date - t.entry_time).days
                del state.open_trade_idx[leg_index]
            else:
                t.qty = st["position"]

        if st["position"] == 0:
            st["capital_allocated"] = 0
            st["entry_price"] = 0
            st["entry_date"] = None
            st["highest"] = 0
            st["lowest"] = float('inf')
        if reason.startswith("止损"):
            self._stop_dates[st["leg"].symbol] = fill.date

    def _apply_execution_constraints(
        self, qty: int, row: pd.Series, df_sig: pd.DataFrame, idx_pos: int,
    ) -> int:
        """Enforce lot size and participation rate on *qty*.  Return adjusted qty."""
        if self.lot_size > 0:
            qty = (qty // self.lot_size) * self.lot_size
        if self.max_participation_rate > 0 and qty > 0:
            try:
                vol = float(df_sig.iloc[idx_pos]["Volume"])
                if vol > 0:
                    max_qty = int(vol * self.max_participation_rate)
                    qty = min(qty, max_qty)
            except (KeyError, IndexError, TypeError):
                pass
        return qty

    def _allocate(self, leg: Leg, available_cash: float, num_legs: int,
                  current_equity: float = 0) -> float:
        """Determine how much cash to allocate to a new position."""
        if self.allocation == "equal":
            ref = current_equity if current_equity > 0 else self.initial_capital
            return min(available_cash, ref / max(num_legs, 1))
        elif self.allocation == "dynamic_equal":
            ref = current_equity if current_equity > 0 else self.initial_capital
            return min(available_cash, ref / max(num_legs, 1))
        elif self.allocation == "fraction":
            return available_cash * 0.25
        return available_cash


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class PortfolioResult:
    equity_history: List[Tuple[pd.Timestamp, float]]
    initial_capital: float
    final_equity: float
    legs: List[Leg]
    trades: List[PortfolioTrade] = field(default_factory=list)
    rejections: list = field(default_factory=list)

    @property
    def equity_curve(self) -> pd.Series:
        df = pd.DataFrame(self.equity_history, columns=["date", "equity"])
        df = df.drop_duplicates("date").set_index("date").sort_index()
        return df["equity"]

    @property
    def total_return_pct(self) -> float:
        return (self.final_equity / self.initial_capital - 1) * 100

    @property
    def cagr_pct(self) -> float:
        curve = self.equity_curve
        days = (curve.index[-1] - curve.index[0]).days
        if days <= 0:
            return 0
        return ((self.final_equity / self.initial_capital) ** (365.25 / days) - 1) * 100

    @property
    def sharpe_ratio(self) -> float:
        rets = self.equity_curve.pct_change(fill_method=None).dropna()
        if rets.std() == 0:
            return 0
        return np.sqrt(252) * rets.mean() / rets.std()

    @property
    def max_drawdown_pct(self) -> float:
        curve = self.equity_curve
        rolling_max = curve.expanding().max()
        dd = (curve - rolling_max) / rolling_max * 100
        return float(dd.min())

    # --- Trade statistics ---

    @property
    def closed_trades(self) -> List[PortfolioTrade]:
        return [t for t in self.trades if t.exit_price is not None]

    @property
    def total_trades(self) -> int:
        return len(self.closed_trades)

    @property
    def win_rate_pct(self) -> float:
        closed = self.closed_trades
        if not closed:
            return 0
        wins = sum(1 for t in closed if t.pnl is not None and t.pnl > 0)
        return wins / len(closed) * 100

    @property
    def profit_factor(self) -> float:
        gross_win = sum(t.pnl for t in self.closed_trades if t.pnl is not None and t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.closed_trades if t.pnl is not None and t.pnl < 0))
        return gross_win / gross_loss if gross_loss > 0 else float("inf")

    @property
    def avg_win(self) -> float:
        wins = [t.pnl for t in self.closed_trades if t.pnl is not None and t.pnl > 0]
        return sum(wins) / len(wins) if wins else 0

    @property
    def avg_loss(self) -> float:
        losses = [t.pnl for t in self.closed_trades if t.pnl is not None and t.pnl < 0]
        return sum(losses) / len(losses) if losses else 0

    @property
    def avg_hold_days(self) -> float:
        closed = [t for t in self.closed_trades if t.hold_days is not None]
        return sum(t.hold_days for t in closed) / len(closed) if closed else 0

    def summary(self):
        print(f"\n{'=' * 60}")
        print(f"  组合回测结果")
        print(f"{'=' * 60}")
        print(f"  标的数量:      {len(self.legs)}")
        print(f"  初始资金:      ${self.initial_capital:,.0f}")
        print(f"  最终权益:      ${self.final_equity:,.0f}")
        print(f"  总收益率:      {self.total_return_pct:+.2f}%")
        print(f"  年化收益率:    {self.cagr_pct:+.2f}%")
        print(f"  夏普比率:      {self.sharpe_ratio:.2f}")
        print(f"  最大回撤:      {self.max_drawdown_pct:.2f}%")
        print()

        # Trade statistics
        print(f"  --- 交易统计 ---")
        print(f"  总交易笔数:    {self.total_trades}")
        print(f"  胜率:          {self.win_rate_pct:.1f}%")
        print(f"  盈亏比:        {self.profit_factor:.2f}")
        print(f"  平均盈利:      ${self.avg_win:,.0f}")
        print(f"  平均亏损:      ${self.avg_loss:,.0f}")
        print(f"  平均持仓天数:  {self.avg_hold_days:.1f}")
        print()

        # Per-symbol breakdown
        if self.closed_trades:
            by_symbol: dict = {}
            for t in self.closed_trades:
                by_symbol.setdefault(t.symbol, []).append(t)
            print(f"  {'标的':<8s} {'笔数':>4s} {'胜率':>6s} {'总PnL':>10s} {'平均PnL':>10s}")
            print(f"  {'─' * 44}")
            for sym, sym_trades in sorted(by_symbol.items()):
                n = len(sym_trades)
                wr = sum(1 for t in sym_trades if t.pnl is not None and t.pnl > 0) / n * 100
                total_pnl = sum(t.pnl or 0 for t in sym_trades)
                avg_pnl = total_pnl / n
                print(f"  {sym:<8s} {n:>4d} {wr:>5.0f}% ${total_pnl:>9,.0f} ${avg_pnl:>9,.0f}")
            print()

        # Rejection log
        if self.rejections:
            print(f"  --- 风控拦截 ({len(self.rejections)} 次) ---")
            # Group by reason
            by_reason: dict = {}
            for r in self.rejections:
                by_reason.setdefault(r["reason"], []).append(r)
            for reason, items in sorted(by_reason.items()):
                syms = sorted(set(r.get("symbol", "") for r in items))
                print(f"  {reason}: {len(items)}次  {', '.join(syms)}")
            # Show last 5
            print(f"\n  最近拦截:")
            for r in self.rejections[-5:]:
                d = str(r["date"])[:10]
                sym = r.get("symbol", "")
                print(f"    {d} {sym:<6s} {r['reason']}: {r['detail']}")
            print()

    def plot(self, save_path: str = "charts/portfolio_result.png"):
        try:
            setup_chinese_font()
        except Exception:
            pass

        fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})

        curve = self.equity_curve
        ax1 = axes[0]
        ax1.plot(curve.index, curve, color="#2ca02c", linewidth=1.2, label="组合权益")
        ax1.axhline(y=self.initial_capital, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
        ax1.set_ylabel("Equity ($)")
        ax1.set_title("组合权益曲线", fontsize=13, fontweight="bold")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

        ax2 = axes[1]
        rolling_max = curve.expanding().max()
        drawdown = (curve - rolling_max) / rolling_max * 100
        ax2.fill_between(drawdown.index, drawdown, 0, color="#d62728", alpha=0.4)
        ax2.plot(drawdown.index, drawdown, color="#d62728", linewidth=0.6)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Date")
        ax2.set_title("组合回撤", fontsize=13, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"图表已保存: {save_path}")
        plt.close()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


DEFAULT_PORTFOLIO = [
    Leg("AAPL", "weekly_macd_kdj"),
    Leg("NVDA", "weekly_macd_kdj"),
    Leg("TSLA", "weekly_macd_kdj"),
    Leg("QQQ",  "weekly_macd_kdj"),
    Leg("SPY",  "turtle_trading",  {}),
    Leg("510300", "turtle_trading", {}),
]


def main():
    print("组合回测")
    print(f"{'=' * 60}")

    legs = DEFAULT_PORTFOLIO
    bt = PortfolioBacktest(
        legs=legs,
        initial_capital=100000,
        allocation="equal",
    )
    result = bt.run(start="2018-01-01")
    result.summary()
    result.plot()


if __name__ == "__main__":
    main()
