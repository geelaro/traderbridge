from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

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
from engine.execution import ExecutionConfig, ExecutionModel, ExecutionTiming
from utils.sizing import calc_risk_budget_qty

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_pct: float
    exit_reason: str
    direction: str = "LONG"
    holding_days: int = 0

    def __post_init__(self):
        if self.entry_date and self.exit_date:
            self.holding_days = (self.exit_date - self.entry_date).days


@dataclass
class BacktestResult:
    trades: List[Trade]
    equity_curve: pd.Series
    total_return_pct: float
    cagr_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    profit_factor: float
    avg_win_pct: float
    avg_loss_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    buy_hold_return_pct: float
    initial_capital: float
    final_equity: float
    rejections: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Simulated trading environment with slippage and commission."""

    def __init__(self, initial_capital=10000, commission_rate=0.0003, slippage_pct=0.0001,
                 sizing_mode="fixed_capital", risk_per_trade=0.005, risk_atr_mult=2.0,
                 cooldown_after_stop_days: int = 0, execution_model: ExecutionModel | None = None,
                 execution_timing: str = "next_open", short_margin_ratio: float = 1.5):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_pct = slippage_pct
        if execution_model is None:
            execution_model = ExecutionModel(ExecutionConfig(
                timing=ExecutionTiming(execution_timing),
                slippage_pct=slippage_pct,
                commission_rate=commission_rate,
            ))
        self.execution_model = execution_model
        self.sizing_mode = sizing_mode
        self.risk_per_trade = risk_per_trade
        self.risk_atr_mult = risk_atr_mult
        self.cooldown_after_stop_days = cooldown_after_stop_days
        # Reg-T style maintenance margin for short positions. Short proceeds
        # inflate ``cash``, but ``available_cash`` deducts the locked margin
        # so they cannot be re-used to open new long positions.
        self.short_margin_ratio = short_margin_ratio
        self.reset()

    def reset(self):
        self.cash = self.initial_capital
        self.position = 0  # positive = long, negative = short
        self.trades: List[Trade] = []
        self.equity_history: List[Tuple[pd.Timestamp, float]] = []
        self.current_entry: Optional[Dict] = None
        self._current_price = 0.0
        self._last_stop_date: Optional[pd.Timestamp] = None
        self.rejections: List[dict] = []

    @property
    def equity(self):
        return self.cash + self.position * self._current_price

    @property
    def available_cash(self) -> float:
        """Cash available for opening LONG positions.

        Short proceeds are added to ``self.cash`` but locked as maintenance
        margin (``short_margin_ratio`` × notional) and therefore cannot be
        spent on new long entries.
        """
        if self.position >= 0:
            return self.cash
        short_notional = abs(self.position) * self._current_price
        return self.cash - short_notional * self.short_margin_ratio

    @property
    def _direction(self) -> str:
        """Current position direction: 'LONG', 'SHORT', or 'FLAT'."""
        if self.position > 0:
            return "LONG"
        if self.position < 0:
            return "SHORT"
        return "FLAT"

    def _calc_risk_budget_qty(self, capital: float, price: float, atr: float) -> int:
        """Return position size such that stop_distance loss = risk_per_trade % of capital.

        Callers should pass ``self.available_cash`` (not ``self.cash``) for
        the LONG entry path so short-side margin lockup is respected.
        """
        raw_qty = calc_risk_budget_qty(capital, price, atr, self.risk_per_trade, self.risk_atr_mult)
        if price <= 0 or capital <= 0:
            return 0
        # Cost per share = price × (1+slippage) × (1+commission); the
        # additive (1+s+c) approximation is conservative but mis-models
        # the multiplicative pipeline at higher fee/slippage values.
        max_shares = int(capital / (price * (1 + self.slippage_pct) * (1 + self.commission_rate)))
        return max(0, min(raw_qty, max_shares))

    def buy(self, date, price, quantity, direction="LONG"):
        """Open or add to a position. *direction* = 'LONG' or 'SHORT'."""
        if quantity <= 0 or price <= 0:
            return False
        if direction == "LONG":
            actual_price = price * (1 + self.slippage_pct)
            total_cost = actual_price * quantity * (1 + self.commission_rate)
            # Cap by available_cash, not raw cash: short proceeds are locked
            # as margin and must not fund new long entries.
            spendable = self.available_cash if self.position < 0 else self.cash
            if total_cost > spendable:
                quantity = int(spendable / (actual_price * (1 + self.commission_rate)))
                if quantity <= 0:
                    return False
                total_cost = actual_price * quantity * (1 + self.commission_rate)
            self.cash -= total_cost
            self.position += quantity
            self.current_entry = {'date': date, 'price': actual_price, 'quantity': quantity, 'direction': 'LONG'}
            return True
        else:  # SHORT
            actual_price = price * (1 - self.slippage_pct)
            proceeds = actual_price * quantity * (1 - self.commission_rate)
            self.cash += proceeds
            self.position -= quantity
            self.current_entry = {'date': date, 'price': actual_price, 'quantity': quantity, 'direction': 'SHORT'}
            return True

    def _apply_buy_fill(self, fill) -> bool:
        """Execute a BUY fill — long entry OR short cover."""
        if fill.filled_qty <= 0 or fill.fill_price <= 0:
            return False
        is_cover = self.position < 0
        total_cost = fill.gross_value + fill.commission
        # Cover (BUY against short) consumes raw cash — short margin is being
        # released. Long entry/add must respect available_cash to keep short
        # proceeds from funding new longs.
        spendable = self.cash if is_cover else (
            self.available_cash if self.position < 0 else self.cash
        )
        if total_cost > spendable:
            affordable = int(spendable / (fill.fill_price * (1 + self.commission_rate)))
            if affordable <= 0:
                return False
            fill.filled_qty = min(fill.filled_qty, affordable)
            fill.gross_value = fill.fill_price * fill.filled_qty
            fill.commission = fill.gross_value * self.commission_rate
            total_cost = fill.gross_value + fill.commission
        self.cash -= total_cost
        old_qty = self.position
        old_entry = self.current_entry
        self.position += fill.filled_qty

        if is_cover:
            # Covering a short — record trade
            if old_entry and old_entry.get('direction') == 'SHORT':
                e = old_entry
                cover_qty = min(fill.filled_qty, e['quantity'])
                entry_proceeds = e['price'] * cover_qty * (1 - self.commission_rate)
                pnl = entry_proceeds - total_cost
                short_pnl_pct = (pnl / (e['price'] * cover_qty) * 100) if e['price'] > 0 else 0
                trade = Trade(
                    entry_date=e['date'], exit_date=fill.date,
                    entry_price=e['price'], exit_price=fill.fill_price,
                    quantity=cover_qty,
                    pnl=pnl,
                    pnl_pct=short_pnl_pct,
                    exit_reason=fill.reason if hasattr(fill, 'reason') else 'cover',
                    direction='SHORT')
                self.trades.append(trade)
                if self.position == 0:
                    self.current_entry = None
                else:
                    remain = e['quantity'] - cover_qty
                    if remain > 0:
                        self.current_entry = {**e, 'quantity': remain}
                    else:
                        self.current_entry = None
                return True
            # Fallback: no entry record — just update position
            if self.position == 0:
                self.current_entry = None
            return True

        # Long entry / add
        if old_entry and old_qty > 0 and old_entry.get('direction') != 'SHORT':
            avg_price = (
                old_entry['price'] * old_qty + fill.fill_price * fill.filled_qty
            ) / self.position
            self.current_entry = {
                'date': old_entry['date'],
                'price': avg_price,
                'quantity': self.position,
                'direction': 'LONG',
            }
        else:
            self.current_entry = {
                'date': fill.date,
                'price': fill.fill_price,
                'quantity': fill.filled_qty,
                'direction': 'LONG',
            }
        return True

    def sell(self, date, price, quantity=None, reason='signal'):
        """Exit a long position."""
        if quantity is None:
            quantity = self.position
        quantity = min(quantity, self.position)
        if quantity <= 0 or self.position <= 0:
            return None
        actual_price = price * (1 - self.slippage_pct)
        proceeds = actual_price * quantity * (1 - self.commission_rate)
        self.cash += proceeds
        self.position -= quantity

        trade = None
        if self.current_entry and self.current_entry.get('quantity', 0) > 0:
            e = self.current_entry
            entry_p = e['price']
            trade = Trade(
                entry_date=e['date'], exit_date=date,
                entry_price=entry_p, exit_price=actual_price,
                quantity=quantity,
                pnl=(actual_price - entry_p) * quantity,
                pnl_pct=(actual_price / entry_p - 1) * 100,
                exit_reason=reason,
                direction=e.get('direction', 'LONG'))
            self.trades.append(trade)
            if self.position == 0:
                self.current_entry = None
            else:
                self.current_entry = {**e, 'quantity': e['quantity'] - quantity}
            if trade and trade.exit_reason.startswith("止损"):
                self._last_stop_date = date
        elif self.position >= 0 and self.current_entry:
            trade = Trade(
                entry_date=date, exit_date=date,
                entry_price=actual_price, exit_price=actual_price,
                quantity=quantity, pnl=0.0, pnl_pct=0.0,
                exit_reason=f"{reason}(状态异常)",
                direction='LONG')
            self.trades.append(trade)
            if self.position == 0:
                self.current_entry = None
        return trade

    def _apply_sell_fill(self, fill, reason='signal'):
        """Execute a SELL fill — long exit OR short entry."""
        if fill.filled_qty <= 0:
            return None
        is_short_entry = self.position <= 0

        if is_short_entry:
            # Short entry — receive cash, position goes negative
            proceeds = fill.fill_price * fill.filled_qty - fill.fill_price * fill.filled_qty * self.commission_rate
            self.cash += proceeds
            self.position -= fill.filled_qty
            old_entry = self.current_entry
            if old_entry and old_entry.get('direction') == 'SHORT' and old_entry.get('quantity', 0) > 0:
                avg_price = (old_entry['price'] * old_entry['quantity'] + fill.fill_price * fill.filled_qty) / (old_entry['quantity'] + fill.filled_qty)
                self.current_entry = {
                    'date': old_entry['date'], 'price': avg_price,
                    'quantity': old_entry['quantity'] + fill.filled_qty, 'direction': 'SHORT',
                }
            else:
                self.current_entry = {
                    'date': fill.date, 'price': fill.fill_price,
                    'quantity': fill.filled_qty, 'direction': 'SHORT',
                }
            return None  # No trade record — trade is recorded on cover

        # Long exit
        quantity = min(fill.filled_qty, self.position)
        if quantity <= 0:
            return None
        proceeds = fill.fill_price * quantity - fill.fill_price * quantity * self.commission_rate
        self.cash += proceeds
        self.position -= quantity

        trade = None
        if self.current_entry and self.current_entry.get('quantity', 0) > 0:
            e = self.current_entry
            trade = Trade(
                entry_date=e['date'], exit_date=fill.date,
                entry_price=e['price'], exit_price=fill.fill_price,
                quantity=quantity,
                pnl=(fill.fill_price - e['price']) * quantity,
                pnl_pct=(fill.fill_price / e['price'] - 1) * 100,
                exit_reason=reason,
                direction=e.get('direction', 'LONG'))
            self.trades.append(trade)
            if self.position == 0:
                self.current_entry = None
            else:
                self.current_entry = {**e, 'quantity': e['quantity'] - quantity}
            if trade and trade.exit_reason.startswith("止损"):
                self._last_stop_date = fill.date
        return trade

    def update(self, date, price):
        self._current_price = price
        self.equity_history.append((date, self.equity))

    def get_result(self, benchmark_returns=None):
        if not self.equity_history:
            raise ValueError("No equity history recorded")
        eq_df = pd.DataFrame(self.equity_history, columns=['date', 'equity']).set_index('date')
        curve = eq_df['equity']
        rets = curve.pct_change(fill_method=None).dropna()

        final = curve.iloc[-1]
        total_ret = (final / self.initial_capital - 1) * 100
        days = (curve.index[-1] - curve.index[0]).days
        cagr = ((final / self.initial_capital) ** (365.25 / days) - 1) * 100 if days > 0 else 0
        sharpe = np.sqrt(252) * rets.mean() / rets.std() if rets.std() > 0 else 0

        rolling_max = curve.expanding().max()
        dd = (curve - rolling_max) / rolling_max * 100
        max_dd = dd.min()

        trades = self.trades
        if trades:
            wins = [t for t in trades if t.pnl > 0]
            losses = [t for t in trades if t.pnl <= 0]
            win_rate = len(wins) / len(trades) * 100
            avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
            avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0
            total_wins = sum(t.pnl for t in wins)
            total_losses = abs(sum(t.pnl for t in losses))
            pf = total_wins / total_losses if total_losses > 0 else float('inf')
        else:
            win_rate = avg_win = avg_loss = pf = 0

        bh_ret = ((1 + benchmark_returns).prod() - 1) * 100 if benchmark_returns is not None else 0

        return BacktestResult(
            trades=trades, equity_curve=curve,
            total_return_pct=total_ret, cagr_pct=cagr, sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd, win_rate_pct=win_rate, profit_factor=pf,
            avg_win_pct=avg_win, avg_loss_pct=avg_loss,
            total_trades=len(trades),
            winning_trades=len([t for t in trades if t.pnl > 0]),
            losing_trades=len([t for t in trades if t.pnl <= 0]),
            buy_hold_return_pct=bh_ret,
            initial_capital=self.initial_capital, final_equity=final,
            rejections=self.rejections)

    def run(self, strategy, df, close_out: bool = True) -> pd.Series:
        """Execute backtest loop over *df* using *strategy* signals.

        Signals are evaluated on bar close and executed at the next bar's open.

        Signal semantics
        ----------------
          1  = long entry
         -1  = short entry
          0  = neutral (hold / wait)

        Returns
        -------
        benchmark_returns : pd.Series
            Buy-and-hold returns of the close price over the backtest period.
        """
        self._highest = 0.0
        self._lowest = float('inf')
        self._pending_order = None

        for i in range(strategy.min_bars, len(df)):
            date_idx = df.index[i]
            close_price = float(df["Close"].iloc[i])

            if self._pending_order is not None:
                self._process_pending_order(df, i, date_idx, close_price)
                self.update(date_idx, close_price)
                continue

            if self.position > 0 and self.current_entry:
                self._check_exit_signal(strategy, df, i, close_price)

            elif self.position < 0 and self.current_entry:
                self._check_cover_signal(strategy, df, i, close_price)

            elif self.position == 0 and i < len(df) - 1:
                sig = strategy.entry_signal(df, i)
                if isinstance(sig, bool):  # old-style: True means long entry
                    if sig:
                        self._check_entry_signal(strategy, df, i, close_price, date_idx, "LONG")
                elif sig == 1:
                    self._check_entry_signal(strategy, df, i, close_price, date_idx, "LONG")
                elif sig == -1 and not getattr(strategy, 'long_only', True):
                    self._check_entry_signal(strategy, df, i, close_price, date_idx, "SHORT")

            self.update(date_idx, close_price)

        if close_out and self.position != 0:
            self._close_out(df, close_out)

        return df["Close"].pct_change(fill_method=None).dropna()

    # -- close-out -----------------------------------------------------------

    def _close_out(self, df, close_out):
        last_price = float(df["Close"].iloc[-1])
        last_date = df.index[-1]
        if self.position > 0:
            self.sell(last_date, last_price, reason="回测结束")
        elif self.position < 0:
            qty = abs(self.position)
            actual_price = last_price * (1 + self.slippage_pct)
            cost = actual_price * qty * (1 + self.commission_rate)
            if self.current_entry and self.current_entry.get('direction') == 'SHORT':
                e = self.current_entry
                entry_proceeds = e['price'] * qty * (1 - self.commission_rate)
                pnl = entry_proceeds - cost
                short_pnl_pct = (pnl / (e['price'] * qty) * 100) if e['price'] > 0 else 0
                trade = Trade(
                    entry_date=e['date'], exit_date=last_date,
                    entry_price=e['price'], exit_price=actual_price,
                    quantity=qty,
                    pnl=pnl,
                    pnl_pct=short_pnl_pct,
                    exit_reason="回测结束",
                    direction='SHORT')
                self.trades.append(trade)
            self.cash -= cost
            self.position = 0
            self.current_entry = None
        self._current_price = last_price
        self.equity_history[-1] = (last_date, self.equity)

    # -- loop sub-methods ----------------------------------------------------

    def _process_pending_order(self, df, i, date_idx, close_price):
        """Execute a pending order against bar *i*."""
        fill = self.execution_model.execute_bar(
            self._pending_order,
            df.iloc[i],
            date_idx,
            i,
            available_qty=abs(self.position) if self._pending_order.side == OrderSide.SELL and self.position > 0 else (abs(self.position) if self._pending_order.side == OrderSide.BUY and self.position < 0 else None),
        )
        if self._pending_order.side == OrderSide.BUY:
            if fill.status in (OrderStatus.FILLED, OrderStatus.PARTIAL):
                if self._apply_buy_fill(fill):
                    if self.position > 0:
                        self._highest = fill.fill_price
                        self._lowest = fill.fill_price
                    self._pending_order.quantity -= fill.filled_qty
            if fill.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED) or self._pending_order.quantity <= 0:
                self._pending_order = None
        elif self._pending_order.side == OrderSide.SELL:
            if fill.status in (OrderStatus.FILLED, OrderStatus.PARTIAL):
                self._apply_sell_fill(fill, reason=self._pending_order.reason)
                if self.position < 0:
                    self._lowest = fill.fill_price
                    self._highest = fill.fill_price
            if (fill.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED)
                    or self.position == 0):
                self._pending_order = None

    def _check_exit_signal(self, strategy, df, i, close_price):
        """Check for exit signal on an existing LONG position."""
        if close_price > self._highest:
            self._highest = close_price
        exit_now, reason = strategy.check_exit(
            df, i,
            entry_price=self.current_entry["price"],
            highest_since_entry=self._highest,
            position=self.current_entry,
        )
        if exit_now:
            self._pending_order = self.execution_model.make_plan(
                symbol="",
                side=OrderSide.SELL,
                quantity=self.position,
                created_index=i,
                reason=reason,
            )

    def _check_cover_signal(self, strategy, df, i, close_price):
        """Check for cover signal on an existing SHORT position."""
        if close_price < self._lowest:
            self._lowest = close_price
        exit_now, reason = strategy.check_exit(
            df, i,
            entry_price=self.current_entry["price"],
            highest_since_entry=self._highest,
            lowest_since_entry=self._lowest,
            position=self.current_entry,
        )
        if exit_now:
            self._pending_order = self.execution_model.make_plan(
                symbol="",
                side=OrderSide.BUY,
                quantity=abs(self.position),
                created_index=i,
                reason=reason,
            )

    def _check_entry_signal(self, strategy, df, i, close_price, date_idx, direction):
        """Check for entry signal — sizing, cooldown gate, create order."""
        if not self._apply_stop_cooldown(date_idx):
            return
        atr = float(df["ATR"].iloc[i]) if "ATR" in df.columns else 0.0
        # LONG entry must use available_cash (short proceeds locked as margin).
        # SHORT entry sizing uses raw cash as the risk-budget reference — the
        # actual proceeds inflate cash but margin is enforced via available_cash
        # on subsequent long entries.
        capital_ref = self.available_cash if direction == "LONG" else self.cash
        if self.sizing_mode == "risk_budget":
            qty = self._calc_risk_budget_qty(capital_ref, close_price, atr)
        else:
            qty = strategy.position_size(capital_ref, close_price, atr)
        if qty <= 0:
            return
        if direction == "LONG":
            self._pending_order = self.execution_model.make_plan(
                symbol="", side=OrderSide.BUY, quantity=qty, created_index=i,
            )
        else:  # SHORT
            self._pending_order = self.execution_model.make_plan(
                symbol="", side=OrderSide.SELL, quantity=qty, created_index=i,
                reason="short_entry",
            )

    def _apply_stop_cooldown(self, date_idx) -> bool:
        """Return True if entry is allowed (not in stop-loss cooldown)."""
        if self.cooldown_after_stop_days <= 0 or self._last_stop_date is None:
            return True
        days_since_stop = (date_idx - self._last_stop_date).days
        if days_since_stop < self.cooldown_after_stop_days:
            self.rejections.append({
                "date": date_idx, "reason": "冷却期",
                "detail": f"距止损仅{days_since_stop}天 (<{self.cooldown_after_stop_days})",
            })
            return False
        return True


# ---------------------------------------------------------------------------
# Backtest runner & visualization
# ---------------------------------------------------------------------------



def run_backtest(symbol="QQQ", start="2020-01-01", end=None,
                 initial_capital=10000, strategy_cls=None,
                 commission_rate=0.0003, slippage_pct=0.0001,
                 sizing_mode="fixed_capital", risk_per_trade=0.005,
                 risk_atr_mult=2.0, cooldown_after_stop_days: int = 0,
                 **strategy_params):
    """Run a full backtest and return results + dataframe.

    Args:
        strategy_cls: Strategy class (default: MACDKDJStrategy, freq="W").
                      Use TrendFollower for Chandelier exit strategy.
        commission_rate: Trading commission rate (default 0.0003 = 3bp).
        slippage_pct: Slippage percentage (default 0.0001 = 1bp).
        sizing_mode: "fixed_capital" | "risk_budget".
        risk_per_trade: Fraction of capital at risk per trade (risk_budget mode).
        risk_atr_mult: Stop distance = ATR × this (risk_budget mode).
        cooldown_after_stop_days: Block re-entry for N days after a stop-loss exit.
    """
    if strategy_cls is None:
        from strategy.macd_kdj import MACDKDJStrategy
        strategy_cls = MACDKDJStrategy
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    print(f"获取 {symbol} 数据 ({start} ~ {end}) ...")
    provider = DataProvider()
    df = provider.get_daily(symbol, start=start, end=end)
    if df is None or df.empty:
        raise RuntimeError(f"无法获取 {symbol} 数据 ({start} ~ {end})，数据管线全部失败")
    print(f"  OK: {len(df)} 根K线")

    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])

    strategy = strategy_cls(**strategy_params)
    df = strategy.calculate_indicators(df)
    engine = BacktestEngine(initial_capital=initial_capital,
                            commission_rate=commission_rate,
                            slippage_pct=slippage_pct,
                            sizing_mode=sizing_mode,
                            risk_per_trade=risk_per_trade,
                            risk_atr_mult=risk_atr_mult,
                            cooldown_after_stop_days=cooldown_after_stop_days)
    benchmark_rets = engine.run(strategy, df)
    result = engine.get_result(benchmark_rets)
    return result, df


def print_result(result):
    """Pretty-print backtest results."""
    print("\n" + "=" * 60)
    print("  回测结果")
    print("=" * 60)
    print(f"  初始资金:      ${result.initial_capital:,.2f}")
    print(f"  最终权益:      ${result.final_equity:,.2f}")
    print(f"  总收益率:      {result.total_return_pct:+.2f}%")
    print(f"  年化收益率:    {result.cagr_pct:+.2f}%")
    print(f"  买入持有:      {result.buy_hold_return_pct:+.2f}%")
    print(f"  夏普比率:      {result.sharpe_ratio:.2f}")
    print(f"  最大回撤:      {result.max_drawdown_pct:.2f}%")
    print(f"  交易次数:      {result.total_trades}")
    print(f"  胜率:          {result.win_rate_pct:.1f}% ({result.winning_trades}胜/{result.losing_trades}负)")
    print(f"  盈亏比:        {result.profit_factor:.2f}")
    print(f"  平均盈利:      {result.avg_win_pct:+.2f}%")
    print(f"  平均亏损:      {result.avg_loss_pct:+.2f}%")

    alpha = result.total_return_pct - result.buy_hold_return_pct
    print(f"\n  超额收益:      {alpha:+.2f}%")

    if result.rejections:
        print(f"\n  风控拦截:      {len(result.rejections)} 次")
        for r in result.rejections[-5:]:
            d = str(r["date"])[:10]
            print(f"    {d}  {r['reason']}: {r['detail']}")

    print("=" * 60)

    if result.trades:
        print("\n  最近 10 笔交易:")
        print(f"  {'入场':<12} {'出场':<12} {'持仓天':<8} {'盈亏%':<10} {'原因':<10}")
        print("  " + "-" * 52)
        for t in result.trades[-10:]:
            print(f"  {t.entry_date.strftime('%Y-%m-%d'):<12} "
                  f"{t.exit_date.strftime('%Y-%m-%d'):<12} "
                  f"{t.holding_days:<8} "
                  f"{t.pnl_pct:+.2f}%{'':<4} "
                  f"{t.exit_reason:<10}")


def plot_result(result, df, symbol="QQQ", save_path=None, show=True):
    """Plot backtest results: price + trades, equity curve, drawdown."""
    try:
        setup_chinese_font()
    except Exception:
        pass

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1.5, 1]})

    # --- Panel 1: Price + trades ---
    ax1 = axes[0]
    price_data = df['Close']
    # Align to backtest period
    if not result.equity_curve.empty:
        price_data = price_data[price_data.index >= result.equity_curve.index[0]]
    ax1.plot(price_data.index, price_data, color='#1f77b4', linewidth=0.8, label=f'{symbol} Close')

    # Mark buy/sell points
    for t in result.trades:
        if t.pnl > 0:
            color = 'green'
            marker = '^'
        else:
            color = 'red'
            marker = 'v'
        ax1.scatter(t.entry_date, t.entry_price, c=color, marker=marker, s=60, zorder=5, alpha=0.8)
        ax1.scatter(t.exit_date, t.exit_price, c=color, marker='o', s=40, zorder=5, alpha=0.6)

    # Add dummy points for legend
    ax1.scatter([], [], c='green', marker='^', s=60, label='入场 (盈)')
    ax1.scatter([], [], c='red', marker='v', s=60, label='入场 (亏)')
    ax1.scatter([], [], c='green', marker='o', s=40, label='出场 (盈)')
    ax1.scatter([], [], c='red', marker='o', s=40, label='出场 (亏)')

    ax1.set_ylabel('Price ($)')
    ax1.set_title(f'{symbol} 策略回测 — 交易信号', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Equity curve ---
    ax2 = axes[1]
    if not result.equity_curve.empty:
        eq = result.equity_curve
        ax2.plot(eq.index, eq, color='#2ca02c', linewidth=1.2, label='策略权益')

        # Buy & hold benchmark
        if result.buy_hold_return_pct != 0:
            bh_start = result.initial_capital
            bh_curve = bh_start * (price_data / price_data.iloc[0])
            ax2.plot(bh_curve.index, bh_curve, color='#d62728', linewidth=0.8,
                     linestyle='--', alpha=0.7, label='买入持有')

        ax2.axhline(y=result.initial_capital, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
        ax2.set_ylabel('Equity ($)')
        ax2.set_title('权益曲线', fontsize=13, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    # --- Panel 3: Drawdown ---
    ax3 = axes[2]
    if not result.equity_curve.empty:
        rolling_max = result.equity_curve.expanding().max()
        drawdown = (result.equity_curve - rolling_max) / rolling_max * 100
        ax3.fill_between(drawdown.index, drawdown, 0, color='#d62728', alpha=0.4, label='回撤')
        ax3.plot(drawdown.index, drawdown, color='#d62728', linewidth=0.6)
        ax3.set_ylabel('Drawdown (%)')
        ax3.set_xlabel('Date')
        ax3.set_title('回撤', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n图表已保存至: {save_path}")
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    result, df = run_backtest(symbol="QQQ", start="2020-01-01")
    print_result(result)
    plot_result(result, df, symbol="QQQ")
