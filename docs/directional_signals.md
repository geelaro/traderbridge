# Four-Way Research Signals

The research scanner exposes four explicit actions in addition to the legacy
`signal` integer. These actions describe a hypothetical strategy position,
not an instruction to trade a real account.

| Action | Meaning | Position before | Position after |
| --- | --- | --- | --- |
| `open_long` | Open long | Flat | Long |
| `open_short` | Open short | Flat | Short |
| `close_long` | Close long | Long | Flat |
| `close_short` | Cover short | Short | Flat |
| `hold` | No position change | Any | Unchanged |

## Strategy Semantics

- `Signal` remains unchanged for compatibility with existing backtests and
  numeric golden tests. It is not sufficient to identify a position action.
- `strategy.actions.replay_actions` replays a private strategy copy using
  close prices, `entry_signal`, `long_only`, `min_bars`, and `check_exit`.
- A held position exits according to the strategy's own exit logic, including
  trailing stops when `Signal == 0`. Repeated entry signals do not reset the
  entry date. An exit cannot reverse into a new position on the same bar.
- `BaseStrategy.check_exit` handles both long and short directions.
- Existing long-only strategies remain long-only. Turtle, Donchian and
  ensemble strategies retain their existing bidirectional definitions.
- Scanner results include `action`, `action_reason`, `position_before`,
  `position_after`, `long_only`, and `position_source="hypothetical_close"`.
- Research metadata is stored as `_action` and `_action_reason` inside the
  existing `signal_history.indicators` JSON. No schema migration is needed.
  Old history retains BUY/SELL labels because its position context is unknown.
- Data after the requested date and weekly bars labelled after that date are
  excluded from action replay. A weekly action is associated with its weekly
  bar date, not a newly invented daily event.

This is a close-price state reconstruction, not an execution backtest: it
does not model next-open fills, buying power, slippage, commissions, or borrow
fees. The backtest engines remain the execution/performance research tools.

## Display

Today's signals, history, the daily console summary, and signal notifications
use the explicit labels. The hypothetical-position table shows long/short
direction and signed PnL. Short PnL is `(entry - current) / entry`.
The unrealized-PnL summary respects the same direction. Existing portfolio
VaR/decomposition and Brinson views are long-only calculations; they exclude
hypothetical short legs until their weight math supports signed exposure.
The Chandelier reference stop is above the running low for shorts and below
the running high for longs. It is a reference, not a claim that every strategy
uses a Chandelier exit; actual research closes use `check_exit`.

## Directional Market Risk

`analysis.risk_monitor.compute_directional_risk` returns independent `long`
and `short` risk levels and reasons. It does not return trading actions.

- Opposing the MA200 trend beyond the existing 3% buffer raises direction risk.
- Overbought prices raise long chase risk; oversold prices raise short rebound
  risk. Existing RSI 70/80 thresholds are mirrored to 30/20 for shorts.
- Extreme extension (15%), high VIX (30), or high realized annualized
  volatility (40% when VIX is unavailable) can raise risk to red.
- Moderate volatility and rapid five-day moves raise caution on both sides.
- Insufficient data is unknown, never green. SPY/QQQ and Chinese index inputs
  are assessed separately; aggregate levels take the more cautious available
  constituent. Aggregate dates use the oldest constituent date.

These are transparent heuristic risk flags, not calibrated probabilities.
Both sides can be red at once. Existing sentiment temperatures and stress
lights are retained; neither those lights nor these flags trigger KillSwitch.

## Execution Boundary

`OrderManager` blocks new short entries with `SHORT_ENTRY_RESEARCH_ONLY`,
including raw SELL signals received while flat. Existing long-position closes
and short-position covers remain possible. No watchlist strategy is switched,
no real-account mode is enabled, and no orders are submitted by this feature.
Actual short execution accounting and broker readiness remain separate work.

## Verification

`tests/test_trade_actions.py` covers the action matrix, state replay, stops,
history, independent strategy parameters, short PnL and directional risk.
`tests/test_directional_dashboard.py` renders the relevant views offline using
Streamlit AppTest. Existing engine, order-manager and golden tests also apply.
