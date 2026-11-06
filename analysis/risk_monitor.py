"""Market risk light — downside risk plus upside overheat detection.

Designed as a glance-and-go indicator at the top of the dashboard. Keep the
logic intentionally simple: 3 inputs, explicit thresholds, human-readable
reasons. No optimisation, no fitting — these levels are conventions, not
backtested.

The light deliberately separates two ideas that are often conflated:
``overbought`` means the market has stretched upward, while ``top risk``
means several independent stretch signals agree.  Neither is a prediction
that the next bar must fall; top risk is a caution against chasing entries.

Thresholds
----------
GREEN  (low risk):
    SPY > MA200 by ≥ 3%  AND  SPY ADX > 25  AND  VIX < 18

RED    (danger):
    SPY < MA200 by ≥ 3%   OR   VIX > 30   OR   SPY 5-day return < -5%

YELLOW (caution):
    everything else — typically SPY near MA200 (±3%), or ADX 15-25 (trend
    weakening), or VIX 18-30 (elevated fear). A high top-risk score also
    is reported separately from the main green/yellow/red light.

Upside stretch signals:
    RSI(14) >= 70, SPY >= 10% above MA200, or SPY 20-day return >= 8%.
    At least two active signals are required for ``top_risk_level=high``.

Output
------
Each compute_risk_state() call returns a :class:`RiskState` with:
- ``level`` — RiskLevel enum (GREEN / YELLOW / RED)
- ``reasons`` — list of human-readable bullets explaining the call
- ``indicators`` — current numeric values (SPY/MA200/ADX/VIX/5d return)
- ``top_risk_level`` — ``normal`` / ``elevated`` / ``high`` / ``unknown``
- ``top_risk_reasons`` — the independent upside-stretch signals

Usage
-----
    from analysis.risk_monitor import compute_risk_state
    state = compute_risk_state(spy_df, vix_df)
    print(state.level.value, state.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import pandas as pd

from strategy.base import compute_adx

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


# Tuned for US equity markets 2018-2026 regime. Override via params if needed
# for other markets / regimes (e.g. crypto vol baseline is much higher).
_DEFAULTS = {
    "ma200_period": 200,
    "ma_buffer_pct": 0.03,        # ±3% band around MA200 → yellow zone
    "adx_strong": 25.0,
    "adx_weak": 15.0,
    "vix_low": 18.0,
    "vix_high": 30.0,
    "crash_5d_pct": -5.0,         # SPY drops > 5% in 5 days → instant red
    "rsi_period": 14,
    "rsi_overbought": 70.0,
    "rsi_extreme": 80.0,
    "extension_warn_pct": 10.0,   # SPY above MA200
    "extension_high_pct": 15.0,
    "rally_20d_warn_pct": 8.0,
    "rally_20d_high_pct": 12.0,
}


@dataclass
class RiskState:
    """Snapshot of the market risk light."""

    level: RiskLevel
    reasons: List[str] = field(default_factory=list)
    indicators: dict = field(default_factory=dict)
    top_risk_level: str = "unknown"
    top_risk_reasons: List[str] = field(default_factory=list)

    @property
    def emoji(self) -> str:
        return {RiskLevel.GREEN: "🟢", RiskLevel.YELLOW: "🟡", RiskLevel.RED: "🔴"}[self.level]

    def summary(self) -> str:
        """One-line summary plus reason bullets."""
        head = f"{self.emoji} {self.level.value.upper()}"
        if not self.reasons:
            return head
        body = "\n".join(f"  - {r}" for r in self.reasons)
        return f"{head}\n{body}"


def compute_risk_state(
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    params: Optional[dict] = None,
    realtime_vix: Optional[float] = None,
) -> RiskState:
    """Compute the current risk light from SPY and VIX OHLCV.

    Parameters
    ----------
    spy_df : pd.DataFrame
        SPY daily OHLCV. Must include at least ``ma200_period + 30`` bars
        for ADX warm-up.
    vix_df : pd.DataFrame
        VIX daily OHLCV (Close used). Trading-day index aligned with SPY
        is preferred; mismatched dates are resolved by forward-fill.
    params : dict, optional
        Override default thresholds.
    realtime_vix : float, optional
        Live VIX value (e.g. from :func:`data.realtime.get_realtime_vix`).
        When provided, overrides the EOD value from ``vix_df`` for the
        RED/GREEN threshold check.  Use during US trading hours so the
        risk light reflects intraday spikes rather than yesterday's close.
        The ``vix`` indicator in the returned RiskState is the realtime
        value when this override is active.

    Returns
    -------
    RiskState
    """
    p = {**_DEFAULTS, **(params or {})}

    if spy_df is None or spy_df.empty:
        return RiskState(level=RiskLevel.YELLOW,
                         reasons=["SPY 数据缺失, 默认黄"])
    if vix_df is None or vix_df.empty:
        return RiskState(level=RiskLevel.YELLOW,
                         reasons=["VIX 数据缺失, 默认黄"])

    spy = spy_df.copy()
    if len(spy) < p["ma200_period"] + 20:
        return RiskState(level=RiskLevel.YELLOW,
                         reasons=[f"SPY 数据不足 {p['ma200_period']+20} 根, 无法判定"])

    # Indicators
    spy["MA200"] = spy["Close"].rolling(p["ma200_period"]).mean()
    spy_adx = compute_adx(spy.copy(), 14)  # adds ADX column in place
    last = spy.iloc[-1]
    spy_close = float(last["Close"])
    ma200 = float(last["MA200"])
    adx = float(spy_adx["ADX"].iloc[-1]) if "ADX" in spy_adx.columns else 0.0
    ma_pct = (spy_close / ma200 - 1) * 100 if ma200 > 0 else 0.0

    # SPY 5-day return
    if len(spy) >= 6:
        ret_5d = (spy_close / float(spy["Close"].iloc[-6]) - 1) * 100
    else:
        ret_5d = 0.0

    # Upside stretch indicators. These are intentionally kept separate from
    # the downside risk-light reasons: a strong trend can be both healthy and
    # overbought, and the right response is usually to avoid chasing rather
    # than to liquidate.
    rsi_period = int(p["rsi_period"])
    close_series = spy["Close"].astype(float)
    delta = close_series.diff()
    gains = delta.clip(lower=0).rolling(rsi_period).mean()
    losses = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rs = gains / losses.replace(0, pd.NA)
    rsi_series = 100 - (100 / (1 + rs))
    rsi_series = rsi_series.where(~((losses == 0) & (gains > 0)), 100.0)
    rsi_series = rsi_series.where(~((losses == 0) & (gains == 0)), 50.0)
    rsi = float(rsi_series.iloc[-1]) if pd.notna(rsi_series.iloc[-1]) else 50.0
    ret_20d = (
        (spy_close / float(spy["Close"].iloc[-21]) - 1) * 100
        if len(spy) >= 22 else 0.0
    )

    top_risk_reasons: List[str] = []
    top_score = 0
    active_stretch = 0
    if rsi >= p["rsi_extreme"]:
        active_stretch += 1
        top_score += 2
        top_risk_reasons.append(
            f"RSI14={rsi:.1f} 极度超买 (≥ {p['rsi_extreme']:.0f})"
        )
    elif rsi >= p["rsi_overbought"]:
        active_stretch += 1
        top_score += 1
        top_risk_reasons.append(
            f"RSI14={rsi:.1f} 超买 (≥ {p['rsi_overbought']:.0f})"
        )

    if ma_pct >= p["extension_high_pct"]:
        active_stretch += 1
        top_score += 2
        top_risk_reasons.append(
            f"SPY 高于 MA200 {ma_pct:+.1f}% (极端乖离, ≥ {p['extension_high_pct']:.0f}%)"
        )
    elif ma_pct >= p["extension_warn_pct"]:
        active_stretch += 1
        top_score += 1
        top_risk_reasons.append(
            f"SPY 高于 MA200 {ma_pct:+.1f}% (乖离偏大, ≥ {p['extension_warn_pct']:.0f}%)"
        )

    if ret_20d >= p["rally_20d_high_pct"]:
        active_stretch += 1
        top_score += 2
        top_risk_reasons.append(
            f"SPY 20日涨幅 {ret_20d:+.1f}% (加速上涨, ≥ {p['rally_20d_high_pct']:.0f}%)"
        )
    elif ret_20d >= p["rally_20d_warn_pct"]:
        active_stretch += 1
        top_score += 1
        top_risk_reasons.append(
            f"SPY 20日涨幅 {ret_20d:+.1f}% (短期偏热, ≥ {p['rally_20d_warn_pct']:.0f}%)"
        )

    if active_stretch >= 2 and top_score >= 3:
        top_risk_level = "high"
    elif active_stretch >= 1:
        top_risk_level = "elevated"
    else:
        top_risk_level = "normal"

    # VIX latest — realtime override takes precedence if supplied,
    # otherwise forward-fill onto SPY's last date from the EOD frame.
    if realtime_vix is not None and realtime_vix > 0:
        vix_now = float(realtime_vix)
        vix_source = "realtime"
    else:
        vix_aligned = vix_df["Close"].reindex(spy.index, method="ffill")
        vix_now = float(vix_aligned.iloc[-1]) if not vix_aligned.empty else float(vix_df["Close"].iloc[-1])
        vix_source = "eod"

    indicators = {
        "spy_close": round(spy_close, 2),
        "spy_ma200": round(ma200, 2),
        "spy_vs_ma200_pct": round(ma_pct, 2),
        "spy_adx": round(adx, 1),
        "spy_5d_return_pct": round(ret_5d, 2),
        "spy_20d_return_pct": round(ret_20d, 2),
        "spy_rsi14": round(rsi, 1),
        "overbought": rsi >= p["rsi_overbought"],
        "top_risk_score": top_score,
        "top_risk_level": top_risk_level,
        "vix": round(vix_now, 2),
        "vix_source": vix_source,
        "as_of": spy.index[-1].date().isoformat(),
    }

    # ── RED triggers (any one) ─────────────────────────────────────
    red_reasons: List[str] = []
    if ma_pct < -p["ma_buffer_pct"] * 100:
        red_reasons.append(
            f"SPY 跌破 MA200 {abs(ma_pct):.1f}% (阈值 -{p['ma_buffer_pct']*100:.0f}%)"
        )
    if vix_now > p["vix_high"]:
        red_reasons.append(f"VIX={vix_now:.1f} > {p['vix_high']:.0f} (极端恐慌)")
    if ret_5d < p["crash_5d_pct"]:
        red_reasons.append(
            f"SPY 5日跌幅 {ret_5d:.1f}% < {p['crash_5d_pct']:.0f}% (闪崩信号)"
        )

    if red_reasons:
        return RiskState(
            level=RiskLevel.RED,
            reasons=red_reasons,
            indicators=indicators,
            top_risk_level=top_risk_level,
            top_risk_reasons=top_risk_reasons,
        )

    # ── GREEN check (all three) ────────────────────────────────────
    green_ok = (
        ma_pct >= p["ma_buffer_pct"] * 100
        and adx >= p["adx_strong"]
        and vix_now < p["vix_low"]
    )
    if green_ok:
        return RiskState(
            level=RiskLevel.GREEN,
            reasons=[
                f"SPY 站上 MA200 {ma_pct:+.1f}%",
                f"ADX={adx:.1f} (强趋势, > {p['adx_strong']:.0f})",
                f"VIX={vix_now:.1f} (低波, < {p['vix_low']:.0f})",
            ],
            indicators=indicators,
            top_risk_level=top_risk_level,
            top_risk_reasons=top_risk_reasons,
        )

    # ── YELLOW (default) ───────────────────────────────────────────
    yellow_reasons: List[str] = []
    if abs(ma_pct) < p["ma_buffer_pct"] * 100:
        yellow_reasons.append(
            f"SPY 接近 MA200 ({ma_pct:+.1f}%, ±{p['ma_buffer_pct']*100:.0f}% 缓冲区内)"
        )
    elif ma_pct >= p["ma_buffer_pct"] * 100:
        yellow_reasons.append(f"SPY 高于 MA200 {ma_pct:+.1f}%, 但其它条件未达绿")

    if p["adx_weak"] <= adx < p["adx_strong"]:
        yellow_reasons.append(f"ADX={adx:.1f} 趋势衰竭 ({p['adx_weak']:.0f}-{p['adx_strong']:.0f})")
    elif adx < p["adx_weak"]:
        yellow_reasons.append(f"ADX={adx:.1f} 无趋势 (< {p['adx_weak']:.0f})")

    if p["vix_low"] <= vix_now <= p["vix_high"]:
        yellow_reasons.append(f"VIX={vix_now:.1f} 偏高 ({p['vix_low']:.0f}-{p['vix_high']:.0f})")

    if not yellow_reasons:
        yellow_reasons.append("综合判定: 介于绿与红之间")

    if top_risk_level == "high":
        yellow_reasons.append("上涨过热信号共振: 追高风险较高, 等待回撤或确认")

    return RiskState(
        level=RiskLevel.YELLOW,
        reasons=yellow_reasons,
        indicators=indicators,
        top_risk_level=top_risk_level,
        top_risk_reasons=top_risk_reasons,
    )
