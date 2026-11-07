"""Market state monitor for SPY, QQQ, sentiment, and market stress.

The primary state is a numeric 0-100 sentiment temperature:
0 = extreme oversold, 50 = neutral, 100 = extreme overbought.
SPY and QQQ are calculated independently, then averaged for the overall
market sentiment. A separate green/yellow/red stress light remains for VIX,
trend breaks, and fast drawdowns; sentiment and stress are different axes.
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


_DEFAULTS = {
    "ma200_period": 200,
    "ma_buffer_pct": 0.03,
    "adx_strong": 25.0,
    "adx_weak": 15.0,
    "vix_low": 18.0,
    "vix_high": 30.0,
    "crash_5d_pct": -5.0,
    "rsi_period": 14,
    "rsi_overbought": 70.0,
    "rsi_extreme": 80.0,
    "extension_warn_pct": 10.0,
    "extension_high_pct": 15.0,
    "rally_20d_warn_pct": 8.0,
    "rally_20d_high_pct": 12.0,
    "sentiment_rsi_weight": 0.50,
    "sentiment_extension_weight": 0.30,
    "sentiment_momentum_weight": 0.20,
    "sentiment_extension_range_pct": 15.0,
    "sentiment_momentum_range_pct": 15.0,
}

_LEVEL_RANK = {RiskLevel.GREEN: 0, RiskLevel.YELLOW: 1, RiskLevel.RED: 2}
_SENTIMENT_RANK = {
    "极度超卖": 0, "超卖": 1, "中性": 2, "超买": 3, "极度超买": 4,
}


@dataclass
class RiskState:
    """Market stress plus numeric sentiment for SPY and QQQ."""

    level: RiskLevel
    reasons: List[str] = field(default_factory=list)
    indicators: dict = field(default_factory=dict)
    top_risk_level: str = "unknown"
    top_risk_reasons: List[str] = field(default_factory=list)
    market_states: dict = field(default_factory=dict)
    sentiment_score: Optional[float] = None
    sentiment_label: str = "unknown"
    sentiment_components: dict = field(default_factory=dict)
    market_sentiments: dict = field(default_factory=dict)

    @property
    def emoji(self) -> str:
        return {RiskLevel.GREEN: "🟢", RiskLevel.YELLOW: "🟡", RiskLevel.RED: "🔴"}[self.level]

    def summary(self) -> str:
        head = f"{self.emoji} {self.level.value.upper()}"
        if not self.reasons:
            return head
        body = "\n".join(f"  - {reason}" for reason in self.reasons)
        return f"{head}\n{body}"


def _rsi(close: pd.Series, period: int) -> float:
    delta = close.diff()
    gains = delta.clip(lower=0).rolling(period).mean()
    losses = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gains / losses.replace(0, pd.NA)
    values = 100 - (100 / (1 + rs))
    values = values.where(~((losses == 0) & (gains > 0)), 100.0)
    values = values.where(~((losses == 0) & (gains == 0)), 50.0)
    return float(values.iloc[-1]) if pd.notna(values.iloc[-1]) else 50.0


def _sentiment_label(score: float) -> str:
    if score < 20:
        return "极度超卖"
    if score < 35:
        return "超卖"
    if score > 80:
        return "极度超买"
    if score > 65:
        return "超买"
    return "中性"


def _sentiment_score(ma_pct: float, ret_20d: float, rsi: float, p: dict) -> dict:
    """Convert RSI, long-term extension, and momentum into 0-100."""
    extension_range = max(float(p["sentiment_extension_range_pct"]), 1.0)
    momentum_range = max(float(p["sentiment_momentum_range_pct"]), 1.0)
    extension = max(0.0, min(100.0, 50.0 + ma_pct / extension_range * 50.0))
    momentum = max(0.0, min(100.0, 50.0 + ret_20d / momentum_range * 50.0))
    rsi_component = max(0.0, min(100.0, rsi))
    weights = [
        float(p["sentiment_rsi_weight"]),
        float(p["sentiment_extension_weight"]),
        float(p["sentiment_momentum_weight"]),
    ]
    total = sum(weights) or 1.0
    score = (
        rsi_component * weights[0]
        + extension * weights[1]
        + momentum * weights[2]
    ) / total
    score = round(max(0.0, min(100.0, score)), 1)
    return {
        "score": score,
        "label": _sentiment_label(score),
        "components": {
            "rsi": round(rsi_component, 1),
            "ma200_extension": round(extension, 1),
            "momentum_20d": round(momentum, 1),
        },
    }


def compute_symbol_sentiment(
    df: pd.DataFrame, params: Optional[dict] = None,
) -> Optional[dict]:
    """Return a 0-100 oversold/overbought snapshot for one symbol.

    Uses the same RSI, MA200 extension, and 20-day momentum blend as the
    SPY/QQQ market sentiment display. Returns ``None`` until there is enough
    clean daily history; it is display-only and does not imply a trade action.
    """
    p = {**_DEFAULTS, **(params or {})}
    if df is None or df.empty or "Close" not in df.columns:
        return None

    close = pd.to_numeric(df["Close"], errors="coerce").dropna().astype(float)
    ma_period = int(p["ma200_period"])
    required = ma_period + 20
    if len(close) < required:
        return None

    ma200 = float(close.rolling(ma_period).mean().iloc[-1])
    if not pd.notna(ma200) or ma200 <= 0:
        return None
    last_close = float(close.iloc[-1])
    if last_close <= 0:
        return None

    ma_pct = (last_close / ma200 - 1) * 100
    ret_20d = (last_close / float(close.iloc[-21]) - 1) * 100
    rsi = _rsi(close, int(p["rsi_period"]))
    sentiment = _sentiment_score(ma_pct, ret_20d, rsi, p)
    as_of = pd.Timestamp(close.index[-1]).date().isoformat()
    return {
        **sentiment,
        "rsi14": round(rsi, 1),
        "vs_ma200_pct": round(ma_pct, 2),
        "ret_20d_pct": round(ret_20d, 2),
        "as_of": as_of,
    }


def _top_risk(ma_pct: float, ret_20d: float, rsi: float, p: dict, label: str):
    """Retain the legacy chase-risk fields for compatible consumers."""
    reasons: List[str] = []
    score = 0
    active = 0
    if rsi >= p["rsi_extreme"]:
        active += 1
        score += 2
        reasons.append(f"{label} RSI14={rsi:.1f} 极度超买 (≥ {p['rsi_extreme']:.0f})")
    elif rsi >= p["rsi_overbought"]:
        active += 1
        score += 1
        reasons.append(f"{label} RSI14={rsi:.1f} 超买 (≥ {p['rsi_overbought']:.0f})")
    if ma_pct >= p["extension_high_pct"]:
        active += 1
        score += 2
        reasons.append(f"{label} 高于 MA200 {ma_pct:+.1f}% (极端乖离)")
    elif ma_pct >= p["extension_warn_pct"]:
        active += 1
        score += 1
        reasons.append(f"{label} 高于 MA200 {ma_pct:+.1f}% (乖离偏大)")
    if ret_20d >= p["rally_20d_high_pct"]:
        active += 1
        score += 2
        reasons.append(f"{label} 20日涨幅 {ret_20d:+.1f}% (加速上涨)")
    elif ret_20d >= p["rally_20d_warn_pct"]:
        active += 1
        score += 1
        reasons.append(f"{label} 20日涨幅 {ret_20d:+.1f}% (短期偏热)")
    level = "high" if active >= 2 and score >= 3 else "elevated" if active else "normal"
    return level, score, reasons


def _compute_market_detail(df: pd.DataFrame, label: str, vix: float, p: dict) -> dict:
    required = int(p["ma200_period"]) + 20
    if df is None or df.empty:
        return {"available": False, "reasons": [f"{label} 数据缺失"]}
    if len(df) < required:
        return {"available": False, "reasons": [f"{label} 数据不足 {required} 根, 无法判定"]}

    frame = df.copy()
    frame["MA200"] = frame["Close"].rolling(p["ma200_period"]).mean()
    adx_frame = compute_adx(frame.copy(), 14)
    last = frame.iloc[-1]
    close = float(last["Close"])
    ma200 = float(last["MA200"])
    adx = float(adx_frame["ADX"].iloc[-1]) if "ADX" in adx_frame.columns else 0.0
    ma_pct = (close / ma200 - 1) * 100 if ma200 > 0 else 0.0
    ret_5d = (close / float(frame["Close"].iloc[-6]) - 1) * 100
    ret_20d = (close / float(frame["Close"].iloc[-21]) - 1) * 100
    rsi = _rsi(frame["Close"].astype(float), int(p["rsi_period"]))
    sentiment = _sentiment_score(ma_pct, ret_20d, rsi, p)
    top_level, top_score, top_reasons = _top_risk(ma_pct, ret_20d, rsi, p, label)

    red_reasons: List[str] = []
    if ma_pct < -p["ma_buffer_pct"] * 100:
        red_reasons.append(f"{label} 跌破 MA200 {abs(ma_pct):.1f}%")
    if vix > p["vix_high"]:
        red_reasons.append(f"VIX={vix:.1f} > {p['vix_high']:.0f} (极端恐慌)")
    if ret_5d < p["crash_5d_pct"]:
        red_reasons.append(f"{label} 5日跌幅 {ret_5d:.1f}% (快速下跌)")

    if red_reasons:
        level, reasons = RiskLevel.RED, red_reasons
    else:
        green_ok = ma_pct >= p["ma_buffer_pct"] * 100 and adx >= p["adx_strong"] and vix < p["vix_low"]
        if green_ok:
            level = RiskLevel.GREEN
            reasons = [f"{label} 站上 MA200 {ma_pct:+.1f}%", f"{label} ADX={adx:.1f} 强趋势", f"VIX={vix:.1f} 低波"]
        else:
            level = RiskLevel.YELLOW
            reasons = []
            if abs(ma_pct) < p["ma_buffer_pct"] * 100:
                reasons.append(f"{label} 接近 MA200 ({ma_pct:+.1f}%)")
            if adx < p["adx_strong"]:
                reasons.append(f"{label} ADX={adx:.1f} 趋势不足")
            if p["vix_low"] <= vix <= p["vix_high"]:
                reasons.append(f"VIX={vix:.1f} 偏高")
            if not reasons:
                reasons.append(f"{label} 综合压力处于中间状态")

    indicators = {
        "close": round(close, 2), "ma200": round(ma200, 2),
        "vs_ma200_pct": round(ma_pct, 2), "adx": round(adx, 1),
        "ret_5d_pct": round(ret_5d, 2), "ret_20d_pct": round(ret_20d, 2),
        "rsi14": round(rsi, 1), "overbought": rsi >= p["rsi_overbought"],
        "top_risk_score": top_score, "top_risk_level": top_level,
        "sentiment_score": sentiment["score"], "sentiment_label": sentiment["label"],
        "sentiment_components": sentiment["components"],
        "as_of": frame.index[-1].date().isoformat(),
    }
    return {
        "available": True, "level": level, "reasons": reasons,
        "indicators": indicators, "top_risk_level": top_level,
        "top_risk_reasons": top_reasons,
    }


def compute_risk_state(
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    params: Optional[dict] = None,
    realtime_vix: Optional[float] = None,
    qqq_df: Optional[pd.DataFrame] = None,
) -> RiskState:
    """Compute SPY, QQQ, overall sentiment, and composite stress state."""
    p = {**_DEFAULTS, **(params or {})}
    if spy_df is None or spy_df.empty:
        return RiskState(level=RiskLevel.YELLOW, reasons=["SPY 数据缺失, 默认黄"])
    if vix_df is None or vix_df.empty:
        return RiskState(level=RiskLevel.YELLOW, reasons=["VIX 数据缺失, 默认黄"])

    vix_aligned = vix_df["Close"].reindex(spy_df.index, method="ffill")
    if realtime_vix is not None and realtime_vix > 0:
        vix_now, vix_source = float(realtime_vix), "realtime"
    else:
        vix_now = float(vix_aligned.iloc[-1]) if not vix_aligned.empty else float(vix_df["Close"].iloc[-1])
        vix_source = "eod"

    details = {"SPY": _compute_market_detail(spy_df, "SPY", vix_now, p)}
    if qqq_df is not None and not qqq_df.empty:
        qqq_detail = _compute_market_detail(qqq_df, "QQQ", vix_now, p)
        if qqq_detail.get("available"):
            details["QQQ"] = qqq_detail

    spy_detail = details["SPY"]
    if not spy_detail.get("available"):
        return RiskState(
            level=RiskLevel.YELLOW,
            reasons=spy_detail.get("reasons", ["SPY 数据不可用, 默认黄"]),
            indicators={"vix": round(vix_now, 2), "vix_source": vix_source},
        )

    available = [detail for detail in details.values() if detail.get("available")]
    if not available:
        return RiskState(level=RiskLevel.YELLOW, reasons=["SPY/QQQ 数据不足, 默认黄"])

    level = max((detail["level"] for detail in available), key=_LEVEL_RANK.get)
    reasons = [reason for detail in available for reason in detail["reasons"]]
    top_levels = [detail.get("top_risk_level", "unknown") for detail in available]
    top_level = max(top_levels, key=lambda value: {"unknown": -1, "normal": 0, "elevated": 1, "high": 2}.get(value, -1))
    top_reasons = [reason for detail in available for reason in detail.get("top_risk_reasons", [])]

    scores = [detail["indicators"]["sentiment_score"] for detail in available]
    overall_score = round(sum(scores) / len(scores), 1)
    overall_label = _sentiment_label(overall_score)
    market_sentiments = {
        label: {
            "score": detail["indicators"]["sentiment_score"],
            "label": detail["indicators"]["sentiment_label"],
            "components": detail["indicators"]["sentiment_components"],
        }
        for label, detail in details.items() if detail.get("available")
    }

    indicators = {"vix": round(vix_now, 2), "vix_source": vix_source, "top_risk_level": top_level}
    for label, detail in details.items():
        prefix = label.lower()
        di = detail.get("indicators", {})
        indicators.update({
            f"{prefix}_close": di.get("close"), f"{prefix}_ma200": di.get("ma200"),
            f"{prefix}_vs_ma200_pct": di.get("vs_ma200_pct"), f"{prefix}_adx": di.get("adx"),
            f"{prefix}_5d_return_pct": di.get("ret_5d_pct"), f"{prefix}_20d_return_pct": di.get("ret_20d_pct"),
            f"{prefix}_rsi14": di.get("rsi14"), f"{prefix}_sentiment_score": di.get("sentiment_score"),
            f"{prefix}_sentiment_label": di.get("sentiment_label"),
            f"{prefix}_risk_level": detail["level"].value,
        })

    spy = details["SPY"]["indicators"]
    indicators.update({
        "spy_close": spy["close"], "spy_ma200": spy["ma200"], "spy_vs_ma200_pct": spy["vs_ma200_pct"],
        "spy_adx": spy["adx"], "spy_5d_return_pct": spy["ret_5d_pct"], "spy_20d_return_pct": spy["ret_20d_pct"],
        "spy_rsi14": spy["rsi14"], "overbought": spy["overbought"], "top_risk_score": spy["top_risk_score"],
        "as_of": spy["as_of"], "market_sentiment_score": overall_score,
        "market_sentiment_label": overall_label,
        "spy_sentiment_score": spy["sentiment_score"], "spy_sentiment_label": spy["sentiment_label"],
        "qqq_sentiment_score": details.get("QQQ", {}).get("indicators", {}).get("sentiment_score"),
        "qqq_sentiment_label": details.get("QQQ", {}).get("indicators", {}).get("sentiment_label", "数据不可用"),
    })

    market_states = {
        label: {
            "level": detail["level"].value,
            "reasons": detail["reasons"],
            "indicators": detail["indicators"],
            "sentiment_score": detail["indicators"]["sentiment_score"],
            "sentiment_label": detail["indicators"]["sentiment_label"],
        }
        for label, detail in details.items() if detail.get("available")
    }
    return RiskState(
        level=level, reasons=reasons, indicators=indicators,
        top_risk_level=top_level, top_risk_reasons=top_reasons,
        market_states=market_states, sentiment_score=overall_score,
        sentiment_label=overall_label,
        sentiment_components={label: item["components"] for label, item in market_sentiments.items()},
        market_sentiments=market_sentiments,
    )
