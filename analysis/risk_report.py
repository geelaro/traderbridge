"""Weekly / monthly risk report aggregator.

Pulls every analysis module's output for the current portfolio into a
single structured dict, then formats it for Feishu (rich card) or
Markdown (plain text / email).

Sections (in order):
    1. Header — date, portfolio size, win/loss snapshot
    2. Risk Light — SPY MA200 + ADX + VIX
    3. VaR / ES — historical, parametric, EVT at 95/99/99.5/99.9
    4. Stress Testing — 5 historical scenarios
    5. Concentration — HHI, sector exposure, top-3
    6. Correlation — effective bets, max pair, clusters
    7. Risk Decomposition — Marginal/Component VaR
    8. Performance Attribution — Brinson
    9. PnL Breakdown — Realized + Unrealized
    10. Drawdown — MaxDD, Underwater %, top episodes

Each section is independent — failures are caught and logged, the
report still renders with partial data.

Usage
-----
    rr = RiskReport(config, provider, cache, target_date=date.today())
    data = rr.build()             # structured dict
    md = rr.to_markdown()         # plain Markdown
    card = rr.to_feishu_card()    # Feishu interactive card payload
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Section:
    """One logical block in the report."""
    title: str
    summary: str = ""              # one-line headline
    metrics: dict = field(default_factory=dict)
    tables: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


class RiskReport:
    """Aggregator for the weekly risk report.

    Heavy I/O happens in :meth:`build` — fetches SPY, VIX, watchlist
    prices, sector ETFs, runs ten different analyses.  Failures in any
    section are caught and reported in the section's ``warnings`` list,
    so partial reports are always produced.
    """

    def __init__(self, config: dict, provider, cache, target_date: Optional[date] = None):
        self.config = config or {}
        self.provider = provider
        self.cache = cache
        self.target_date = target_date or date.today()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> dict:
        """Compute all sections and return a structured dict."""
        sections: list[Section] = []
        for builder in (
            self._build_risk_light,
            self._build_var,
            self._build_stress,
            self._build_concentration,
            self._build_correlation,
            self._build_risk_decomposition,
            self._build_brinson,
            self._build_pnl,
            self._build_drawdown,
        ):
            try:
                sec = builder()
                if sec is not None:
                    sections.append(sec)
            except Exception as exc:
                # WARNING (not ERROR/exception) so an installed
                # NotifyLogHandler doesn't auto-forward this to Feishu —
                # the failing section is already surfaced inside the
                # report itself.
                logger.warning(
                    "Risk report section %s failed: %s: %s",
                    builder.__name__, type(exc).__name__, exc,
                )
                sections.append(Section(
                    title=f"⚠ {builder.__name__}",
                    summary=f"模块失败: {type(exc).__name__}",
                    warnings=[f"{type(exc).__name__}: {exc}"],
                ))

        return {
            "as_of": self.target_date.isoformat(),
            "watchlist_size": len(self.config.get("watchlist", []) or []),
            "sections": sections,
        }

    def to_markdown(self) -> str:
        """Return the report as plain Markdown."""
        data = self.build()
        lines = [
            f"# Traderbridge 风险报告 — {data['as_of']}",
            f"",
            f"组合规模: {data['watchlist_size']} 个 watchlist 标的",
            f"",
        ]
        for sec in data["sections"]:
            lines.append(f"## {sec.title}")
            if sec.summary:
                lines.append(f"_{sec.summary}_")
            lines.append("")
            for k, v in sec.metrics.items():
                lines.append(f"- **{k}**: {v}")
            if sec.metrics:
                lines.append("")
            for table in sec.tables:
                lines.append(self._table_to_md(table))
                lines.append("")
            for w in sec.warnings:
                lines.append(f"> ⚠ {w}")
            if sec.warnings:
                lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def to_feishu_card(self) -> dict:
        """Return a Feishu interactive card payload."""
        data = self.build()
        elements: list = []
        for sec in data["sections"]:
            body = []
            if sec.summary:
                body.append(f"_{sec.summary}_")
            for k, v in sec.metrics.items():
                body.append(f"**{k}**: {v}")
            if sec.warnings:
                body.append("\n".join(f"⚠ {w}" for w in sec.warnings))
            content = "\n".join(body) if body else "—"
            elements.append({
                "tag": "div",
                "text": {"tag": "lark_md",
                         "content": f"**{sec.title}**\n{content}"},
            })
            elements.append({"tag": "hr"})
        return {
            # ``wide_screen_mode`` is required for Feishu APP-mode interactive
            # cards to render properly in some app versions; without it the
            # message may be silently degraded to plain text or dropped.
            "config": {"wide_screen_mode": True, "enable_forward": True},
            "header": {
                "title": {
                    "tag": "plain_text",
                    "content": f"风险报告 — {data['as_of']}",
                },
                "template": "blue",
            },
            "elements": elements,
        }

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _build_risk_light(self) -> Optional[Section]:
        from analysis.risk_monitor import compute_risk_state

        spy_df, vix_df = self._spy_vix_data(years=2)
        if spy_df.empty or vix_df.empty:
            return Section(title="🚦 风险灯", warnings=["SPY/VIX 数据缺失"])

        # Try realtime override (silent on failure)
        try:
            from data.realtime import get_realtime_vix
            live = get_realtime_vix()
        except Exception:
            live = None

        state = compute_risk_state(spy_df, vix_df, realtime_vix=live)
        ind = state.indicators or {}
        label = {"green": "🟢 GREEN", "yellow": "🟡 YELLOW",
                 "red": "🔴 RED"}.get(state.level.value, state.level.value)
        top_label = {
            "normal": "正常",
            "elevated": "偏热",
            "high": "顶部风险偏高",
            "unknown": "数据不足",
        }.get(state.top_risk_level, state.top_risk_level)
        reasons = list(state.reasons or [])
        if state.top_risk_reasons:
            reasons.append(
                f"上涨过热/顶部风险: {top_label} "
                f"({'; '.join(state.top_risk_reasons)})"
            )
        return Section(
            title="🚦 风险灯",
            summary=f"当前 **{label}** | " + " ; ".join(reasons),
            metrics={
                "SPY": f"${ind.get('spy_close', 0):.2f}",
                "vs MA200": f"{ind.get('spy_vs_ma200_pct', 0):+.2f}%",
                "SPY 5日": f"{ind.get('spy_5d_return_pct', 0):+.2f}%",
                "SPY 20日": f"{ind.get('spy_20d_return_pct', 0):+.2f}%",
                "RSI14": f"{ind.get('spy_rsi14', 0):.1f}",
                "顶部风险": top_label,
                "SPY ADX": f"{ind.get('spy_adx', 0):.1f}",
                "VIX": f"{ind.get('vix', 0):.2f} "
                       f"({'实时' if ind.get('vix_source') == 'realtime' else '昨收'})",
            },
        )

    def _build_var(self) -> Optional[Section]:
        from analysis.var import portfolio_returns, var_summary
        from analysis.evt import evt_summary

        prices, weights = self._portfolio_panel(lookback_years=2)
        if prices is None or prices.empty or not weights:
            return Section(title="📉 VaR / 期望损失",
                           warnings=["持仓或价格数据缺失"])
        pf_ret = portfolio_returns(prices, weights)
        if pf_ret.empty:
            return Section(title="📉 VaR / 期望损失",
                           warnings=["组合日收益序列为空"])
        var = var_summary(pf_ret)
        evt = evt_summary(pf_ret)
        m95 = var.get("95%", {})
        m99 = var.get("99%", {})
        # Find EVT entries — defensive against future evt_summary refactors
        # that might return {"comparison": None} or omit the key.
        evt_by = {row["confidence"]: row
                  for row in (evt.get("comparison") or [])}
        return Section(
            title="📉 VaR / 期望损失 (单日)",
            summary=f"基于 {var['n_obs']} 个交易日; 日波动 {var['std'] * 100:.2f}%",
            metrics={
                "95% Historical": f"{m95.get('historical', 0) * 100:.2f}%",
                "95% ES (CVaR)": f"{m95.get('cvar', 0) * 100:.2f}%",
                "99% Historical": f"{m99.get('historical', 0) * 100:.2f}%",
                "99% ES (CVaR)": f"{m99.get('cvar', 0) * 100:.2f}%",
                "99.5% EVT": (f"{evt_by.get('99.5%', {}).get('evt', 0) * 100:.2f}%"
                              if evt_by.get('99.5%', {}).get('evt') else "—"),
                "99.9% EVT": (f"{evt_by.get('99.9%', {}).get('evt', 0) * 100:.2f}%"
                              if evt_by.get('99.9%', {}).get('evt') else "—"),
            },
            warnings=[evt["warning"]] if evt.get("warning") else [],
        )

    def _build_stress(self) -> Optional[Section]:
        from analysis.stress import SCENARIOS, run_scenarios

        prices, weights = self._portfolio_panel(lookback_years=18)
        if prices is None or prices.empty:
            return Section(title="⚡ 历史压力场景",
                           warnings=["历史价格数据不足"])
        results = run_scenarios(prices, weights)
        worst_id, worst = min(
            ((sid, r) for sid, r in results.items() if pd.notna(r["return_pct"])),
            key=lambda kv: kv[1]["return_pct"],
            default=(None, None),
        )
        metrics = {}
        for sid, r in results.items():
            cfg = SCENARIOS[sid]
            if pd.notna(r["return_pct"]):
                metrics[cfg["name"]] = (
                    f"组合 {r['return_pct']:+.1f}% | MaxDD {r['max_dd_pct']:.1f}% "
                    f"(SPY 当时 {cfg['spy_pct']:+.0f}%)"
                )
            else:
                metrics[cfg["name"]] = "数据不足"
        return Section(
            title="⚡ 历史压力场景",
            summary=(f"最坏场景 **{SCENARIOS[worst_id]['name']}** → "
                     f"组合 {worst['return_pct']:+.1f}%") if worst_id else "",
            metrics=metrics,
        )

    def _build_concentration(self) -> Optional[Section]:
        from analysis.concentration import concentration_summary, hhi_label
        from utils.sectors import DEFAULT_SECTORS

        _, weights = self._portfolio_panel(lookback_years=1)
        if not weights:
            return Section(title="🧱 集中度", warnings=["持仓为空"])
        summ = concentration_summary(weights, sector_map=DEFAULT_SECTORS)
        return Section(
            title="🧱 集中度",
            summary=f"HHI {summ['hhi']:.0f} ({summ['hhi_label']}); "
                    f"有效持仓 {summ['effective_n']:.1f}",
            metrics={
                "持仓数": summ["n_holdings"],
                "HHI": f"{summ['hhi']:.0f} ({hhi_label(summ['hhi'])})",
                "有效持仓数": f"{summ['effective_n']:.2f}",
                "Top-3 占比": f"{summ['top_3_weight'] * 100:.1f}%",
                "行业 HHI": (f"{summ.get('sector_hhi', 0):.0f} "
                             f"({hhi_label(summ.get('sector_hhi', 0))})"
                             if summ.get('sector_hhi') is not None else "—"),
            },
        )

    def _build_correlation(self) -> Optional[Section]:
        from analysis.correlation_analysis import correlation_summary

        prices, weights = self._portfolio_panel(lookback_years=1)
        if prices is None or prices.empty:
            return Section(title="🔗 相关性", warnings=["价格数据不足"])
        s = correlation_summary(prices, weights=weights)
        eb = s.get("effective_bets") or {}
        if not eb:
            return Section(title="🔗 相关性",
                           warnings=["数据不足以计算有效赌注"])
        max_pair = s.get("max_pair") or {}
        clusters = s.get("clusters") or {}
        return Section(
            title="🔗 相关性",
            summary=(f"{eb['n_symbols']} 个持仓 → "
                     f"有效 **{eb['effective_n']:.1f}** 个独立赌注"),
            metrics={
                "持仓数": eb["n_symbols"],
                "有效独立赌注": f"{eb['effective_n']:.2f}",
                "独立比例": f"{eb['concentration_ratio'] * 100:.1f}%",
                "最大对相关性": (
                    f"{max_pair.get('correlation', 0):.2f}"
                    f" ({'/'.join(max_pair.get('symbols', ()))})"
                    if max_pair else "—"
                ),
                "层次聚类数": clusters.get("n_clusters", "—"),
            },
        )

    def _build_risk_decomposition(self) -> Optional[Section]:
        from analysis.risk_decomposition import risk_decomposition_summary

        prices, weights = self._portfolio_panel(lookback_years=2)
        if prices is None or prices.empty or not weights:
            return Section(title="📊 风险贡献分解",
                           warnings=["数据不足"])
        s = risk_decomposition_summary(prices, weights)
        if s.get("by_symbol") is None or s["by_symbol"].empty:
            return Section(title="📊 风险贡献分解",
                           warnings=["分解无结果"])
        df = s["by_symbol"].head(5)
        top_lines = []
        for sym in df.index:
            row = df.loc[sym]
            top_lines.append(
                f"{sym}: 权重 {row['weight'] * 100:.1f}% / "
                f"风险贡献 {row['rc_pct']:.1f}%"
            )
        return Section(
            title="📊 风险贡献分解 (Marginal VaR)",
            summary=(f"组合 VaR(95%) **{s['total_var_pct']:.2f}%/日**; "
                     f"最大贡献 **{s['top_contributor']}** "
                     f"({s['top_contributor_pct']:.1f}%)"),
            metrics={
                "组合 VaR(95%)": f"{s['total_var_pct']:.2f}%",
                "Top 贡献标的": str(s["top_contributor"]),
                "Top 贡献占比": f"{s['top_contributor_pct']:.1f}%",
            },
            tables=[{"title": "Top 5 风险贡献", "rows": top_lines}],
        )

    def _build_brinson(self) -> Optional[Section]:
        """Brinson attribution over the last 30 days."""
        from analysis.brinson import (
            SECTOR_ETF, brinson_attribution, compute_period_returns,
            portfolio_sector_breakdown,
        )
        from utils.sectors import DEFAULT_SECTORS

        # Hypothetical positions for the watchlist
        symbols = [item["symbol"]
                   for item in self.config.get("watchlist", []) or []]
        if not symbols:
            return Section(title="🎯 业绩归因 (30 日)",
                           warnings=["watchlist 为空"])
        end = pd.Timestamp(self.target_date)
        start = end - timedelta(days=30)
        sym_prices = self._fetch_panel(symbols, start, end)
        if sym_prices.empty:
            return Section(title="🎯 业绩归因 (30 日)",
                           warnings=["组合价格数据不足"])
        sym_returns = compute_period_returns(
            sym_prices, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        if not sym_returns:
            return Section(title="🎯 业绩归因 (30 日)",
                           warnings=["区间内收益数据不足"])
        sec_w, sec_r = portfolio_sector_breakdown(
            symbols, DEFAULT_SECTORS, sym_returns)
        if not sec_w:
            return Section(title="🎯 业绩归因 (30 日)",
                           warnings=["无法按行业聚合"])

        etfs = sorted(set(SECTOR_ETF.values()))
        etf_prices = self._fetch_panel(etfs, start, end)
        etf_returns = compute_period_returns(
            etf_prices, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        avail = [s for s in SECTOR_ETF if SECTOR_ETF[s] in etf_returns]
        if not avail:
            return Section(title="🎯 业绩归因 (30 日)",
                           warnings=["SPDR Sector ETF 数据不足"])
        bench_w = {s: 1.0 / len(avail) for s in avail}
        bench_r = {s: etf_returns[SECTOR_ETF[s]] for s in avail}

        result = brinson_attribution(sec_w, sec_r, bench_w, bench_r)
        tot = result["totals"]
        return Section(
            title="🎯 业绩归因 30 日 (Brinson)",
            summary=(f"超额 **{tot['active_return'] * 100:+.2f}%** = "
                     f"配置 {tot['allocation'] * 100:+.2f}% + "
                     f"选股 {tot['selection'] * 100:+.2f}% + "
                     f"交互 {tot['interaction'] * 100:+.2f}%"),
            metrics={
                "组合收益": f"{tot['portfolio_return'] * 100:+.2f}%",
                "基准收益": f"{tot['benchmark_return'] * 100:+.2f}%",
                "超额收益": f"{tot['active_return'] * 100:+.2f}%",
                "配置效应": f"{tot['allocation'] * 100:+.2f}%",
                "选股效应": f"{tot['selection'] * 100:+.2f}%",
                "交互效应": f"{tot['interaction'] * 100:+.2f}%",
            },
        )

    def _build_pnl(self) -> Optional[Section]:
        from analysis.pnl_breakdown import pnl_summary
        from analysis.hypothetical_positions import compute_hypothetical_positions

        try:
            positions = compute_hypothetical_positions(
                self.config, self.target_date, self.provider) or []
            for p in positions:
                p.setdefault("shares", 1)
        except Exception:
            positions = []
        summ = pnl_summary(self.cache, positions, period="30d")
        rl = summ["realized"]
        un = summ["unrealized"]
        return Section(
            title="💰 盈亏分析 (Realized 30 日 + Unrealized 当前)",
            summary=(f"30 日 Realized **${rl['total']:+,.0f}**"
                     f" | Unrealized **${un['total']:+,.0f}**"
                     f" | 合计 **${summ['total']:+,.0f}**"),
            metrics={
                "Realized 笔数": rl["n_trades"],
                "Realized 胜率": f"{rl['win_rate_pct']:.1f}%",
                "Realized 均盈": f"${rl['avg_win']:+,.0f}",
                "Realized 均亏": f"${rl['avg_loss']:+,.0f}",
                "当前持仓": un["n_positions"],
                "浮盈持仓": un["n_winning"],
                "浮亏持仓": un["n_losing"],
                "Unrealized 合计": f"${un['total']:+,.0f}",
            },
        )

    def _build_drawdown(self) -> Optional[Section]:
        from analysis.drawdown import drawdown_summary
        from analysis.var import portfolio_returns

        prices, weights = self._portfolio_panel(lookback_years=2)
        if prices is None or prices.empty or not weights:
            return Section(title="📉 回撤分析",
                           warnings=["价格数据缺失"])
        pf_ret = portfolio_returns(prices, weights)
        if pf_ret.empty:
            return Section(title="📉 回撤分析",
                           warnings=["收益序列为空"])
        summ = drawdown_summary(pf_ret, top_n=3)
        rs = summ.get("recovery_stats") or {}
        return Section(
            title="📉 回撤分析",
            summary=(f"MaxDD **{summ['max_drawdown_pct']:.2f}%**; "
                     f"水下时间 **{summ['pct_time_underwater']:.0f}%**" +
                     (f"; ⚠ 当前仍在回撤中 ({rs['current_dd_days']} 天)"
                      if rs.get("still_in_drawdown") else "")),
            metrics={
                "最大回撤": f"{summ['max_drawdown_pct']:.2f}%",
                "平均回撤深度": f"{summ['avg_drawdown_pct']:.2f}%",
                "水下时间占比": f"{summ['pct_time_underwater']:.1f}%",
                "完成回撤次数": rs.get("n_episodes", 0),
                "中位回本天数": (f"{rs['median_days']:.0f}"
                                  if rs.get("median_days") == rs.get("median_days")
                                  else "—"),
                "95% 回本天数": (f"{rs['p95_days']:.0f}"
                                 if rs.get("p95_days") == rs.get("p95_days")
                                 else "—"),
            },
        )

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _spy_vix_data(self, years: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
        end = pd.Timestamp(self.target_date)
        start = (end - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")
        try:
            spy = self.provider.get_daily("SPY", start=start, end=end_str)
        except Exception:
            spy = pd.DataFrame()
        try:
            vix = self.provider.get_daily("^VIX", start=start, end=end_str)
        except Exception:
            vix = pd.DataFrame()
        return (spy if spy is not None else pd.DataFrame(),
                vix if vix is not None else pd.DataFrame())

    def _portfolio_panel(self, lookback_years: int = 2) -> tuple[Optional[pd.DataFrame], dict]:
        """Build (prices_df, equal_weights_dict) for hypothetical positions or watchlist.

        Caches the full-history panel (max lookback = 18y) so callers
        asking for shorter windows just slice the cached frame instead of
        re-fetching from the provider.  Build() may call this 6+ times
        across different sections; without the cache that's 60+ symbol
        round-trips per report.
        """
        # Per-instance lazy cache: full panel + symbols
        if not hasattr(self, "_panel_cache"):
            self._panel_cache = None
            self._panel_cache_years = 0

        from analysis.hypothetical_positions import compute_hypothetical_positions

        if self._panel_cache is None:
            try:
                rows = compute_hypothetical_positions(
                    self.config, self.target_date, self.provider) or []
                symbols = [r["symbol"] for r in rows]
            except Exception:
                symbols = []
            if not symbols:
                symbols = [item["symbol"]
                           for item in self.config.get("watchlist", []) or []]
            if not symbols:
                self._panel_cache = pd.DataFrame()
                self._panel_cache_years = 0
                return None, {}
            # Fetch the largest window we'll need — sections later slice down.
            self._panel_cache_years = 18
            end = pd.Timestamp(self.target_date)
            start = end - pd.DateOffset(years=self._panel_cache_years)
            self._panel_cache = self._fetch_panel(symbols, start, end)

        full = self._panel_cache
        if full is None or full.empty:
            return None, {}

        end = pd.Timestamp(self.target_date)
        start = end - pd.DateOffset(years=lookback_years)
        sliced = full.loc[full.index >= start]
        if sliced.empty:
            return None, {}
        weights = {s: 1.0 for s in sliced.columns}
        return sliced, weights

    def _fetch_panel(self, symbols, start, end) -> pd.DataFrame:
        start_str = (start.strftime("%Y-%m-%d") if hasattr(start, "strftime")
                     else str(start))
        end_str = (end.strftime("%Y-%m-%d") if hasattr(end, "strftime")
                   else str(end))
        series = {}
        for sym in symbols:
            try:
                df = self.provider.get_daily(sym, start=start_str, end=end_str)
            except Exception:
                continue
            if df is None or df.empty or "Close" not in df.columns:
                continue
            series[sym] = df["Close"]
        if not series:
            return pd.DataFrame()
        return pd.concat(series, axis=1).sort_index()

    def _table_to_md(self, table: dict) -> str:
        lines = [f"**{table.get('title', '')}**"]
        for row in table.get("rows", []):
            lines.append(f"- {row}")
        return "\n".join(lines)
