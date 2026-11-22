"""Single-symbol backtest tab."""

from datetime import date, timedelta

import pandas as pd
import streamlit as st

import utils
from utils import get_logger, load_toml
from utils.font import setup_chinese_font
from strategy import STRATEGY_MAP
from engine.trader import BacktestEngine
from analysis.risk_metrics import risk_adjusted_summary
from analysis.drawdown import drawdown_summary, underwater_curve

import matplotlib.pyplot as plt

logger = get_logger("dashboard.single_backtest")


def render_single_backtest(selected_symbol, selected_strategy, backtest_years,
                           target_date, provider, cache):
    st.subheader(f"策略回测: {selected_symbol} + {selected_strategy}")

    @st.cache_data(ttl=3600, show_spinner="获取数据中...")
    def _cached_get_daily(symbol, start, end):
        return provider.get_daily(symbol, start=start, end=end)

    start = (pd.Timestamp(target_date) - pd.DateOffset(years=backtest_years)).strftime("%Y-%m-%d")
    end = target_date.isoformat()

    strategy_cls = STRATEGY_MAP.get(selected_strategy)
    if strategy_cls:
        try:
            df = _cached_get_daily(selected_symbol, start, end)
            if df is not None and not df.empty:
                strategy = strategy_cls()
                df_sig = strategy.calculate_indicators(df)
                engine = BacktestEngine(initial_capital=10000)
                bench = engine.run(strategy, df_sig)
                result = engine.get_result(bench)

                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("总收益", f"{result.total_return_pct:+.1f}%")
                col2.metric("夏普", f"{result.sharpe_ratio:.2f}")
                col3.metric("最大回撤", f"{result.max_drawdown_pct:.1f}%")
                col4.metric("胜率", f"{result.win_rate_pct:.1f}%")
                col5.metric("交易", result.total_trades)

                setup_chinese_font()

                fig, ax = plt.subplots(figsize=(10, 4))
                eq = result.equity_curve
                ax.plot(eq.index, eq.values,
                        color="#2ca02c", linewidth=1.2, label="策略权益")

                if result.buy_hold_return_pct != 0:
                    bh_start = result.initial_capital
                    price_data = df["Close"]
                    price_data = price_data[price_data.index >= eq.index[0]]
                    bh_curve = bh_start * (price_data / price_data.iloc[0])
                    ax.plot(bh_curve.index, bh_curve.values,
                            color="#d62728", linewidth=0.8, linestyle="--", alpha=0.7,
                            label="买入持有")

                if result.trades:
                    for t in result.trades:
                        try:
                            pos = eq.index.get_loc(t.entry_date)
                            idx = pos if isinstance(pos, int) else pos.start
                            ax.scatter(eq.index[idx], float(eq.iloc[idx]),
                                       color="limegreen", marker="^", s=50, zorder=5)
                        except (KeyError, IndexError):
                            pass
                        try:
                            pos = eq.index.get_loc(t.exit_date)
                            idx = pos if isinstance(pos, int) else pos.start
                            ax.scatter(eq.index[idx], float(eq.iloc[idx]),
                                       color="red", marker="v", s=50, zorder=5)
                        except (KeyError, IndexError):
                            pass

                ax.axhline(y=10000, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
                ax.set_title(f"{selected_symbol} — {selected_strategy}")
                ax.legend(loc="upper left")
                ax.grid(True, alpha=0.3)
                ax.yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
                st.pyplot(fig)
                plt.close(fig)

                _render_risk_adjusted_section(result)

                with st.expander("最新指标"):
                    last_row = df_sig.iloc[-1]
                    indicator_cols = [c for c in df_sig.columns
                                      if c not in ("Open", "High", "Low", "Close", "Volume", "Signal")]
                    cols = st.columns(min(len(indicator_cols), 6))
                    for i, col_name in enumerate(indicator_cols[:18]):
                        val = last_row[col_name]
                        if isinstance(val, (float, int)) and not pd.isna(val):
                            cols[i % 6].metric(col_name, f"{float(val):.4f}")

                with st.expander(f"交易明细 ({result.total_trades} 笔)"):
                    if result.trades:
                        pf_str = "∞" if result.profit_factor == float("inf") else f"{result.profit_factor:.2f}"
                        st.caption(
                            f"胜率 {result.win_rate_pct:.1f}%  |  "
                            f"盈亏比 {pf_str}  |  "
                            f"均盈 {result.avg_win_pct:+.2f}%  |  "
                            f"均亏 {result.avg_loss_pct:+.2f}%"
                        )
                        trade_rows = []
                        for t in result.trades:
                            direction = getattr(t, 'direction', 'LONG')
                            label = "做多" if direction == "LONG" else "做空"
                            trade_rows.append({
                                "方向": label,
                                "入场日": t.entry_date.strftime("%Y-%m-%d") if hasattr(t.entry_date, "strftime") else str(t.entry_date)[:10],
                                "出场日": t.exit_date.strftime("%Y-%m-%d") if hasattr(t.exit_date, "strftime") else str(t.exit_date)[:10],
                                "数量": t.quantity,
                                "入场价": round(t.entry_price, 2),
                                "出场价": round(t.exit_price, 2),
                                "PnL": round(t.pnl, 0),
                                "PnL%": f"{t.pnl_pct:+.2f}%",
                                "原因": t.exit_reason,
                                "持仓天": t.holding_days,
                            })
                        st.dataframe(pd.DataFrame(trade_rows), width="stretch", hide_index=True)
                    else:
                        st.info("无交易记录")
            else:
                st.warning(f"无 {selected_symbol} 数据")
        except Exception as e:
            st.error(f"回测失败: {e}")

        _render_signal_history_and_comparison(
            selected_symbol, start, end, provider, cache)


def _render_risk_adjusted_section(result):
    """Risk-adjusted metrics + underwater curve + top drawdown episodes.

    Reads from ``result.equity_curve`` which is a pd.Series indexed by date.
    """
    eq = result.equity_curve
    if eq is None or len(eq) < 2:
        return
    rets = eq.pct_change(fill_method=None).dropna()
    if rets.empty:
        return

    st.subheader("风险调整收益")

    metrics = risk_adjusted_summary(rets)
    cols = st.columns(5)
    cols[0].metric("Sortino", f"{metrics['sortino']:.2f}",
                   help="(年化收益 − rf) / 下行波动. 只惩罚下行, 比 Sharpe 更贴近\"风险\".")
    cols[1].metric("Calmar", f"{metrics['calmar']:.2f}",
                   help="CAGR / MaxDD. 越大越好, 1.0 视为合格.")
    omega = metrics["omega"]
    omega_str = "∞" if omega == float("inf") else f"{omega:.2f}"
    cols[2].metric("Omega", omega_str,
                   help="阈值 0 之上 vs 之下的期望比. >1 表示盈亏期望胜出.")
    cols[3].metric("Pain Index", f"{metrics['pain_index_pct']:.2f}%",
                   help="平均回撤深度 (%). 比 MaxDD 反映持续性, 越低越好.")
    cols[4].metric("Pain Ratio", f"{metrics['pain_ratio']:.2f}",
                   help="CAGR / Pain Index. Calmar 的钝化版.")

    # Underwater curve
    uw = underwater_curve(rets)
    if not uw.empty:
        st.caption("水下曲线 (Underwater curve) — 资金距前高的距离, 越长说明回本越慢")
        setup_chinese_font()
        fig, ax = plt.subplots(figsize=(10, 2.5))
        ax.fill_between(uw.index, uw.values, 0,
                        color="#d62728", alpha=0.4, linewidth=0)
        ax.plot(uw.index, uw.values, color="#a01010", linewidth=0.7)
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle=":")
        ax.set_ylabel("Underwater %")
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        st.pyplot(fig)
        plt.close(fig)

    # Drawdown episodes
    dd = drawdown_summary(rets, top_n=5)
    rs = dd["recovery_stats"]
    inline = (
        f"平均回撤深度 **{dd['avg_drawdown_pct']:.2f}%** | "
        f"水下时间占比 **{dd['pct_time_underwater']:.1f}%**"
    )
    if rs["n_episodes"] > 0:
        inline += (
            f" | 已完成 {rs['n_episodes']} 次回撤, "
            f"中位回本 **{rs['median_days']:.0f}** 天 / "
            f"95% 分位 **{rs['p95_days']:.0f}** 天"
        )
    if rs["still_in_drawdown"]:
        inline += f" | ⚠ 当前仍处于回撤, 已 **{rs['current_dd_days']}** 天未回本"
    st.markdown(inline)

    if dd["top_episodes"]:
        with st.expander(f"前 {len(dd['top_episodes'])} 大回撤"):
            rows = []
            for ep in dd["top_episodes"]:
                rec = ep["recovery_date"]
                rec_str = rec.strftime("%Y-%m-%d") if pd.notna(rec) and hasattr(rec, "strftime") else "未恢复"
                rows.append({
                    "起峰日": ep["peak_date"].strftime("%Y-%m-%d") if hasattr(ep["peak_date"], "strftime") else str(ep["peak_date"])[:10],
                    "谷底日": ep["trough_date"].strftime("%Y-%m-%d") if hasattr(ep["trough_date"], "strftime") else str(ep["trough_date"])[:10],
                    "恢复日": rec_str,
                    "深度": f"{ep['depth_pct']:.2f}%",
                    "至谷底天数": int(ep["duration_to_trough_days"]) if not pd.isna(ep["duration_to_trough_days"]) else "-",
                    "回本天数": int(ep["duration_to_recovery_days"]) if not pd.isna(ep["duration_to_recovery_days"]) else "-",
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _render_signal_history_and_comparison(selected_symbol, start, end, provider, cache):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("近期信号历史")
        since = (date.today() - timedelta(days=7)).isoformat()
        rows = cache.query_signals(scan_date=since)
        if rows:
            df_hist = pd.DataFrame(rows)
            df_hist["signal_label"] = df_hist["signal"].map({1: "买入", -1: "卖出", 0: "—"})
            st.dataframe(
                df_hist[["scan_date", "symbol", "strategy", "signal_label", "price"]].tail(20),
                width="stretch", hide_index=True,
            )
        else:
            st.info("近 7 天无信号记录")

    with col2:
        st.subheader("策略 Sharpe 对比")
        strategies_to_compare = ["weekly_macd_kdj", "turtle_trading", "enhanced_macd",
                                 "donchian_breakout", "bollinger_mean_reversion"]

        @st.cache_data(ttl=3600, show_spinner="获取数据中...")
        def _cached_get_daily_cmp(symbol, s_start, s_end):
            return provider.get_daily(symbol, start=s_start, end=s_end)

        compare_data = []
        for sn in strategies_to_compare:
            cls = STRATEGY_MAP.get(sn)
            if cls is None:
                continue
            try:
                df_cmp = _cached_get_daily_cmp(selected_symbol, start, end)
                if df_cmp is None or df_cmp.empty:
                    continue
                s = cls()
                df_s = s.calculate_indicators(df_cmp)
                eng = BacktestEngine(initial_capital=10000)
                ben = eng.run(s, df_s)
                res = eng.get_result(ben)
                compare_data.append({
                    "策略": sn,
                    "Sharpe": round(res.sharpe_ratio, 2),
                    "收益%": round(res.total_return_pct, 1),
                    "回撤%": round(res.max_drawdown_pct, 1),
                    "胜率%": round(res.win_rate_pct, 1),
                    "交易": res.total_trades,
                })
            except Exception:
                pass

        if compare_data:
            df_comp = pd.DataFrame(compare_data).sort_values("Sharpe", ascending=False)
            st.dataframe(df_comp, width="stretch", hide_index=True)
