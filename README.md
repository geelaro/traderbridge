# TraderBridge

> 原名 `mytrader` — 2026-05-31 改名

个人量化**风险管理 + 决策辅助**平台 — 数据管线 → 13 策略库 → 回测 + 实盘桥梁 → **22 个风险/业绩分析模块** → 风险告警 + Kill Switch + 周报。**不替你下单，给你海图。**

[![CI](https://github.com/geelaro/traderbridge/actions/workflows/ci.yml/badge.svg)](https://github.com/geelaro/traderbridge/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/geelaro/traderbridge/branch/master/graph/badge.svg)](https://codecov.io/gh/geelaro/traderbridge)

📐 **[完整架构与设计决策 → `docs/architecture.md`](docs/architecture.md)**

## 规模

| 维度 | 数据 |
|------|------|
| 业务代码 | ~19,000 行 Python |
| 测试 | **1400 passed / 2 skipped / 67 文件 / 79.3% 覆盖率** |
| Dashboard tab | **11 个** |
| 数据源 | 6 个(sina_us / tencent / cboe / yahoo_chart / yahoo_realtime / futu) |
| 策略库 | 13 个(9 active + 3 弃用 + 1 ensemble) |
| 分析模块 | **22 个**(VaR / EVT / Brinson / Drawdown / Marginal VaR ...) |

## 核心功能

### 数据 + 策略 + 回测(基础设施)

| 模块 | 说明 |
|------|------|
| 数据管线 | 6 数据源 + SQLite 增量缓存 + 跨源校验 + 拆股统一(`apply_us_splits`) |
| 策略库 | 13 策略,BaseStrategy + ChandelierTrailingExit Mixin + Ensemble 投票 |
| MTF 框架 | 多时间框架接口 `calculate_indicators(df, df_weekly)` |
| 策略门控 | SignalGate:市场状态感知 + 风控暂停 + 敞口检查 |
| 回测引擎 | 含滑点佣金,有符号持仓(多空),单标 + 组合回测,Walk-forward 优化 |
| 仓位管理 | `fixed_capital` / `risk_budget` 双模式,回测实盘共用 `utils/sizing.py` |

### 风险管理 + 业绩分析(平台核心)

| 模块 | 说明 |
|------|------|
| **VaR / 期望损失** | Historical / Parametric / Expected Shortfall,95% 99% 双置信度 |
| **EVT 尾部估计** | GPD POT 拟合,99.5% / 99.9% 高分位 VaR 外推 |
| **历史压力测试** | 5 场景:2008 雷曼 / 2018 Q4 / 2020 COVID / 2022 加息 / 2015 8.11 |
| **Marginal / Component VaR** | Gaussian 解析分解,Risk Parity 权重求解 |
| **What-If 假设调仓** | 调仓前预演 VaR / HHI / 行业 / 集中度变化 |
| **Brinson 业绩归因** | 行业配置 / 选股 / 交互三效应分解 |
| **6 因子归因** | Jensen α + Newey-West HAC,SPDR ETF 代理 |
| **集中度** | HHI / Effective N / Sector HHI / Top-N 占比 |
| **相关性结构** | 层次聚类 + Effective Bets(PCA),最大对相关性 |
| **回撤分析** | Underwater Curve + Drawdown Episodes + Time-to-Recover |
| **风险调整收益** | Sortino / Calmar / Omega / Pain Index / Information Ratio |
| **盈亏分析** | Realized(trade_pnl 表) + Unrealized(持仓浮动)拆分 |

### 实时告警 + 决策辅助

| 模块 | 说明 |
|------|------|
| **风险灯** | SPY MA200 + ADX + VIX **(实时,Yahoo spark)** → 🟢/🟡/🔴 |
| **风险告警状态机** | 风险灯转 RED / VIX 突破 / 持仓临近止损,飞书推送 + 历史落盘 |
| **Kill Switch** | 一键紧急平仓,纯手动 + 双确认(VIX>50 实证是抄底信号,不自动触发) |
| **风险报告** | `scripts/weekly_risk_report.py`,9 section 综合周报 → Markdown + 飞书 |
| **CBOE VIX** | 官方 CSV 全量日线 + Yahoo 实时旁路覆盖 |

### 实盘桥梁

| 模块 | 说明 |
|------|------|
| Broker 抽象 | MockBroker(dry-run)+ FutuBroker(富途 OpenD) |
| 风控 | 连续亏损熔断、波动率自适应仓位、单日上限、总敞口、行业权重、止损冷却期 |
| 风控持久化 | risk_state + entry_prices 表,守护进程重启后恢复 |
| 订单管理 | 部分成交轮询、限价单超时撤单、滑点统计、批量等风险分配 |

### 运维 + 可观测

| 模块 | 说明 |
|------|------|
| 日志 | 结构化 JSON 日志,/health HTTP 端点,ops_log + alert_history 审计 |
| Dashboard | Streamlit Web UI,11 个 tab(见下) |
| CI | GitHub Actions + Codecov + badge,push/PR 自动跑完整测试套件 |
| Docker | docker-compose.yml 一键部署 traderbridge + futu-opend + HEALTHCHECK |

## Dashboard 11 个 Tab

| Tab | 功能 |
|-----|------|
| 单标的回测 | 选标的/策略 → 权益曲线 + 买卖点 + 策略对比 |
| 组合回测 | 多标组合回测 → 风险看板 + PnL 归因 + 交易明细 |
| 因子归因 | 6 因子 OLS + Newey-West HAC,Jensen α + β 暴露 |
| 业绩归因 Brinson | 行业配置 / 选股 / 交互三效应,vs SPDR Sector ETFs |
| 盈亏分析 | Realized(trade_pnl) + Unrealized(持仓浮动)拆分 |
| 信号有效性 | Forward return 分布,信号预测力 |
| 风险量化 | VaR + EVT + Stress + Concentration + Correlation + Marginal VaR + What-If |
| 风险告警历史 | 三类告警时间线 + 按日柱状图 + 阈值验证 |
| 📑 风险报告 | 9 section 综合周报,Markdown + 推飞书 |
| 🚨 Kill Switch | 紧急平仓,双确认 + Dry Run + Reset |
| 配置管理 | watchlist 编辑器 |

## 策略

| 策略 | 类型 | 方向 | 推荐度 | 描述 |
|------|------|:---:|:---:|------|
| `spy_ma_breakout` | 宏观 + 突破 | 纯多 | ★★★ | SPY MA200 宏观过滤 + MA200 趋势 + 20日新高突破 + ATR 动态止盈 + MA 止损（长线） |
| `weekly_macd_kdj` | 周线趋势 | 纯多 | ★★★ | KDJ 金叉买入 + MACD 死叉卖出（主力）|
| `trend_follower` | 短线趋势 | 纯多 | ★★★ | SMA5/20 + ADX + Chandelier 尾随止损（短线优化版）|
| `turtle_trading` | 趋势 | 多空 | ★★☆ | 大哥2.2 递归 SMA + 唐奇安通道 + ATR 固定止损 + 趋势过滤 |
| `donchian_breakout` | 动量突破 | 多空 | ★★ | 唐奇安通道突破 + 移动止损 |
| `daily_macd_kdj` | 日线 KDJ | 纯多 | ★★ | 日线 KDJ 金叉 + MACD 死叉 + ATR 止损 |
| `atr_breakout` | 波动率突破 | 纯多 | ★★ | MA + N×ATR 突破 + 移动止损 |
| `weekly_macd` | 周线趋势 | 纯多 | ★ | MACD 金叉死叉 |
| `macd_kdj` | 日/周 KDJ | 纯多 | ★ | 统一 MACDKDJStrategy，freq="W"/"D" + use_atr_stop |

> 推荐度基于 2026-05 全策略鲁棒性扫描 + 多标的 6 年回测
>
> ~~`enhanced_macd` `bollinger_mean_reversion` `bollinger_squeeze`~~ — 已从活跃策略移除（过拟合/零交易），源文件保留于 `strategy/` 供测试兼容

### 策略参数速查

| 策略 | 关键参数 | 入场条件 | 出场条件 |
|------|---------|---------|---------|
| `spy_ma_breakout` | ma=200, high=20, tp_ATR=4.0 | SPY>MA200 + Close>MA200 + 20日新高 | 止盈: entry+ATR×4 / 止损: Close<MA |
| `trend_follower` | short=5, long=20, ADX_th=15, trail=2.5 | SMA5>SMA20 + ADX>15 + +DI>-DI | 尾随止损: Close<=最高价-ATR×2.5 |
| `turtle_trading` | short=20, long=50, channel=20, trail=3.0 | 递归SMA交叉 + 通道突破(前根) + 趋势过滤 | 入场ATR固定: Close<=最高High-ATR_entry×3.0 |
| `weekly_macd_kdj` | n=9, k=3, d=3 | KDJ金叉(周线) | MACD死叉(周线) |
| `daily_macd_kdj` | n=9, k=3, d=3, trail=3.0 | KDJ金叉(日线) | MACD死叉 + ATR尾随止损 |

## 项目结构

严格分层,单向依赖,无循环引用。完整架构图见 [`docs/architecture.md`](docs/architecture.md)。

```
traderbridge/
  data/              # 数据管线 (7 文件 / 2081 行)
  ├─ sources.py             #   6 数据源 + apply_us_splits 共享拆股调整
  ├─ provider.py            #   DataProvider 失败链 sina→tencent→yahoo
  ├─ cache.py               #   CacheManager (SQLite) + StateStore + OpsLogger
  ├─ realtime.py            #   实时 VIX 旁路 (Yahoo spark/chart,独立 session)
  ├─ quality.py             #   数据质量校验
  ├─ protocol.py            #   SOURCE_PRIORITY 路由
  └─ splits.json            #   美股拆股调整因子

  strategy/          # 策略库 (16 文件 / 2051 行)
  ├─ base.py                #   BaseStrategy + ChandelierTrailingExit Mixin
  ├─ 9 个 active 策略       #   spy_ma_breakout / weekly_macd_kdj / ...
  ├─ ensemble.py            #   StrategyEnsemble 加权投票
  └─ STRATEGY_GUIDE.md      #   策略使用指南

  broker/            # 券商抽象 (4 文件 / 960 行)
  ├─ base.py                #   Broker ABC + Order/Position/Account dataclass
  ├─ mock.py                #   MockBroker dry-run
  └─ futu.py                #   FutuBroker (富途 OpenD)

  engine/            # 回测引擎 (5 文件 / 2525 行)
  ├─ trader.py              #   BacktestEngine 单标回测,有符号持仓
  ├─ portfolio.py           #   PortfolioBacktest 组合回测
  ├─ execution.py           #   ExecutionModel 订单执行语义
  └─ optimize.py             #   grid_search + walk_forward

  analysis/          # 风险/业绩分析核心 ★ (22 文件 / 5299 行)
  ├─ risk_monitor.py        #   Risk Light (SPY MA200 + ADX + VIX)
  ├─ var.py                 #   Historical / Parametric VaR + ES
  ├─ evt.py                 #   GPD POT 尾部估计
  ├─ stress.py              #   5 历史场景压力测试
  ├─ concentration.py       #   HHI / Sector / Effective N
  ├─ correlation_analysis.py #  层次聚类 + PCA Effective Bets
  ├─ risk_decomposition.py  #   Marginal/Component VaR + Risk Parity
  ├─ what_if.py             #   假设调仓预览
  ├─ risk_metrics.py        #   Sortino/Calmar/Omega/Pain Index
  ├─ drawdown.py            #   Underwater + Episodes + Time-to-Recover
  ├─ brinson.py             #   Brinson 业绩归因(配置/选股/交互)
  ├─ factor_attribution.py  #   6 因子 OLS + Newey-West HAC
  ├─ pnl_breakdown.py       #   Realized + Unrealized 拆分
  ├─ risk_report.py         #   9 section 综合周报
  ├─ rolling_alpha.py       #   滚动 α 衰减
  ├─ forward_return.py      #   信号 forward return 分布
  ├─ cost_sensitivity.py    #   成本敏感性扫描
  ├─ param_robustness.py    #   参数鲁棒性
  ├─ monte_carlo.py         #   Monte Carlo 模拟
  └─ stress_test.py         #   (历史遗留,被 stress.py 取代中)

  live/              # 实盘桥梁 (6 文件 / 1146 行)
  ├─ risk_controller.py     #   风控检查 + 仓位 + 熔断持久化
  ├─ order_manager.py       #   信号→订单 4 路矩阵 + 批量等风险
  ├─ risk_alerts.py         #   三类告警状态机 + 飞书推送
  ├─ kill_switch.py         #   紧急平仓(纯手动 + 双确认)
  └─ position_stops.py      #   Chandelier 止损 + 假设持仓计算

  utils/             # 横切关注点 (12 文件 / 1787 行)
  ├─ notify.py              #   飞书通知 (Webhook/App 双模式 + 5 种卡片)
  ├─ signal_gate.py         #   策略门控(市场状态+风控+敞口)
  ├─ market_state.py        #   四象限市场状态分类
  ├─ signal_scanner.py      #   MTF 跨频率信号扫描
  ├─ logging.py             #   结构化 JSON 日志
  ├─ sizing.py              #   统一仓位计算(回测实盘共用)
  ├─ sectors.py             #   行业分类映射(DEFAULT_SECTORS)
  ├─ risk.py                #   RiskLimits 数据类
  ├─ metrics.py             #   回撤统计、敞口重构
  ├─ env.py                 #   TOML 读写 (load_toml/save_toml)
  ├─ bootstrap.py           #   显式运行时初始化 (.env / Matplotlib / Windows UTF-8)
  └─ font.py                #   中文字体兼容

  dashboard/         # Streamlit UI (14 文件 / 3139 行)
  ├─ main.py                #   11 tab 编排
  ├─ signals.py             #   今日信号 + 风险灯 + 持仓监控
  ├─ single_backtest.py     #   单标回测 + 风险调整收益
  ├─ portfolio_backtest.py  #   组合回测 + Monte Carlo
  ├─ factor_attribution.py  #   因子归因 tab
  ├─ brinson_attribution.py #   Brinson 业绩归因 tab
  ├─ pnl_breakdown.py       #   盈亏分析 tab
  ├─ risk_analytics.py      #   风险量化 tab (VaR/Stress/Conc/Corr/MVaR)
  ├─ signal_effectiveness.py #  信号有效性 tab
  ├─ risk_report.py         #   📑 风险报告 tab
  ├─ kill_switch.py         #   🚨 Kill Switch tab
  ├─ config_editor.py       #   watchlist 配置编辑器
  └─ ops.py                 #   交易记录 + 行业分布

  scripts/           # 命令入口 (4 文件 / 364 行)
  └─ weekly_risk_report.py  #   cron 入口,周报推飞书

  tests/             # 1400 passed, 2 skipped (67 文件 / 12806 行)
  docs/              # 架构文档
  └─ architecture.md
  reports/           # 自动生成的 CSV + PNG
  logs/              # JSON 结构化日志 + 周报 markdown

  live_trader.py     # 实盘信号执行 + 风控 + HTTP /health
  daily.py           # 每日回溯 + --optimize 滚动优化
  dashboard.py       # Streamlit 入口
  watchlist.toml     # 标的 + 策略 + 风控 + 告警阈值
  docker-compose.yml # 一键部署 (traderbridge + futu-opend)
  .coveragerc        # 覆盖率配置
  .codecov.yml       # codecov 阈值
```

## 快速开始

```bash
# Python 3.10+
pip install pipenv
pipenv install --dev
pipenv run pytest -q   # 1400 passed / 2 skipped
```

**国内必须配置代理**(Yahoo / GitHub 直连不通)。在 `.env` 写:

```
HTTPS_PROXY=http://127.0.0.1:7897
HTTP_PROXY=http://127.0.0.1:7897
NO_PROXY=localhost,127.0.0.1
```

Pipenv 会自动加载 `.env`,所有 Python 进程都会用上代理。

## 核心流程（30 分钟上手）

### 1. 数据获取 + 回测（5 min）

```bash
pipenv run python -c "
from engine.trader import run_backtest, print_result
from strategy import WeeklyMACD_KDJ
r, _ = run_backtest('AAPL', '2020-01-01', strategy_cls=WeeklyMACD_KDJ)
print_result(r)
"
```

### 2. Dashboard（5 min）

```bash
pipenv run streamlit run dashboard.py --server.port 8501 --server.headless true  # http://localhost:8501
```

- **单标 Tab**：选标的/策略 → 权益曲线 + 买卖点 + 策略对比
- **组合 Tab**：多标组合回测 → 风险看板 + PnL 归因 + 交易明细筛选导出

### 3. 组合回测（5 min）

```bash
pipenv run python engine/portfolio.py
```

### 4. 策略分析（10 min）

```bash
# 成本敏感性：策略在不同费率下的表现
pipenv run python analysis/cost_sensitivity.py -s weekly_macd_kdj --symbol AAPL

# 参数鲁棒性：参数是否稳定（核心诊断工具）
pipenv run python analysis/param_robustness.py -s weekly_macd_kdj --symbol AAPL
pipenv run python analysis/param_robustness.py -s weekly_macd_kdj --symbol AAPL --sizing-mode risk_budget
```

### 5. 每日扫描（5 min）

```bash
pipenv run python daily.py              # 今日信号
pipenv run python daily.py --history    # 近 7 天历史
pipenv run python daily.py --optimize   # 扫描 + 滚动优化
```

### 6. 周报推送

```bash
# 手动生成 + 推飞书
pipenv run python scripts/weekly_risk_report.py

# Dry-run（只生成 Markdown，不推飞书）
pipenv run python scripts/weekly_risk_report.py --dry-run

# 挂 cron（每周一上午 9 点推送）
# 0 9 * * 1 cd /path/to/traderbridge && pipenv run python scripts/weekly_risk_report.py
```

报告涵盖 9 个 section:风险灯 / VaR + ES + EVT / 5 场景压力测试 / 集中度 /
相关性 / Marginal VaR / Brinson 业绩归因 / Realized+Unrealized PnL / 回撤分析。

### 7. 紧急停机(实盘)

Dashboard 打开 **🚨 Kill Switch** tab:

1. 输入 `CONFIRM` 解锁按钮
2. 填触发原因(必填,审计用)
3. 点 **🚨 紧急平仓全部** 或 **Dry Run 预演**

会自动:
- 对每个非零持仓下市价 opposite 单
- 设 `risk_ctrl.trading_paused = True`(daemon 不再开新仓)
- 写 `alert_history` 审计
- 推飞书 RED 卡

**完全手动触发**,无任何自动阈值绑定。基于实证:CBOE VIX 36 年史中 VIX > 50
共 5 次,之后 SPY 250 日平均涨 **+44.6%**(是抄底信号而非清仓信号)。
任何基于 VIX/回撤的自动平仓都会反向伤害。

## 推荐参数模板

### 主力：spy_ma_breakout（长线）

```python
from engine.trader import run_backtest
from strategy.spy_ma_breakout import SPYMABreakout

result, df = run_backtest(
    "QQQ", "2020-01-01",
    strategy_cls=SPYMABreakout,
    sizing_mode="risk_budget",
    risk_per_trade=0.02,
    ma_period=200, high_period=20, take_profit_atr_mult=4.0,
)
```
| 标的 | MA | OOS 收益 | Sharpe | 评级 |
|------|-----|--------:|------:|:---:|
| NVDA | 200 | +66% | 1.37 | STABLE |
| QQQ | 200 | +37% | 0.74 | STABLE |
| SPY | 200 | +25% | 0.60 | STABLE |

### 短线：trend_follower

```python
from strategy import TrendFollower

result, df = run_backtest(
    "QQQ", "2020-01-01",
    strategy_cls=TrendFollower,
    sizing_mode="risk_budget",
    risk_per_trade=0.02,
    short_ma=5, long_ma=20,
    adx_threshold=15.0, trail_atr_mult=2.5,
)
```
| 标的 | OOS 收益 | Sharpe | 交易 |
|------|--------:|------:|:---:|
| NVDA | +53% | 1.26 | 32 |
| TSLA | +33% | 1.01 | 30 |
| QQQ | +27% | 1.08 | 27 |

### 备选策略

| 策略 | 适用场景 | 推荐标的 |
|------|------|------|
| `turtle_trading` | 多空双向，趋势+震荡 | SPY / QQQ / PFE |
| `weekly_macd_kdj` | 周线低频，低回撤 | AAPL / NVDA |
| `daily_macd_kdj` | 日线高频，分散风险 | AAPL / TSLA |
| `donchian_breakout` | 多空双向，波动率敏感 | QQQ |

## 仓位管理

两种仓位模式，通过 `sizing_mode` 切换。**回测与实盘共用同一仓位计算函数**（`utils/sizing.py`），确保参数优化结果可直接迁移到实盘。

### fixed_capital（默认）

策略自行决定仓位大小（通常为可用资金的 95%）：

```python
result, _ = run_backtest("AAPL", "2020-01-01", strategy_cls=WeeklyMACD_KDJ)
```

### risk_budget（风险预算，推荐实盘使用）

引擎统一按 `风险金额 / 止损距离` 计算仓位。公式与 `RiskController` 共用 `calc_risk_budget_qty()`：

```
qty = capital × risk_pct / (ATR × stop_atr_mult) × vol_scalar
```

其中 `vol_scalar` 实盘启用（根据 `vol_sensitivity` 自适应缩放），回测默认关闭。

```python
result, _ = run_backtest(
    "TSLA", "2020-01-01", strategy_cls=WeeklyMACD_KDJ,
    sizing_mode="risk_budget",
    risk_per_trade=0.02,   # 单笔风险 2%
    risk_atr_mult=2.0,     # 止损 = 2×ATR
)
```

| 模式 | 收益特征 | 回撤特征 | 适用场景 |
|------|:---:|:---:|------|
| fixed_capital | 高（满仓复利） | 高 | 回测研究，了解策略上限 |
| risk_budget 2% | 中 | 可控（<10%） | 实盘稳健执行 |

## 风控

组合回测支持完整的风控层级，所有被拦截信号记录 `rejections`：

```python
from engine.portfolio import PortfolioBacktest, Leg
from utils.sectors import DEFAULT_SECTORS

bt = PortfolioBacktest(
    legs=[Leg("AAPL", "weekly_macd_kdj"), Leg("NVDA", "weekly_macd_kdj"),
          Leg("TSLA", "weekly_macd_kdj"), Leg("SPY", "turtle_trading")],
    initial_capital=100000,
    allocation="dynamic_equal",
    # --- 仓位 ---
    sizing_mode="risk_budget", risk_per_trade=0.02,
    # --- 组合风控 ---
    max_symbol_weight=0.25,           # 单标的上限 25%
    max_sector_weight=0.30,           # 单行业上限 30%
    max_gross_exposure=0.80,          # 总敞口 ≤ 80%
    max_daily_new_positions=3,        # 单日最大新开仓
    cooldown_after_stop_days=10,      # 止损后 10 天内禁止重入
    sector_map=DEFAULT_SECTORS,
)
result = bt.run(start="2020-01-01")
result.summary()
```

## 实盘前检查清单

在启动 `live_trader.py` 实盘前，逐项确认：

- [ ] **637 测试全部通过** `pytest tests/ -q`
- [ ] **黄金样本无漂移** — CI 绿标
- [ ] `param_robustness` 评级 ROBUST 或 STABLE，非 OVERFIT
- [ ] `cost_sensitivity` 评级 A 或 B（10bp/3bp 佣金下仍盈利）
- [ ] `risk_budget` 模式 MaxDD < 15%（扛得住）
- [ ] `daily.py` 能正常输出今日信号，无数据缺失告警
- [ ] FutuOpenD 已启动（如用富途）— `ps aux | grep FutuOpenD`
- [ ] 飞书 Webhook 已配置（如用通知）— `echo $FEISHU_WEBHOOK`
- [ ] MockBroker dry-run 先跑一周，确认无异常
- [ ] 初始资金 ≤ 可承受全部亏损的金额
- [ ] **active 策略对 P2-1 不敏感**（见下方 Known Issues）

## 因子归因（解构组合收益）

回测出的 Sharpe / CAGR 不直接说明策略好坏——可能只是市场 Beta + 因子暴露的杠杆叠加。
`analysis/factor_attribution.py` 用 ETF 代理因子做 OLS 回归（Newey-West HAC 修正），
告诉你"组合收益里 α 占多少、各因子占多少、α 是否统计显著"。

### 因子集

| 因子 | 代理 ETF | 含义 | 起始 |
|---|---|---|---|
| MKT | SPY − SHV | 市场超额 | 2007 |
| SMB | IWM − SPY | 小盘溢价 | 2000 |
| HML | IVE − IVW | 价值 vs 成长 | 2000 |
| MOM | MTUM − SPY | 动量溢价 | **2013-08** |
| QMJ | QUAL − SPY | 质量溢价 | 2013-07 |
| BAB | USMV − SPY | 低波动溢价 | 2011-10 |

`mode="full"` 用 6 因子（数据需 ≥ 2013-08）；`mode="ff3"` 用 MKT/SMB/HML 三因子（可回溯到 2000）。

### 用法

```python
from engine.portfolio import Leg, PortfolioBacktest
from analysis.factor_returns import FactorReturns
from analysis.factor_attribution import FactorAttribution

bt = PortfolioBacktest(
    legs=[Leg("AAPL", "weekly_macd_kdj"), Leg("NVDA", "weekly_macd_kdj"), ...],
    initial_capital=100000,
)
result = bt.run(start="2018-01-01", end="2024-12-31")

factors = FactorReturns(mode="full").load("2018-01-01", "2024-12-31")
attr = FactorAttribution(result.equity_curve, factors)
print(attr.regress().summary())
```

或者直接在 dashboard 里点 "因子归因" tab。

### 判读规则

| α t-stat | R² | 结论 |
|---|---|---|
| > 2.0 | < 0.7 | **真 alpha** — 值得放大 / 加杠杆 |
| > 2.0 | > 0.85 | **可疑** — alpha 显著但 R² 极高，可能是因子组合 |
| < 1.5 | > 0.85 | **基本无 alpha** — 收益基本被因子暴露解释，简化为 ETF 组合 |
| < 1.5 | < 0.7 | **信号弱** — alpha 不显著且 R² 偏低，样本不足或策略不稳 |

### 滚动 α — 检查 alpha 是否退化

```python
rolling = attr.rolling_alpha(window_days=252)  # 1 年滚动窗口
# rolling['alpha_tstat'] 持续 > 2 → α 稳定
# 趋势下行 → α 退化中, 策略需要重新拟合或退役
```

或运行 `pipenv run python analysis/rolling_alpha.py` 输出到 `reports/rolling_alpha.png`。

## 信号有效性（Forward Return）

回答跟回测**不同的**问题：不是"策略整体能否赚钱"，而是"信号本身有没有预测力"。

策略整体不赚钱有 N 种原因（出场太早、止损太紧、sizing 不当），但**信号本身**可能是真有效的。`analysis/forward_return.py` 把这两件事解耦：对每个 `Signal == 1` 的 bar，计算后 30/90/180 个交易日的实际收益分布。

### 用法

```python
from analysis.forward_return import compute_forward_returns, summarise

# df 是策略 calculate_indicators 之后的 DataFrame, 含 Signal 列
fr = compute_forward_returns(df, horizons=[30, 90, 180], direction=1)
stats = summarise(fr, direction=1)
print(stats.summary())
```

或 dashboard 的"信号有效性" tab，选标的 + 策略 + 方向 + horizons → 直接看分布直方图。

### 判读

| Sharpe | win_rate | 含义 |
|---|---|---|
| > 0.5 | > 55% | **信号有效** — 入场时机有预测力，若策略不赚钱大概率是出场逻辑问题 |
| < 0.5 | > 55% | 边际有效 — win_rate 还行但收益分布平 |
| < 0.5 | < 55% | **信号无预测力** — 改入场逻辑或换标的 |
| n < 10 | — | 样本不足，结论不可靠 |

### 示例发现

跑 `weekly_macd_kdj` 买入信号在当前 watchlist (2018-2026) 上：

```
horizon    n   median    win%   sharpe
  30d    370  +11.05%   70%    +0.43
  90d    299  +37.44%   77%    +0.41
 180d    208  +52.59%   84%    +0.51   ← 信号有效
```

**Forward return 看上去高（+52.6% median）远超策略整体 CAGR (+19%)**，第一直觉会想"出场逻辑太严苛，吞掉了 alpha"。但用 `scripts/exit_experiment.py` 实验对比四种出场规则（默认 MACD 死叉 / 加 ATR 止损 / 慢 MACD / 完全不出场 / B&H）后真相相反：

| 策略 | CAGR | Sharpe | MaxDD |
|---|---|---|---|
| 默认 (MACD 死叉) | +19.0% | **1.49** | -43% |
| 不主动出场（alpha 上限）| +32.5% | 1.26 | -72% |
| 等权 B&H | +54.5% | 0.98 | -47% |

**MACD 死叉出场少赚 13.5%/年但救回 29% 回撤，Sharpe 反而是四个里最高的**。Forward return 信号确实有效，**出场逻辑是把"好信号"转成"好策略"的关键机制，不是 alpha 杀手**。结论：默认参数已是风险调整最优。Forward return 在这里的价值是**验证现状是对的**，不是发现要改什么。

## Known Issues

### P2-1：实盘 daemon 用未完成 bar 算信号

**现状**：`live_trader.py --daemon` 每 5 分钟轮询，扫描时用的是"当日未收盘 bar"
（Close = 当前 last_price，盘后会被真实收盘价覆盖）。

**影响因策略而异**：

| 策略 | 实盘安全等级 | 原因 |
|---|---|---|
| `weekly_macd_kdj` | ✓ 安全 | 周线信号本就稀疏（≤1 次/周），日内闪烁概率极低 |
| `turtle_trading` | ⚠ 中 | 突破点附近徘徊时偶有日内闪烁 |
| `daily_macd_kdj` | ⚠ 高 | 日内 close 波动直接驱动 K/D，假入场风险大 |
| `rsi2_mean_reversion` | ⚠ 高 | 依赖"3 连阴"+当日 close 方向，反复触发 |
| `spy_ma_breakout` | ⚠ 极高 | `Close == N_day_high` 盘中创新高就触发 |

**当前主力 `weekly_macd_kdj` 实盘行为是安全的**，因此 P2-1 暂时未修。

**触发修复的场景**（任一即应先做 P2-1）：
- 把任何 ⚠ 标记的策略设为 watchlist 的 `active`
- 启动多因子模型 / Ensemble 实盘下单
- 接入日内策略
- 想让回测优化的参数可严格迁移到实盘

**修复方向**（已立项，不在当前 ROADMAP）：
- 入场延次日开盘提交（次日 MARKET 单，与回测 `next_open` 对齐）
- 出场保持即时响应（紧急止损 / 跳空保护）
- 新增 `pending_orders` 表 + post-close / pre-market 双时点

## 运维特性

### 日志

三类日志隔离 + 文件 JSON 格式输出：

| 日志文件 | 写入者 | 格式 |
|---------|--------|------|
| `logs/live.log` | live_trader.py | JSON 行（Loki/ELK 可解析） |
| `logs/daily.log` | daily.py | JSON 行 |
| `logs/traderbridge.log` | 全部模块 | JSON 行 |

### Docker 部署

```bash
docker compose up -d
```

### 市场状态配置

`watchlist.toml`:

```toml
[market_state]
enabled = true
proxy_symbol = "SPY"
```

## 添加新策略

```python
from strategy.base import BaseStrategy, StrategyParams

class MyParams(StrategyParams):
    ma_period: int = 20

class MyStrategy(BaseStrategy):
    params: MyParams

    def calculate_indicators(self, df):
        df["Signal"] = 0
        df.loc[..., "Signal"] = 1
        return df

    @property
    def min_bars(self) -> int:
        return 20
```

需要同步修改 3 处：
1. `strategy/__init__.py` — 添加到 `STRATEGY_MAP`
2. 策略类定义 `grid` 属性（优化器自动发现）
3. `watchlist.toml` — 添加 `[strategy.xxx]` 参数段

## 接入券商

```python
from broker.futu import FutuBroker
trader = LiveTrader(broker=FutuBroker(host="127.0.0.1", port=11111))
trader.run()
```

实现 `Broker` 抽象接口即可接入任意券商。
