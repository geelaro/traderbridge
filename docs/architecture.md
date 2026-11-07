# Traderbridge 架构

> 截至 2026-11-07 的项目架构与设计决策记录。代码持续演进,如本文与
> 代码不一致请以代码为准并同步更新本文档。

## 项目定位

**个人量化风险管理 + 决策辅助平台**(原名 mytrader,2026-05-31 改名)。
不替用户下单,提供"海图":数据 → 策略 → 回测 → 风险量化 → 实盘桥梁 →
告警 + 报告。

---

## 一、规模数字

| 维度 | 数据 |
|------|------|
| 总 commit | 170+(从 2023 mytrader 时代起算) |
| 业务代码 | **~19,000 行** Python(不含测试) |
| 测试代码 | **68 个测试文件 / 1411 passed / 2 skipped**(2026-11-07) |
| 覆盖率 | **79.6%**(2026-11-07;排除 tests/dashboard/scripts) |
| Dashboard tab | **11 个** |
| 数据源 | 6 个(sina_us / tencent / cboe / yahoo_chart / yahoo_realtime / futu) |
| 策略库 | 13 个(9 个 active + 3 个弃用 + 1 个 ensemble) |
| 分析模块 | 22 个独立 analysis/*.py |

---

## 二、分层架构

严格单向依赖,无循环引用:

```
┌─────────────────────────────────────────────────────────────────┐
│  dashboard/    14 文件 · 3139 行    Streamlit 展示层             │
│                11 个 tab,纯渲染逻辑,不持有业务状态              │
├─────────────────────────────────────────────────────────────────┤
│  scripts/      4 文件 · 364 行     Cron / 命令入口              │
│                daily.py / weekly_risk_report.py                 │
├─────────────────────────────────────────────────────────────────┤
│  analysis/     22 文件 · 5299 行   ★ 风险/业绩分析核心          │
│                纯计算,无 I/O 副作用                              │
│                VaR / Brinson / EVT / Drawdown 等                │
├─────────────────────────────────────────────────────────────────┤
│  strategy/     16 文件 · 2051 行   13 个量化策略                │
│                BaseStrategy + ChandelierTrailingExit Mixin +    │
│                StrategyEnsemble                                  │
├─────────────────────────────────────────────────────────────────┤
│  engine/       5 文件 · 2525 行    回测引擎(单标 + 组合)         │
│                BacktestEngine / PortfolioBacktest / Optimize    │
├─────────────────────────────────────────────────────────────────┤
│  live/         6 文件 · 1146 行    实盘桥梁                     │
│                LiveTrader / RiskController / OrderManager /     │
│                Kill Switch / RiskAlerter / position_stops       │
├─────────────────────────────────────────────────────────────────┤
│  broker/       4 文件 · 960 行     券商抽象                     │
│                Broker(ABC) / MockBroker / FutuBroker            │
├─────────────────────────────────────────────────────────────────┤
│  data/         7 文件 · 2081 行    数据管线                     │
│                CacheManager(SQLite) / DataProvider /            │
│                6 数据源 / 实时 VIX                              │
├─────────────────────────────────────────────────────────────────┤
│  utils/        12 文件 · 1787 行   横切关注点                   │
│                notify / sizing / risk / market_state / env /    │
│                logging / signal_gate / metrics                  │
└─────────────────────────────────────────────────────────────────┘

依赖方向: dashboard → analysis ← engine → live → broker
                                          ↓
                                       data ← utils
```

---

## 三、数据流(从市场到决策)

```
                     ┌─────────────────────────────┐
                     │  外部数据源(6 个)            │
                     │  sina_us / tencent / cboe / │
                     │  yahoo (chart + realtime) / │
                     │  futu OpenD                  │
                     └──────────┬──────────────────┘
                                │
                    ┌───────────▼──────────────┐
                    │   data/sources.py        │
                    │   + apply_us_splits()    │
                    │   + _yahoo_session /     │
                    │     _realtime_session    │
                    │     (隔离避免限流)        │
                    └───────────┬──────────────┘
                                │
                ┌───────────────▼─────────────┐
                │   DataProvider              │
                │   失败链 sina→tencent→yahoo │
                │   _check_cross_source_drift │
                │   持久化 source cooldown   │
                └───────────────┬─────────────┘
                                │
                  ┌─────────────▼────────────┐
                  │  CacheManager (SQLite)   │
                  │  ohlcv_daily / signal_   │
                  │  history / trade_pnl /   │
                  │  alert_history / ops_log │
                  └─────────────┬────────────┘
                                │
                        ┌───────┴────────┐
                        │                │
                ┌───────▼──────┐  ┌──────▼────────┐
                │  analysis/   │  │  strategy/    │
                │  (22 分析)  │  │  (13 策略)    │
                └───────┬──────┘  └──────┬────────┘
                        │                │
                        └────────┬───────┘
                                 │
                  ┌──────────────▼─────────────┐
                  │  engine/                    │
                  │  BacktestEngine / Portfolio │
                  └──────────────┬─────────────┘
                                 │
                  ┌──────────────▼─────────────┐
                  │  live/                      │
                  │  Daemon → OrderManager      │
                  │  ↓ RiskController          │
                  │  ↓ RiskAlerter → Feishu    │
                  │  ↓ KillSwitch              │
                  └──────────────┬─────────────┘
                                 │
                          ┌──────▼──────┐
                          │   broker/    │
                          │   Mock/Futu  │
                          └──────────────┘
```

---

## 四、风险/业绩分析能力 — 与专业平台对标

参考:Aladdin / Barra / Bloomberg POMS。

### 风险测量(Ex-ante)

| 能力 | 实现 | 模块 |
|------|------|------|
| Historical VaR | ✅ | `analysis/var.py` |
| Parametric VaR | ✅ | `analysis/var.py` |
| Expected Shortfall (CVaR) | ✅ | `analysis/var.py` |
| EVT 尾部估计(GPD POT) | ✅ | `analysis/evt.py` |
| 历史场景压力测试(5 场景) | ✅ | `analysis/stress.py` |
| Marginal VaR / Component VaR | ✅ | `analysis/risk_decomposition.py` |
| Risk Parity 权重求解 | ✅ | `analysis/risk_decomposition.py` |
| What-If 假设调仓预览 | ✅ | `analysis/what_if.py` |
| Monte Carlo 模拟 | ✅ | `analysis/monte_carlo.py` |
| Greeks(期权) | N/A | 个人量化不涉及期权 |
| ALM / 信用 / FX | N/A | 单一币种 / 现金股票 |

### 风险测量(Ex-post)

| 能力 | 实现 | 模块 |
|------|------|------|
| Sharpe / Sortino | ✅ | `analysis/risk_metrics.py` |
| Calmar / MAR / Omega | ✅ | `analysis/risk_metrics.py` |
| Pain Index / Pain Ratio | ✅ | `analysis/risk_metrics.py` |
| Information Ratio | ✅ | `analysis/risk_metrics.py` |
| Underwater Curve | ✅ | `analysis/drawdown.py` |
| Drawdown Episodes | ✅ | `analysis/drawdown.py` |
| Time-to-Recover 分布 | ✅ | `analysis/drawdown.py` |
| MaxDD / Pain Index | ✅ | 共享 |

### 结构与归因分析

| 能力 | 实现 | 模块 |
|------|------|------|
| HHI / Effective N | ✅ | `analysis/concentration.py` |
| Sector Concentration / Sector HHI | ✅ | `analysis/concentration.py` |
| Correlation HHI | ✅ | `analysis/concentration.py` |
| Correlation Clustering | ✅ | `analysis/correlation_analysis.py` |
| Effective Bets(PCA) | ✅ | `analysis/correlation_analysis.py` |
| 6 因子归因(Jensen α + Newey-West HAC) | ✅ | `analysis/factor_attribution.py` |
| Brinson 业绩归因 | ✅ | `analysis/brinson.py` |
| Rolling α / α 衰减 | ✅ | `analysis/rolling_alpha.py` |
| Forward Return 信号有效性 | ✅ | `analysis/forward_return.py` |
| 参数鲁棒性 / 成本敏感性 | ✅ | `analysis/param_robustness.py` `cost_sensitivity.py` |

### 告警与决策辅助

| 能力 | 实现 | 模块 |
|------|------|------|
| Risk Light(SPY MA200 + ADX + VIX) | ✅ | `analysis/risk_monitor.py` |
| Realtime VIX(Yahoo spark/chart) | ✅ | `data/realtime.py` |
| 风险告警状态机(3 类) | ✅ | `live/risk_alerts.py` |
| Alert History 审计 | ✅ | `data/cache.py` StateStore |
| Kill Switch(手动 + 双确认) | ✅ | `live/kill_switch.py` |
| 风险报告自动生成(9 section) | ✅ | `analysis/risk_report.py` |
| Realized vs Unrealized PnL 拆分 | ✅ | `analysis/pnl_breakdown.py` |

### 缺口

| 项 | 优先级 | 备注 |
|----|--------|------|
| Days-to-Liquidate(流动性风险) | 低 | 当前 watchlist 全大盘,DTL 接近 0 |
| Style Drift Detection | 低 | 需 ≥6 个月实盘数据才有意义 |
| 自动 PDF 报告 | 低 | 现已支持 Markdown + Feishu,PDF 需要 weasyprint 重依赖 |
| 税务批次会计(FIFO/LIFO) | 低 | 富途自带 |

**对标结论**: 风险测量层覆盖专业平台 **~80%**。

---

## 五、关键设计决策

### 1. mytrader → traderbridge 改名 + 定位转向

**Why**: 因子归因 + B&H 对比发现 weekly_macd_kdj 7 年 CAGR 12.9% 跑输等权 B&H 32.7%(差 20%),但 Sharpe 1.08 vs 0.61 取胜。典型主动管理悖论。

**决定**: 把 mytrader 定位从"追求超额收益"改为"**风险管理 + 决策辅助**"。

**影响**: Dashboard tab 命名调整;策略选型逻辑变化(看 Sharpe 而非 CAGR);
Brinson 实证显示 watchlist 有效赌注 1.01(12 个持仓实际只是 1 个 Tech 赌注),
进一步确认定位正确。

### 2. 拆股调整统一到所有 US 源

**Why**: NVDA 2023-12-26 出现 909% 单日跳变,排查发现 SinaUSSource 没应用
`splits.json`,而 Tencent 应用了。混源后产生 10x 价格 cliff。

**决定**: 抽出 `apply_us_splits()` 共享辅助,三个 US source 末尾统一调用。

**影响**: 一次性运维清掉 11914 行污染 cache。新增数据源**必须**调用
`apply_us_splits()`,否则破坏跨源一致性。

### 3. 实时 VIX 使用独立 session

**Why**: `_yahoo_session()` 预热时调 `fc.yahoo.com` 设置 cookie,这个 cookie
让 Yahoo 把后续所有请求识别为同一 client,数秒内触发 spark/chart 严格限流。
不带 cookie 的 fresh session 调用同样 endpoint 能稳定拿数据。

**决定**: realtime 用独立的 `_realtime_session()`(仅 UA + trust_env,无
fc.yahoo.com),historical 继续用 `_yahoo_session()`。

**影响**: 多了 50 行代码,但绕开了 Yahoo 限流。任何新的 Yahoo 实时调用
**必须**复用 `_realtime_session()`,不能用 `_yahoo_session()`。

### 4. VIX > 50 不自动触发 Kill Switch

**Why**: CBOE VIX 36 年历史里 VIX > 50 只发生 5 次,之后 SPY 250 日平均
涨 **+44.6%**(vs 基线 +11.4%)— 是**抄底信号**而非清仓信号。任何基于
VIX/回撤的自动平仓触发器都会反向伤害。

**决定**: Kill Switch 完全手动,无任何阈值绑定。Dashboard 双确认 UI
(输 CONFIRM + 必填 reason)+ Dry Run 预演。

**影响**: 防止用户构建反向工具。所有未来的"自动响应"功能都应该先做
实证验证(类似 VIX 这种)再设计触发条件。

### 5. Streamlit st.session_state 持久化

**Why**: Streamlit 按按钮会重渲染整个 page。如果用局部变量持有
"生成的报告" 状态,点"推送飞书"时局部变量已经丢失,导致提前 return。

**决定**: Dashboard 跨多 step 的状态必须用 `st.session_state` 缓存。

**影响**: 所有 dashboard tab 涉及"生成 + 操作"的两步流程都要审查
session_state 用法。

### 6. NotifyLogHandler 不自动安装

**Why**: 自动把 logger.error 转发飞书是危险的 — 一旦某个 section 失败,
Python 默认会用 ERROR 级别 log,推送大量噪音卡片。

**决定**: `install_notify_log_handler` 仅在 `live_trader.py` 主进程显式启用,
其他地方(dashboard / scripts / analysis)用 `logger.warning` 而非
`logger.error` / `logger.exception`,让错误进入 stderr 而非飞书。

**影响**: 风险报告 `_build_*` 失败时降级到 WARNING 级别 + section 内显示
具体 exception type 给用户调试。

---

## 六、技术债清单

按优先级:

| 项 | 严重度 | 备注 |
|----|--------|------|
| broker 接口缺 `list_open_orders` | 中 | Kill Switch 现在只清持仓,不取消挂单 |
| `_yahoo_session` 与 `_realtime_session` 两套 | 低 | 必要的隔离,但模块边界稍碎 |
| trade_pnl 表无 strategy 字段 | 低 | Brinson 等无法按策略再归因 |
| 数据源各 API 超时/解析策略不一致 | 中 | 已有跨源回退、持久化 15 分钟 source cooldown 和标的 retry window;缺统一 endpoint retry/backoff 策略 |
| logs/ 文件没有 rotation | 低 | 长跑可能堆爆 |
| dashboard/ 排除在覆盖率统计外 | 低 | 有少量 render smoke tests,但 UI 行为覆盖有限 |
| `@st.cache_resource` 单例 cache/provider | 低 | multi-user 部署冲突,单用户 OK |
| 11 标 watchlist 全 Tech,有效赌注 1.01 | **高** | **业务层问题**,需要 user 自己调整 watchlist |

### 分析快照持久化

`dashboard/decision_summary.py` 通过 `CacheManager` 将组合风险派生快照写入
`portfolio_risk_snapshots`,与 daemon/Kill Switch 使用的 `risk_state` 隔离。
变化量按最近一条更早的快照计算,因此周末或中间未打开仪表盘时仍可比较。
这张表仅供分析界面读取,不作为交易状态或下单条件。

`data/source_health.py` 将拆股文件指纹保存在 `app_metadata`,与交易风控键
`risk_state` 分开;schema v6 会迁移已有的 `splits_hash` 值。

---

## 七、生产就绪评估

| 维度 | 状态 | 评语 |
|------|------|------|
| 代码质量 | ✅ | 1411 passed / 2 skipped;79.6% 覆盖率(2026-11-07) |
| 错误处理 | ✅ | 每个 section/source 独立 try/except,不级联崩溃 |
| 数据完整性 | ✅ | 跨源校验 + 拆股统一 + drift warning |
| 告警审计 | ✅ | alert_history + ops_log + JSON 结构化日志 |
| CI/CD | ✅ | GitHub Actions + Codecov + badge |
| 文档 | ⚠ | README + AGENTS + ROADMAP + 本架构文档,**但缺操作 runbook** |
| 运维 | ⚠ | docker-compose 有,缺 Prometheus metrics 端点 |
| 实盘安全 | ⚠ | Kill Switch + RiskController 都有,**但 FutuBroker 路径未真实金验证** |
| 可扩展性 | ✅ | Broker ABC / Source ABC / Strategy ABC,新增不痛 |

**结论**: 可用于 **dry-run 和小额实盘**。大额实盘前建议 Futu 模拟盘
连跑 1-2 周验证 OrderManager / Kill Switch / 资金链对账。

---

## 八、最大成就(按价值排序)

1. **风险量化套件 + Brinson + PnL 拆分** — 业绩与风险归因双闭环
2. **NVDA / GOOG 拆股 bug 修复** — 数据基础设施一致性
3. **实时 VIX 联通** — 风险灯从 T+1 升级到 T+0
4. **Kill Switch 基于实证设计**(VIX>50 是抄底信号)— 防止反向工具
5. **风险报告自动化** — 9 个分析模块的散点连成图

---

## 九、剩余工作

### Phase 7(长期方向,有门槛)

| 项 | 门槛 |
|----|------|
| 真实因子分析框架 | ≥1 年实盘交易记录 |
| 日内策略 | FutuOpenD Level-2 数据权限 |
| ML 辅助信号 | ≥2000 笔历史交易 |
| 多账户管理 | broker 接口支持多连接 |
| Style Drift Detection | ≥6 个月实盘数据 |

### 业务层(用户决定)

- 把全 Tech watchlist 改造成跨行业,有效赌注从 1.01 提到 3+
- 实盘运行 1-2 周收集真实数据
- 配置 cron 跑 weekly_risk_report.py 每周一推送

---

## 十、相关文档

- [README.md](../README.md) — 项目介绍 + 快速开始
- [ROADMAP.md](../ROADMAP.md) — 开发进度跟踪
- [AGENTS.md](../AGENTS.md) — 模块速查 + 设计原则
- [strategy/STRATEGY_GUIDE.md](../strategy/STRATEGY_GUIDE.md) — 策略库使用指南
