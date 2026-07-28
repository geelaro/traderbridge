# traderbridge 整改与开发计划

> 原名 `mytrader`, 2026-05-31 改名为 traderbridge (体现"决策辅助/风险管理"定位)

> 审查日期: 2026-05-23 ｜ 基准: ~16,855 行 Python, 76 源文件, 485 测试
>
> 审查来源: 量化交易架构审查报告 (A) + 架构审查报告 (B) 合并版

---

## 总览

```
阶段一  致命修复     第 1 周    ████████████████   (5 项, 已完成)
阶段二  架构矫正     第 2-3 周  ████████████████   (5/5 项, 已完成)
阶段三  质量加固     第 4 周    ████████████████   (6/6 项, 已完成)
阶段四  MTF 框架     第 5-6 周  ████████████████   (4/4 项, 已完成)
阶段五  策略与组合   第 7-8 周  ████████████████   (4/4 项, 已完成)
阶段六  运维与可观测 第 9-10 周 ████████████████   (6/6 项, 已完成)
阶段七  长期方向     第 11 周+  ░░░░░░░░░░░░░░   (5 方向)
```

---

## 阶段一：致命修复（第 1 周）

### P0-1 统合回测/实盘仓位公式

- [x] 部分完成 — `calc_risk_budget_qty()` 已统一，**但 max_position_pct 上限仍分裂**
- **文件:** `engine/trader.py:106` `live/risk_controller.py:163`
- **问题:** 回测用 `capital × risk_per_trade / (ATR × risk_atr_mult)`，实盘用 `capital × base_risk_pct / (ATR × 2) × vol_scalar`。两者差异 2-5 倍，参数优化结果无法迁移到实盘
- **方案:** 提取统一函数到 `utils/risk.py` 或新建 `utils/sizing.py`，两端共用
- **2026-05-31 复评发现遗留问题:** risk-budget 公式已统一为 `utils/sizing.calc_risk_budget_qty()`，
  但 `max_position_pct` 上限仍分裂：策略层 (`MACDKDJParams.max_position_pct=0.95`) 与
  RiskLimits (`max_position_pct=0.30`) 同时存在，回测走 95% / 实盘走 30%，仓位差约 3 倍。
  剩余统一工作记为 **P2-9**（架构性，待单独立项）。

### P0-2 peak_equity 恢复加日期校验

- [x] 完成
- **文件:** `live/risk_controller.py:59`
- **问题:** 当前无条件从 SQLite 恢复 peak_equity。若历史峰值高但今日已大幅回撤，恢复旧值会误触发市场熔断（max_total_drawdown_pct = 30%）
- **方案:** 恢复时校验 `stored_date == today`，跨日则重新初始化为当日 equity

### P0-3 Config YAML 异常改为 warning + fallback

- [x] 完成
- **文件:** `config.py:155-159`
- **问题:** `except Exception: return` 静默丢弃解析错误。配置 YAML 缩进/格式写错时完全无感知，系统以默认值运行
- **方案:** `logger.warning("config.yaml 解析失败: %s", e)` 后继续使用默认值

### P0-4 Monte Carlo n_sims 参数覆盖

- [x] 完成
- **文件:** `analysis/monte_carlo.py:74`
- **问题:** 函数签名声明了 `n_sims` 参数，但第 74 行无条件覆盖为 `n_sims = 2000`。调用者无论传入任何值都被忽略
- **方案:** `n_sims = n_sims or 2000`

### P0-5 删除/归档 3 个失效策略

- [x] 完成
- **文件:** `strategy/` `watchlist.toml` `engine/optimize.py`
- **影响范围:**
  - `bollinger_mean_reversion` — 已知零交易
  - `bollinger_squeeze` — 已知零交易
  - `enhanced_macd` — 已知过拟合（0 星）
- **方案:**
  1. 文件移至 `archive/strategies/` 保留历史引用
  2. 从 `strategy/__init__.py` 的 `STRATEGY_MAP` 和 imports 中移除
  3. 从 `engine/optimize.py` 的 `PARAM_GRIDS` 中移除
  4. 从 `watchlist.toml` 的 monitor 列表中移除引用

---

## 阶段二：架构矫正（第 2-3 周）

### P1-1 消除尾随止损 ×6 重复

- [x] 完成
- **文件:** `strategy/trend_follower.py` `strategy/atr_breakout.py` `strategy/donchian_breakout.py` `strategy/bollinger_squeeze.py` `strategy/turtle_trading.py` `strategy/daily_macd_kdj.py` `strategy/base.py`
- **问题:** Chandelier 尾随止损逻辑 (`price <= highest - trail_atr_mult × ATR`) 在 6 个策略中一字不差重复
- **方案:** 提取 `ChandelierTrailingExit` Mixin 到 `base.py`，各策略仅需声明 `trail_atr_mult` 参数

### P1-2 合并 weekly / daily_macd_kdj

- [x] 完成
- **文件:** `strategy/weekly_macd_kdj.py` `strategy/daily_macd_kdj.py`
- **问题:** 两者 80% 代码重复。差异仅：周/日重采样 + ATR 尾随止损
- **方案:** 合并为单类 `MACDKDJStrategy`，参数控制 `freq="W"|"D"` 和 `use_atr_stop=True|False`。旧文件改为 re-export，`WeeklyMACD_KDJ` / `DailyMACD_KDJ` 保持向后兼容

### P1-3 消除 MarketState 循环依赖

- [x] 完成
- **文件:** `utils/market_state.py:43`
- **问题:** `from strategy import STRATEGY_MAP` 导入了全部 10 个策略，策略文件可能间接依赖 market_state，形成隐式循环
- **方案:** `is_trend_strategy()` / `is_mean_reversion_strategy()` 接收 `regime_map` 参数，由 `signal_gate.py` 注入

### P1-4 Tencent 单源加 fallback 链

- [x] 完成
- **文件:** `data/sources.py` `data/protocol.py` `data/cache.py`
- **问题:** `SOURCE_PRIORITY["us"] = ["tencent"]` — 腾讯 API 为唯一 US 数据源，无 SLA，随时可能变动或限流
- **方案:**
  1. 新增 `SinaUSSource` — 新浪美股日K，回溯至 1984 年，实测可用
  2. 新增 `YahooChartSource` — Yahoo v8 chart API + cookie 流，作为第三级回退
  3. 回退链: `sina_us → tencent → yahoo_chart`
  4. 彻底移除 `yfinance` 依赖和 `YFinanceSource`
  5. `missing_ranges()` 新增缺口合并逻辑，避免几十个小缺口触发级联请求
- **变更:** +208/-75 行源码, +167 行测试, 576 passed

### P1-5 CacheManager 按职责拆分

- [x] 完成
- **文件:** `data/cache.py` (~420 行)
- **问题:** `CacheManager` 管理 6 种数据类型（OHLCV、signal_history、risk_state、entry_prices、trade_pnl、ops_log），`init_schema()` 每次连接执行 11 CREATE TABLE + 6 ALTER TABLE + 1 CREATE INDEX
- **方案:** 拆为 3 个类 + 1 个 facade：
  - `OhlcvCache(_CacheBase)` — OHLCV load/save/date_range/missing_ranges
  - `StateStore(_CacheBase)` — risk_state / entry_prices / trade_pnl
  - `OpsLogger(_CacheBase)` — ops_log / order_log / slippage_log
  - `CacheManager(OhlcvCache, StateStore, OpsLogger)` — 全功能向后兼容 + signal_history

---

## 阶段三：质量加固（第 4 周）

### P2-1 回测引擎主循环重构

- [x] 完成
- **文件:** `engine/trader.py` `run()` (~100 行)
- **方案:** 提取状态机子方法：`_process_pending_order()` / `_check_exit_signal()` / `_check_entry_signal()` / `_apply_stop_cooldown()`

### P2-2 组合回测主循环拆分

- [x] 完成
- **文件:** `engine/portfolio.py` `run()` (~180 行)
- **方案:** 拆分子方法：`_handle_pending_order()` / `_check_leg_exit()` / `_check_leg_entry()`

### P2-3 RSI 计算提取到 base.py

- [x] 完成
- **文件:** `strategy/enhanced_macd.py` `strategy/bollinger_mean_reversion.py` `strategy/base.py`
- **问题:** RSI 计算逻辑在两处独立实现，参数和边界处理可能不一致
- **方案:** `compute_rsi()` 加入 `base.py` 帮助函数组，两策略各 -5 行

### P2-4 硬编码分裂调校外置

- [x] 完成
- **文件:** `data/sources.py` TencentSource
- **问题:** AAPL/NVDA/TSLA/AMZN/GOOGL 的分裂调整因子硬编码在源码中，每次拆股需要手动改代码
- **方案:** 移至 `data/splits.json` 配置文件，`_load_splits()` 自动加载

### P2-5 金标测试扩展到全策略

- [x] 完成
- **文件:** `tests/test_golden.py`
- **新增覆盖:** `atr_breakout` `donchian_breakout` `daily_macd_kdj` `weekly_macd` `MACDKDJStrategy`
- **参数:** seed=42, 300 bars, $10k capital, tolerance ±0.01, 17 tests total

### P2-6 实盘 BUY 后刷新风控

- [x] 完成
- **文件:** `live_trader.py`
- **问题:** `check_global()` 当前仅在 SELL 后调用，BUY 后未刷新。总敞口超限检查滞后一个循环周期
- **方案:** BUY/SELL 成交后统一调用 `check_global()`

---

## 阶段四：Multi-Timeframe 框架（第 5-6 周）

> 中期最有价值改进，从根本提升信号质量

### M-1 重构 BaseStrategy 接口

- [x] 完成
- **文件:** `strategy/base.py` 及所有策略
- **方案:** `calculate_indicators(df, df_weekly=None)` 策略可同时接收日线和周线
- **向后兼容:** 默认 `df_weekly=None`，现有策略无需改动

### M-2 weekly_macd_kdj 迁移为示范

- [x] 完成
- **文件:** 合并后的 `strategy/macd_kdj.py`
- **方案:** freq="W" + df_weekly 时，指标在周线计算后 ffill 映射回日线时间轴。freq="D" 保持原逻辑

### M-3 SignalScanner 跨频率对齐

- [x] 完成
- **文件:** `utils/signal_scanner.py`
- **方案:** `_fetch_weekly()` 自动重采样 → `calculate_indicators(df, df_weekly=...)` → TypeError fallback 兼容旧策略

### M-4 DataProvider 按需喂多频率

- [x] 完成
- **文件:** `data/provider.py`
- **方案:** `get_data(symbol, freqs=["D","W"])` 返回 `{"D": df, "W": df_weekly}`，内部缓存重采样

---

## 阶段五：策略与组合增强（第 7-8 周）

### E-1 StrategyEnsemble 信号组合

- [x] 完成
- **文件:** 新建 `strategy/ensemble.py`
- **方案:** 按市场状态自动加权投票。StrategyEnsemble 继承 BaseStrategy，封装多个子策略 + MarketStateClassifier

### E-2 批量仓位分配（替代顺序处理）

- [x] 完成
- **文件:** `live/order_manager.py`
- **方案:** 三阶段：①处理卖单 → ②收集买入候选 → ③等风险均分 capital/n 批量下单

### E-3 DataQuality 层

- [x] 完成
- **文件:** 新建 `data/quality.py`
- **功能:** flag_missing / flag_price_jumps / flag_non_trading / validate_ohlcv / quality_report / clean

### E-4 参数滚动优化接入实盘

- [x] 完成
- **文件:** `daily.py` + `utils/env.py`
- **方案:** `--optimize` 标记触发 walk-forward。OOS Sharpe 下滑 >30% 自动更新 watchlist.toml。新增 save_toml()

---

## 阶段六：运维与可观测性（第 9-10 周）

### O-1 Docker Compose

- [x] 完成
- **文件:** 新增 `docker-compose.yml`
- **内容:**
  - `services: traderbridge` + `futu-opend`（FutuOpenD 容器）
  - 环境变量注入: `FEISHU_*` `TRADERBRIDGE_DB` (legacy `MYTRADER_DB` 也接受) `FUTU_HOST`
  - 持久化卷: `trading_data.db` `logs/` `reports/`

### O-2 daemon 健康检查

- [x] 完成
- **文件:** `live_trader.py` 内置 HTTP server
- **方案:** `GET /health` → `{"status":"ok","last_tick":"2026-05-23T14:30:00Z","paused":false}`
- **Docker:** `HEALTHCHECK --interval=30s CMD curl -f http://localhost:8080/health`

### O-3 结构化日志

- [x] 完成
- **文件:** `utils/logging.py`
- **方案:** 统一日志格式为 JSON 行（`{"ts":"...","level":"INFO","logger":"live","event":"trade_filled",...}`），供 Loki / ELK 解析

### O-4 飞书日报增强

- [x] 完成
- **文件:** `utils/notify.py`
- **当前:** 简单信号/交易通知
- **方案:** 日终汇总卡片：
  - 昨日 PnL 归因（按策略分解 / 按标的分解）
  - 风控事件（熔断次数、滑点超标次数、连亏计数）
  - 当日预扫描信号预览
  - 当前持仓摘要 + 浮盈/浮亏

### O-5 DB migration 版本化

- [x] 完成
- **文件:** `data/cache.py`
- **方案:** 引入 `schema_version` 表 + 版本化迁移函数列表。替换现有 `try: ALTER TABLE except: pass` 模式

### O-6 DB 路径绝对化

- [x] 完成
- **文件:** `data/cache.py`
- **问题:** `os.environ.get("MYTRADER_DB", "trading_data.db")` 相对路径依赖 CWD。cron 触发 `daily.py` 时工作目录可能不一致
- **方案:** `Path(...).resolve()` 或基于 `PROJECT_ROOT` 生成绝对路径

---

## 阶段七：长期方向（第 11 周+）

### 因子分析框架

- [ ] 开始研究
- **门槛:** 需积累 ≥1 年实盘交易记录
- **功能:** Beta 暴露分解、Rank IC 分析 (按日/周)、因子相关性热力图、IC decay 曲线

### 日内策略

- [ ] 开始研究
- **门槛:** FutuOpenD Level-2 数据权限
- **功能:** 分钟级数据管道、MomentumBreakout 策略、TWAP/VWAP 算法订单执行

### ML 辅助信号

- [ ] 开始研究
- **门槛:** ≥2000 笔历史交易记录
- **方案:** LightGBM 预测未来 N 日方向（特征=现有指标列）→ 输出置信度权重 → 与规则策略加权融合。规则策略保持为基线

### 黑天鹅预案

- [ ] 开始研究
- **功能:**
  - 2020-03 / 2022-06 式闪崩自动识别并强制平仓
  - 极端波动（VIX > 40）自动降低持仓至 30% 上限
  - 可选：深度虚值 put 对冲信号

### 多账户管理

- [ ] 开始研究
- **门槛:** broker 接口需支持多连接
- **功能:** 多 Futu 账户同时管理、跨账户风控聚合、账户间资金调拨

---

## 专业风险管理平台对标 (2026-06-02 全部完成 ✅)

按 Aladdin / Barra / Bloomberg POMS 对标, 12 项中 9 项完成, 2 项跳过
(门槛未到 / 优先级低), 1 项主动放弃 (VIX>50 自动平仓被实证否决).

### 第一批 — 风险测量基础 ✅

- ✅ **VaR / Expected Shortfall** — `analysis/var.py` Historical / Parametric / CVaR, 1d 95%/99%
- ✅ **历史场景压力测试** — `analysis/stress.py` 2008/2018/2020/2022/2015 五场景
- ✅ **集中度指标** — `analysis/concentration.py` HHI / Top-N / Effective N / 行业 / 相关性

### 第二批 — 实盘相关

- ✅ **Kill Switch / 紧急平仓** — `live/kill_switch.py` 手动触发 + 双确认 + 飞书.
  自动 VIX>50 触发**被实证否决**: CBOE 36 年史 5 次 VIX>50 后 SPY 250 日均
  +44.6% (vs 基线 +11.4%), 是抄底信号而非清仓信号
- 跳过 **流动性风险 Days-to-Liquidate** — 当前 watchlist 全大盘, DTL 接近 0;
  未来加中小盘配置时再做

### 第三批 — 业绩分析深化 ✅

- ✅ **Risk-Adjusted Metrics 深化** — `analysis/risk_metrics.py`
  Sortino / Calmar / MAR / Omega / Pain Index / Information Ratio
- ✅ **Drawdown Analytics 深化** — `analysis/drawdown.py`
  Underwater curve / Episodes / Time-to-recover (median/p75/p95)
- ✅ **Brinson Performance Attribution** — `analysis/brinson.py`
  配置 / 选股 / 交互三效应分解, vs SPDR Sector ETFs
- ✅ **Realized vs Unrealized PnL 拆分** — `analysis/pnl_breakdown.py`
  Dashboard 新 tab + 7d/30d/90d/1y/YTD 区间

### 第四批 — 报告与合规

- ✅ **风险报告自动生成** — `analysis/risk_report.py` + `scripts/weekly_risk_report.py`
  9 section 综合周报, Markdown + 飞书富文本卡片; Dashboard "📑 风险报告" tab 手动触发
- 跳过 **税务批次会计 (FIFO/LIFO)** — 富途自带, 优先级低
- 跳过 **Style drift detection** — 需 ≥6 个月实盘数据才有意义, 现在做空跑

### Bonus — 同期完成的非对标项

- ✅ **Marginal / Component VaR** — `analysis/risk_decomposition.py`
  欧拉分解, Risk Parity 权重求解, Top 风险贡献者识别
- ✅ **What-If 假设调仓** — `analysis/what_if.py`
  调仓前预演 VaR / HHI / sector 变化, 5 个预设方案
- ✅ **EVT 尾部估计** — `analysis/evt.py`
  GPD POT 拟合, 99.5%/99.9% 高分位 VaR 外推
- ✅ **相关性聚类 / Effective Bets** — `analysis/correlation_analysis.py`
  层次聚类 + PCA 特征值分解, watchlist 揭露"12 持仓实际只是 1.01 个独立赌注"
- ✅ **实时 VIX 旁路** — `data/realtime.py` Yahoo spark/chart 端点
  独立 session 绕开 fc.yahoo.com cookie 触发的限流
- ✅ **风险灯 + 三类告警状态机** — `analysis/risk_monitor.py` + `live/risk_alerts.py`
- ✅ **告警历史审计** — `data/cache.py` alert_history 表 + Dashboard 时间线
- ✅ **拆股调整统一** — `data/sources.py:apply_us_splits()`
  三个 US source 共享, 修复 NVDA / GOOG cache 跳变 bug

---

## 2026-06-02 风险管理 sprint (27 commit 单日)

| Commit | 类型 | 内容 |
|--------|------|------|
| `c5f6d0e` | feat | VaR + Stress + Concentration 套件 |
| `47e3a76` | feat | Sortino/Calmar + Drawdown 深度分析 |
| `e29a856` | feat | Marginal VaR + What-If 假设分析 |
| `9b42490` | feat | EVT 尾部估计 + 相关性聚类 |
| `48f4a8d` | fix | 统一拆股调整到所有 US 源 (修 NVDA/GOOG) |
| `460458b` | feat | 实时 VIX 旁路接入 |
| `424b74f` | fix | Yahoo session 启用 trust_env |
| `433316f` | fix | 实时 VIX 加 HTML scrape fallback |
| `2bea382` | fix | 移除 HTML fallback (CDN 缓存 EOD) |
| `f5304d7` | fix | 实时 VIX 独立 session 绕开限流 |
| `d16cfa5` | feat | Brinson 业绩归因 |
| `e172899` | feat | Realized vs Unrealized PnL 拆分 |
| `b3d1642` | fix | SPDR Sector ETFs 加 Tencent code map |
| `d33dea5` | feat | Kill Switch 紧急平仓 (手动 + 飞书) |
| `52e5a53` | feat | 风险报告自动生成 |
| `daf38fa` | chore | Codecov coverage CI + badge |
| `728db26` | fix | 风险报告 section 失败日志降级 |
| `64e56f3` | fix | 风险报告 dashboard 推送飞书按钮 (session_state) |

测试 710 → **1043 用例**, 覆盖率 **75.9%**.

---

## 剩余待办 (Phase 7 长期方向)

都有门槛, 实盘数据 / 数据权限 / 训练样本未到位时做没意义.

---

## 优先级决策矩阵

```
             ┌── 高影响 ────┬── 中影响 ────┐
             │              │              │
低难度 ───── P0-1 公式对齐 │ P0-3 配置日志 │
             P0-4 MC 修复  │ P2-6 BUY刷新  │
             P0-5 删失效   │ P2-3 RSI提取  │
             ──────────────┼───────────────┤
中难度 ───── P1-4 多源链   │ P1-3 环依赖   │
             P1-1 止损去重 │ P2-1 引擎重构  │
             P1-2 KDJ合并  │ P2-5 金标扩展  │
             ──────────────┼───────────────┤
高难度 ───── P1-5 缓存拆分 │ M-1 MTF 框架  │
             ──────────────┤ E-1 策略组合   │
                           │ O-4 日报增强   │
```

## 进度记录

| 日期 | 阶段 | 项目 | 状态 | 备注 |
|------|------|------|------|------|
| 2026-05-23 | 阶段一 | P0-1 ~ P0-5 | 已完成 | 5/5 全绿, 571 passed, 0 failed |
| 2026-05-23 | 阶段一 | P0-1 统合回测/实盘仓位公式 | 完成 | 新增 utils/sizing.py, 两端共用 calc_risk_budget_qty() |
| 2026-05-23 | 阶段一 | P0-2 peak_equity 日期校验 | 完成 | 恢复逻辑移入 stored_date==today 分支内 |
| 2026-05-23 | 阶段一 | P0-3 Config YAML 异常警告 | 完成 | 改用 logging.warning() 记录解析错误 |
| 2026-05-23 | 阶段一 | P0-4 Monte Carlo 参数覆盖 | 完成 | n_sims = n_sims or 2000 |
| 2026-05-23 | 阶段一 | P0-5 失效策略移除 | 完成 | 从 STRATEGY_MAP/watchlist.toml 移除, 保留源文件供测试导入 |
| 2026-05-23 | 阶段二 | P1-4 数据源多链 | 完成 | SinaUSSource + YahooChartSource, 移除 yfinance, 回退链 sina_us→tencent→yahoo_chart, missing_ranges 缺口合并 |
| 2026-05-25 | 阶段二 | P1-1 尾随止损去重 | 完成 | ChandelierTrailingExit Mixin 提取到 base.py, 6 策略复用 |
| 2026-05-25 | 阶段二 | P1-2 KDJ合并 | 完成 | MACDKDJStrategy 统一类, freq="W"\|"D" + use_atr_stop, 旧文件re-export |
| 2026-05-25 | 阶段二 | P1-3 循环依赖 | 完成 | is_trend_strategy/is_mean_reversion_strategy 接收 regime_map 参数 |
| 2026-05-25 | 阶段二 | P1-5 缓存拆分 | 完成 | OhlcvCache / StateStore / OpsLogger + CacheManager facade |
| 2026-05-25 | 阶段三 | P2-1 回测引擎重构 | 完成 | 4 个子方法提取，run() 主循环 ~50 行 |
| 2026-05-25 | 阶段三 | P2-2 组合回测重构 | 完成 | _handle_pending_order / _check_leg_exit / _check_leg_entry |
| 2026-05-25 | 阶段三 | P2-3 RSI 提取 | 完成 | compute_rsi() 加入 base.py，双策略复用 |
| 2026-05-25 | 阶段三 | P2-4 分裂外置 | 完成 | splits.json + _load_splits() |
| 2026-05-25 | 阶段三 | P2-5 金标扩展 | 完成 | 5 策略新增 → 17 golden tests, 586 passed |
| 2026-05-25 | 阶段三 | P2-6 BUY刷新风控 | 完成 | check_global() 统一在 BUY/SELL 成交后调用 |
| 2026-05-25 | 阶段四 | M-1 BaseStrategy 接口 | 完成 | calculate_indicators(df, df_weekly=None) 多频签名 |
| 2026-05-25 | 阶段四 | M-2 macd_kdj MTF | 完成 | freq="W"+df_weekly → 周线指标 ffill 映射日线 |
| 2026-05-25 | 阶段四 | M-3 SignalScanner | 完成 | _fetch_weekly + TypeError fallback |
| 2026-05-25 | 阶段四 | M-4 DataProvider 多频 | 完成 | get_data(freqs=["D","W"]) |
| 2026-05-25 | 阶段五 | E-1 StrategyEnsemble | 完成 | 策略组合加权投票, MarketRegime 自适应权重 |
| 2026-05-25 | 阶段五 | E-2 批量仓位分配 | 完成 | 三阶段: 卖单→收集→等风险均分批量下单 |
| 2026-05-25 | 阶段五 | E-3 DataQuality 层 | 完成 | quality.py: flag/clean/validate 全套检查 |
| 2026-05-25 | 阶段五 | E-4 滚动优化接入 | 完成 | daily.py --optimize + save_toml |
| 2026-05-25 | 阶段六 | O-1 Docker Compose | 完成 | docker-compose.yml + HEALTHCHECK |
| 2026-05-25 | 阶段六 | O-2 健康检查 | 完成 | live_trader.py 内嵌 HTTP /health |
| 2026-05-25 | 阶段六 | O-3 结构化日志 | 完成 | JsonFormatter → JSON 行日志 |
| 2026-05-25 | 阶段六 | O-4 日报增强 | 完成 | daily_card() PnL归因+风控+持仓 |
| 2026-05-25 | 阶段六 | O-5 DB migration | 完成 | schema_version + _run_migrations() |
| 2026-05-25 | 阶段六 | O-6 DB路径绝对化 | 完成 | Path(...).resolve() |
| 2026-05-31 | 复评-P0 | S1 做空虚增买入力 | 完成 (`589756a`) | 引入 available_cash + short_margin_ratio=1.5 (Reg-T), 开多检查改用 available_cash |
| 2026-05-31 | 复评-P0 | S2 ensemble 死代码 + 索引错位 | 完成 (`589756a`) | reindex 到 df.index, long/short_mask 真冲突仲裁, member_weights 归一化除均值 |
| 2026-05-31 | 复评-P0 | S3 SPYMABreakout 死代码 | 完成 (`589756a`) | 移除 short 信号 + SHORT 出场 + N_day_low |
| 2026-05-31 | 复评-P0 | S4 默认策略 | 完成 (`589756a`) | run_backtest 默认 enhanced_macd → MACDKDJStrategy |
| 2026-05-31 | 复评-P0 | S5 turtle O(N²) → O(1) | 完成 (`589756a`) | 实例缓存 _cur_entry_atr/_cur_highest_high/_cur_lowest_low 增量更新 |
| 2026-05-31 | 复评-P1 | P1-3 peak_equity 跨日 | 完成 (`b5f3165`) | peak_equity + consecutive_losses 从 today 分支移出 |
| 2026-05-31 | 复评-P1 | P1-5 max_slippage_pct | 完成 (`b5f3165`) | 2% → 0.5% (50bp) |
| 2026-05-31 | 复评-P1 | P1-6 SignalGate 动态化 | 完成 (`b5f3165`) | 移除模块期固化, dataclass field + 懒加载 _build_regime_map |
| 2026-05-31 | 复评-P1 | P1-7 跨源校验 | 完成 (`b5f3165`) | _check_cross_source_drift: 重叠 close >1% 或边界 >20% warn |
| 2026-05-31 | 复评-P2 | P2-2/3/4/6/7/8/10 | 完成 (`c9f9eff`) | 公式精确化 / 缺口阈值 3→7 / health 刷新 / 融券利息 / ensemble typo warn / MTF min_bars×5 / 废弃清理 |
| 2026-05-31 | 复评-P3 | P3-1/2/4/5/7 | 完成 | Windows signal 兼容 / sector_map fallback warn / print-logger 约定文档化 / SQLite 读路径加锁 / 本表同步 |

## 复评后剩余待办

| 编号 | 题 | 状态 |
|------|----|------|
| P2-5 | PortfolioBacktest → PortfolioState 重构 | ✅ 完成 (`252a5e8`) |
| P2-9 | max_position_pct 组合回测与实盘对齐 | ✅ 完成 (`0c47f3b`) |
| P2-1 | 回测 vs 实盘信号时点对齐 | 📌 已立项, 暂不修 |

### P2-1 决策记录 (2026-05-31)

**决策**: 暂不动代码, 仅做文档化标注。

**理由**:
- 当前 watchlist 主力 `weekly_macd_kdj` 周线信号稀疏, 日内闪烁概率极低,
  实盘行为实际安全。
- 出场即时响应 (跳空止损保护) 是用户明确需要的实盘特性, 不能为对齐回测牺牲。

**未来触发修复的场景** (任一即必修):
1. 把 daily_macd_kdj / rsi2 / spy_ma_breakout 任一设为 active
2. 启动多因子模型 / Ensemble 实盘下单
3. 接入日内策略
4. 想让回测优化的参数可严格迁移到实盘

**修复方向草案**:
- 入场延次日开盘提交 (MARKET 单, 与回测 next_open 对齐)
- 出场保持即时响应
- 新增 pending_orders 表 + post-close / pre-market 双时点 cron / daemon

**2026-07-15 复评**: 再次确认暂不修。当前不跑 `live_trader.py` 自动化 daemon
(手动经纪商下单为主), 触发场景 1-4 均未激活。维持"已立项, 暂不修"状态,
下方阶段八按此前提重新排定优先级。

---

## 阶段八：组合结构与决策体验（第 12 周+）

> 2026-07-15 立项。前提：当前不跑 `live_trader.py` 自动化 daemon，手动经纪商
> 下单为主 → 实盘自动化相关项（P2-1 信号时点对齐、Sec-1 部分内容）降权，
> 优先做"不依赖实盘自动化也能立即受益"的组合结构和决策体验改进。
>
> 执行顺序：**W-1 → D-1 → Ops-1 → Sec-1 → CI-1**（长期方向见既有"阶段七"，
> 未新增独立编号）

```
阶段八  组合与体验   第 12 周+  ████████████░   (4/5 项, W-1 已决策 / D-1 / Ops-1 / CI-1 已完成 / Sec-1 降级搁置)
```

### W-1 Watchlist 组合结构优化

- [x] 分析完成 (2026-07-15) — **决策: 维持现状, 不修改 watchlist.toml**
- **文件:** `watchlist.toml` `analysis/correlation_analysis.py` `analysis/concentration.py` `analysis/what_if.py` `utils/sectors.py` — 新增 `scripts/portfolio_diversification_audit.py` (可复现审计工具, 参照 `strategy_fit_audit.py` 模式)
- **问题:** 当前 13 个观察标的 (AAPL/NVDA/TSLA/GOOG/AMZN/MU/INTC/ORCL/QQQ/SPY/SMH/MSFT/DRAM) 集中在科技/半导体, 相关性聚类此前已揭示"12 持仓实际只是 1.01 个独立赌注"(见阶段五 Bonus 项); 组合层面的分散化收益预期大于继续堆策略指标
- **方法论调整:** 原计划用 Brinson 拆行业暴露, 改用 `analysis/concentration.py` 的 `sector_hhi`/`sector_exposure` — Brinson 需要已实现收益 + 基准分解, 回答"为什么跑赢/跑输", 不适合静态持仓快照问题
- **发现:**
  1. Effective Bets = **1.10 / 13**(PCA 第一主成分解释 95.2% 方差); 层次聚类只抓到一组紧密簇 `MU/QQQ/SPY/SMH/DRAM` (MU↔DRAM corr=0.940), 其余 8 个标的各自独立成簇但仍共享同一市场因子
  2. `utils/sectors.py` 缺 SMH/DRAM 的 ETF 映射, 修复后 (已提交) Technology 暴露从 61.5% 修正为 **76.9%**, 行业 HHI 从 4320 修正为 **6213** (高度集中)
  3. What-If: 半导体簇减半仓 + 加 TLT/GLD/XLE/IWM → VaR(95%,1d) 3.07%→1.37%, 行业HHI 6213→4349 (仍 >2500, 减仓力度不够彻底分散)
- **决策 (2026-07-15):** 用户复核后选择维持现状, 不改 `watchlist.toml`。已完成的 `utils/sectors.py` 修复保留 (数据质量修正, 与是否调整持仓无关)。审计工具留存, 未来标的增减或定期复核时重跑 `scripts/portfolio_diversification_audit.py`

### D-1 Dashboard 决策摘要化

- [x] 完成 (2026-07-15)
- **文件:** 新增 `dashboard/decision_summary.py` (`build_decision_summary` 纯计算 + `render_decision_summary` 渲染) + `dashboard/main.py` (接在 `render_risk_light` 之前) + `tests/test_decision_summary.py` (16 用例)
- **问题:** Dashboard 功能齐全但偏"模块陈列", 缺一句话结论: 今日该不该交易/为什么/风险变化多少/信号被拒了几个
- **实现落地时与原方案的偏差** (调研发现原方案两处引用不成立, 详见实现过程):
  1. **被拒信号数据源写错**: `RiskController` 没有 `reason_if_blocked`; 真正的 `(bool,reason)` 门禁是 `utils/signal_gate.py` 的 `SignalGate.allow_buy/allow_sell`, 但此前只有 `live_trader.py` 的实盘循环调用过, Dashboard 从未跑过。`decision_history.signal_ignored` 也是零调用点的死记录类型。
     **v1 范围**: 只跑 `PAUSE_*` 和 `RANGING_BLOCK_*`/`TRENDING_BLOCK_*` 两类判定 (不需要 qty)。`EXPOSURE_CAP_EXCEEDED` (需要真实下单量) 和 `ORPHAN_BUY_BLOCKED` (live_trader 专属概念) 明确排除, 假设持仓语境下没有真实数据支撑, 不假装覆盖。
  2. **风险较昨日 Δ 没有持久化基础**: `risk_state` 表原本只存 `day_start_equity`/`peak_equity` 等交易状态。用户确认方案: **Dashboard 直接写** — 复用 `risk_state` 现成的 key/value 结构, 写入 `risk_snapshot:<date>` JSON, 不新增表、不需要 schema migration。理由: 这是只读分析缓存, 不是仓位/订单等交易执行状态, 不落入 AGENTS.md "dashboard 不能直接写新表"规则要防的那类风险。
- **验证**: 1251 个已有测试全绿 (无回归) + 16 个新测试 (`_verdict` 5 分支 / `_rejected_signals` regime 拦截 / `_risk_delta` 快照 round-trip) + 对真实 `watchlist.toml` 跑通完整链路 (6 信号中 2 个被 regime 门禁真实拦截, sector_hhi=5555 与 W-1 结论吻合) + Streamlit server 启动无异常。**未验证**: 浏览器实际渲染效果 — 环境无 `chromium-cli`/`playwright`, 未截图确认, 留给用户目测复核。
- **一句话结论算法**: `trading_paused` → 🛑 已暂停; `risk_level==RED` → 🔴 不建议开仓; 无信号 → ⚪; 信号全被拒 → 🟡; 否则 → 🟢 可正常交易 + N 个可执行信号

### Ops-1 数据源可观测性

- [x] 完成 (2026-07-16)
- **文件:** 新增 `data/source_health.py` (`record_fetch_result` / `is_in_cooldown` / `source_health_report` / `check_splits_changed`) + 改 `data/cache.py` (migration v4: `source_health` 表 + `StateStore.record_source_health`/`get_last_source_failure`/`get_source_health`) + 改 `data/provider.py` (`_fetch_from_sources` 接入 cooldown 判定与健康记录) + 改 `dashboard/ops.py` (两个新 expander + splits 提醒) + `tests/test_source_health.py` (18 用例)
- **问题:** 已有多源 fallback + 拆股修正 + drift warning, 但故障是否发生/ 哪个源在退化不可见, 也没有面板
- **实现落地时与原方案的偏差** (调研发现原方案假设不成立, 详见实现过程):
  1. **"失败冷却机制"此前名不副实**: `DataProvider._failed_sources` 是纯内存、只增不减的会话内熔断, 每次新建 `DataProvider()`（包括每次 Streamlit rerun）就清零, 起不到"冷却"的观测价值。新的 `is_in_cooldown` 是持久化、真正随时间过期的时间窗口判定（`source_health` 表按 `(source,date)` 记 success/failure 计数 + `last_failure_ts`）, 跨 rerun/跨进程一致。旧的 `_failed_sources` 集合作为同 tick 内的快速短路保留, 不冲突。
  2. **`data/quality.py` 复用的是函数本身, 不是现成集成** — 此前零调用点。接入时发现 `quality_report(df)` 依赖 `flag_missing`/`flag_price_jumps`/`flag_non_trading` 先跑过打标列, 直接传原始 df 会静默返回全 `None`（原方案没提到这个前置步骤）。
  3. **没有用 `ops_log`** — 它是交易语义表 (`gate_reject`/`risk_reject`/`trading_paused`), 数据源健康事件语义不搭, 改用新表。
  4. **`splits.json` 变更检测只做提醒, 不自动生效** — `_US_SPLITS` 是模块级常量, 检测到变更只能提示"需重启进程 + force_refresh", 不会自动重载或触发批量重拉（那是破坏性操作, 需人工确认）。
- **高风险改动的安全设计**: `_fetch_from_sources` 是 `daily.py`/`live_trader.py`/Dashboard 每次 render 都会走的热路径, 健康记录/冷却查询全部包 `try/except`, 任何异常都不能影响实际取数 — 专门写了 `test_cache_write_failure_does_not_break_fetch` 验证这条安全属性。
- **验证**: 1269 个测试全绿 (无回归, 较 D-1 时的 1251 净增 18) + 对真实 `watchlist.toml`/生产 DB 跑通 `render_ops`/`source_health_report`, 真实抓到 sina_us/tencent/yahoo_chart 的成功率与失败详情。浏览器渲染效果未截图确认 (环境无 chromium-cli/playwright)。

### Sec-1 实盘执行安全增强

- [ ] 未开始 — **优先级降级**: 无 daemon 运行时非阻塞, 可延后或跳过
- **文件:** `broker/base.py` (新增 `list_open_orders`) `live/kill_switch.py` `live/order_manager.py`
- **方案:** 取消未成交挂单 / 订单幂等 / 成交后对账 / 券商连接中断恢复 / 异常重试节流。`list_open_orders` 补齐后 Kill Switch 可先撤挂单再平仓
- **备注:** 与 P2-1 的 `pending_orders` 状态机有设计耦合 — 若未来两者都要做, 建议先合并设计挂单状态机 schema, 避免两套"挂单"概念打架

### CI-1 测试与 CI 加固

- [x] 完成 (2026-07-16)
- **文件:** 新增 `tests/test_layer_boundaries.py` (4 用例) + `tests/test_dashboard_smoke.py` (19 用例) + `tests/test_source_contract.py` (43 用例) + `tests/test_cache_migrations.py` (6 用例) + `scripts/check_golden_drift.py`。`.github/workflows/ci.yml` **未改动**——新测试文件在 `tests/` 下自动被现有 `pytest tests/` 步骤覆盖, 不需要新增 CI 步骤
- **实现落地时与原方案的偏差**:
  1. **自写层级检查, 不引入 import-linter 依赖** — AGENTS.md 的规则里有"broker 值类型 vs broker 实现类"这种细粒度区分, 标准 import-linter contract 类型表达不了, 用现成工具反而要多绕一层配置 DSL。用 `ast`（不是正则/grep）扫描 import 语句和 `.submit_order` 调用点——纯文本 grep 会把 `live/decision_logger.py`/`broker/mock.py` 里文档字符串中提到"submit_order"的地方也算违规（这两处 AGENTS.md 原始 grep 命令同样会误报, `ast.walk()` 只看真实语法节点, 不看字符串字面量), 天然规避这个坑。
  2. **dashboard smoke test 明确限定范围** — 只做了 (a) 全部 16 个子模块的 import 冒烟 + (b) `render_decision_summary`/`render_ops`/`render_risk_light` 三个高频函数的 bare-mode 调用冒烟（复用 D-1/Ops-1 已验证过的"无 ScriptRunContext 只警告不报错"结论）。`single_backtest`/`portfolio_backtest`/`factor_attribution`/`brinson_attribution`/`kill_switch` 等每个都需要独立的 broker/回测年数/选中标的上下文, 未覆盖, 留白不假装做完。
  3. **golden drift 脚本不接入 CI 门禁** — 定位是"我怀疑该更新黄金值了"时用的诊断工具（仿 `scripts/strategy_fit_audit.py`）, 不自动改写 `tests/test_golden.py`, 也不阻断 CI——数值变更必须人工决定 + 写清楚为什么, 工具只负责把"变了什么"摆清楚。
- **验证**: 1341 个测试全绿 (较 CI-1 开始前净增 72, 无回归) + 故意在 `analysis/` 临时注入 `import live` 违规, 确认 layer-boundary 测试真的会失败并精确报出文件:行号, revert 后确认恢复绿——不是摆设。`check_golden_drift.py` 实跑一遍, 8 个策略 × fixed_capital/risk_budget 两种模式全部"无漂移"（本次没有改动任何策略计算逻辑, 符合预期）。

### 阶段八后追加: 历史信号 Tab

- [x] 完成 (2026-07-25)
- **文件:** 新增 `dashboard/signal_history.py` (`render_signal_history`) + 改 `dashboard/main.py` (研究 tab 组新增"历史信号"子 tab, 位于"信号有效性"和"历史类比"之间) + `tests/test_dashboard_smoke.py` (import 冒烟 + 1 个 render 冒烟用例)
- **问题:** 用户想看"今日信号"的历史记录。调研发现 `render_signal_detail`("今日信号"区块) 每次调用 `daily.scan_day` 时, `SignalScanner` 已经把每次扫描结果写入 `signal_history` 表 (`data/cache.py`) —— 这个持久化本来就存在, 只是没有任何 UI 能看到, 唯一入口是 CLI `daily.py --history`。**不需要新增持久化逻辑, 纯读取现有表。**
- **关键设计: 按 (标的, 策略, K线日期, 信号) 去重** — weekly 策略的同一次信号会在 bar 收盘前被逐日重复扫描并重复写入 (例如 QQQ 的 weekly_macd 死叉在 7/24-7/25 两天扫描中各写了一条, bar_date 相同)。直接罗列 scan_date 会把 1 次真实信号显示成多次。去重后展示"首次探测 / 最近确认 / 确认天数", 用真实生产库验证: 101 条原始非零记录 → 71 个去重后的信号事件。
- **明确排除**: 不重新对全历史价格跑策略算信号时间线 (那是 `signal_effectiveness.py` 的 forward-return 分析在做的事, 不同问题) —— 这个 tab 只是`signal_history` 表的浏览器。
- **验证**: 21 个 dashboard smoke 测试全绿 + 全量回归 1343 个测试全绿 (无回归) + 用真实生产 DB 手动验证去重逻辑 (QQQ 死叉正确合并为 1 行, confirmations=2)。浏览器渲染效果未截图确认 (环境无 chromium-cli/playwright, 与 D-1/Ops-1/CI-1 一致的已知限制)。

### 阶段八后追加: 修复 DataProvider 永久性源黑名单 bug

- [x] 完成 (2026-07-28)
- **文件:** 改 `data/provider.py`(`__init__` 移除 `_failed_sources` 字段, `_fetch_from_sources` 移除该黑名单的读写逻辑, 只保留 Ops-1 已有的 `is_in_cooldown`/`_record_health` 持久化冷却机制)
- **问题:** 用户发现 Dashboard"市场风险灯"截止日期停留在 7/24, 而当天已经是 7/28。排查发现 `data/sources.py` 的 sina_us/tencent 直接测试都能拿到 7/27 的新数据, 说明不是数据源没数据, 是 `DataProvider` 层的 bug。
- **根因:** `_failed_sources`(`data/provider.py:99` 原有代码, 注释写着"stale sources — skip for all symbols")是一个**没有过期机制的全局黑名单**——只要某个源对**任意标的**的近期抓取返回过一次空结果, 就会在这个 `DataProvider` 实例的整个生命周期里被跳过。配合 `dashboard/main.py` 的 `@st.cache_resource def get_provider()` 单例模式, "session" 的实际含义变成了"直到 Streamlit 进程重启为止"——一次偶发的空结果(比如 yahoo_chart 被限流那次)就能让 SPY/QQQ/^VIX 这类标的的数据新鲜度永久性劣化。
- **修复:** 删除 `_failed_sources` 这套永久黑名单, 统一改用 Ops-1 已经建好的 `is_in_cooldown`(基于 `source_health` 表, 15 分钟自动过期)。两套机制原本并行存在, 新的这套有时效性但从未真正接管旧逻辑的位置, 这次是把旧的一次性剔除。
- **验证:** 用干净的临时 cache(无历史失败记录)复现修复前后行为 —— 修复前 `force_refresh=True` 仍卡在 7/24 (旧黑名单在同进程内测试脚本触发过一次空结果后永久生效); 修复后干净环境下 sina_us 正确拿到 7/27 数据。`tests/test_source_health.py` + `tests/test_provider.py`(75 用例)全绿, 全量回归 1343 个测试无回归。

---

> 此文档将随开发进度持续更新。每完成一项，勾选其 checkbox 并在进度表中记录。
