"""Feishu (Lark) notification client — webhook + app mode.

Setup
-----
Webhook 模式 (简单):
    1. 飞书群 → 设置 → 群机器人 → 添加自定义机器人
    2. 复制 Webhook URL
    3. export FEISHU_WEBHOOK="https://open.feishu.cn/open-apis/bot/v2/hook/xxx"

App 模式 (功能更强，可发到任意群/人):
    1. 飞书开放平台 → 创建企业自建应用 → 获取 appId / appSecret
    2. 给应用开启"获取群组信息"和"发送消息"权限
    3. export FEISHU_APP_ID="cli_xxx"
       export FEISHU_APP_SECRET="xxx"
       export FEISHU_CHAT_ID="oc_xxx"  (群聊ID，从飞书开发者后台看)

Usage
-----
    from utils.notify import Notifier

    nf = Notifier()                              # 自动读取环境变量
    nf.text("AAPL trend_follower 买入信号 $195")  # 简单文本
    nf.signal_card(signals)                      # 结构化信号卡片
    nf.trade_card(order)                         # 成交通知
    nf.error("数据源连接失败")                     # 错误告警
"""

import atexit
import json
import logging as _logging
import os
import queue
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

import requests

from utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Feishu card color palette
# ---------------------------------------------------------------------------

COLOR = {
    "buy": "green",
    "sell": "red",
    "info": "blue",
    "warn": "yellow",
    "error": "red",
    "title": "turquoise",
}


# ---------------------------------------------------------------------------
# Notifier
# ---------------------------------------------------------------------------


class Notifier:
    """Feishu notification client — auto-detects webhook vs app mode.

    Priority: webhook URL > app credentials > dry-run

    Webhook mode: POST directly to webhook URL (no auth needed)
    App mode:     POST to message API with tenant_access_token

    Parameters
    ----------
    url : str | None
        Webhook URL.  If None, reads FEISHU_WEBHOOK from env.
    app_id : str | None
        Feishu app ID.  If None, reads FEISHU_APP_ID from env.
    app_secret : str | None
        Feishu app secret.  If None, reads FEISHU_APP_SECRET from env.
    chat_id : str | None
        Target chat ID for app mode.  Reads FEISHU_CHAT_ID from env.
    dry_run : bool
        If True, print messages instead of sending.

    Environment
    -----------
    FEISHU_WEBHOOK    : webhook URL (takes priority if set)
    FEISHU_APP_ID     : app ID for app mode
    FEISHU_APP_SECRET : app secret for app mode
    FEISHU_CHAT_ID    : target chat ID for app mode
    """

    def __init__(
        self,
        url: Optional[str] = None,
        app_id: Optional[str] = None,
        app_secret: Optional[str] = None,
        chat_id: Optional[str] = None,
        dry_run: bool = False,
        async_mode: bool = True,
    ):
        self.url = url or os.getenv("FEISHU_WEBHOOK", "")
        self.app_id = app_id or os.getenv("FEISHU_APP_ID", "")
        self.app_secret = app_secret or os.getenv("FEISHU_APP_SECRET", "")
        self.chat_id = chat_id or os.getenv("FEISHU_CHAT_ID", "")
        self.dry_run = dry_run
        self._async = async_mode
        self._token: Optional[str] = None
        self._token_expires: float = 0
        self._queue: queue.Queue = queue.Queue(maxsize=1000)
        self._worker: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        if self.url:
            self._mode = "webhook"
        elif self.app_id and self.app_secret:
            self._mode = "app"
        else:
            self._mode = "none"
            if not dry_run:
                logger.warning("飞书通知未配置 — 设置 FEISHU_WEBHOOK 或 FEISHU_APP_ID+FEISHU_APP_SECRET")

        if self._async and self.available:
            self._start_worker()

    @property
    def available(self) -> bool:
        return self._mode != "none" or self.dry_run

    # ------------------------------------------------------------------
    # Async worker
    # ------------------------------------------------------------------

    def _start_worker(self):
        if self._worker and self._worker.is_alive():
            return
        self._worker = threading.Thread(
            target=self._consume, name="notify-worker", daemon=True,
        )
        self._worker.start()
        atexit.register(self.stop)

    def _consume(self):
        while not self._stop_event.is_set():
            try:
                payload = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._send(payload)
            except Exception:
                logger.warning("通知消费异常", exc_info=True)

    def stop(self):
        self._stop_event.set()
        try:
            atexit.unregister(self.stop)
        except Exception:
            pass
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=2)

    def enqueue(self, payload: dict) -> bool:
        """Enqueue a payload for async delivery. Falls back to sync if async_mode=False."""
        if not self._async:
            return self._send(payload)
        if not self.available:
            return False
        try:
            self._queue.put_nowait(payload)
            return True
        except queue.Full:
            logger.warning("通知队列已满(%d)，丢弃消息", self._queue.maxsize)
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def text(self, content: str) -> bool:
        """Send a plain-text message.  Max 20KB."""
        payload = {
            "msg_type": "text",
            "content": {"text": content},
        }
        return self.enqueue(payload)

    _strat_cn = {
        "trend_follower": "趋势跟踪",
        "weekly_macd": "周线MACD",
        "macd_kdj": "KDJ+MACD",
        "weekly_macd_kdj": "周线KDJ+MACD",
        "daily_macd_kdj": "日线KDJ+MACD",
        "turtle_trading": "海龟交易",
        "donchian_breakout": "唐奇安突破",
        "atr_breakout": "ATR突破",
    }

    def _strat_label(self, key: str) -> str:
        return self._strat_cn.get(key, key)

    def signal_card(self, signals: List[dict], scan_date: str = "") -> bool:
        """Send a rich card summarising today's signals.

        Parameters
        ----------
        signals : list[dict]
            Each dict has keys: symbol, strategy, signal (1/-1), price
        scan_date : str
            YYYY-MM-DD
        """
        buys = [s for s in signals if s["signal"] == 1]
        sells = [s for s in signals if s["signal"] == -1]

        if not buys and not sells:
            return self.text(f"[{scan_date}] 今日无买入/卖出信号")

        elements = []
        if buys:
            elements.append(self._mk_field(
                f"买入信号 ({len(buys)})",
                "\n".join(f"{s['symbol']}  {self._strat_label(s['strategy'])}  ${s['price']:.2f}"
                          for s in buys),
            ))
        if sells:
            elements.append(self._mk_field(
                f"卖出信号 ({len(sells)})",
                "\n".join(f"{s['symbol']}  {self._strat_label(s['strategy'])}  ${s['price']:.2f}"
                          for s in sells),
            ))

        card = self._mk_card(
            title=f"每日回溯 — {scan_date}",
            color=COLOR["buy"] if buys else COLOR["sell"],
            elements=elements,
            footer=f"共 {len(signals)} 个策略-标的组合  |  {datetime.now().strftime('%H:%M')}",
        )
        return self._send({"msg_type": "interactive", "card": card})

    def trade_card(self, order) -> bool:
        """Send a trade fill notification."""
        side = "买入" if order.side.value == "BUY" else "卖出"
        color = COLOR["buy"] if order.side.value == "BUY" else COLOR["sell"]
        status = "已成交" if order.status.value == "FILLED" else order.status.value

        elements = [
            self._mk_field("标的", order.symbol),
            self._mk_field("方向", f"{side} {order.filled_qty}股"),
            self._mk_field("价格", f"${order.avg_fill_price:.2f}"),
            self._mk_field("状态", status),
            self._mk_field("订单ID", order.order_id),
        ]
        card = self._mk_card(
            title=f"交易通知 — {side} {order.symbol}",
            color=color,
            elements=elements,
        )
        return self._send({"msg_type": "interactive", "card": card})

    def error(self, message: str, context: str = "") -> bool:
        """Send an error / alert notification."""
        elements = [self._mk_field("错误", message)]
        if context:
            elements.append(self._mk_field("上下文", context))
        card = self._mk_card(
            title="系统告警",
            color=COLOR["error"],
            elements=elements,
        )
        return self._send({"msg_type": "interactive", "card": card})

    def daily_summary(self, buy_count: int, sell_count: int, total: int,
                      account_equity: float = 0, positions: int = 0) -> bool:
        """End-of-day summary card."""
        elements = [
            self._mk_field("买入信号", str(buy_count)),
            self._mk_field("卖出信号", str(sell_count)),
            self._mk_field("扫描组合数", str(total)),
        ]
        if account_equity > 0:
            elements.append(self._mk_field("账户权益", f"${account_equity:,.0f}"))
            elements.append(self._mk_field("持仓数", str(positions)))

        card = self._mk_card(
            title=f"每日回溯完成 — {datetime.now().strftime('%Y-%m-%d')}",
            color=COLOR["info"],
            elements=elements,
        )
        return self._send({"msg_type": "interactive", "card": card})

    def daily_card(
        self,
        pnl_by_strategy: dict = None,
        pnl_by_symbol: dict = None,
        positions_data: list = None,
        risk_events: dict = None,
        signals_preview: list = None,
    ) -> bool:
        """Enhanced EOD card with PnL attribution, risk events, positions.

        Parameters
        ----------
        pnl_by_strategy : dict[str, float]
        pnl_by_symbol : dict[str, float]
        positions_data : list[dict]
            Each: symbol, shares, avg_price, last_price, unrealized_pnl
        risk_events : dict
            circuit_breaks, slippage_exceeded, consecutive_losses
        signals_preview : list[dict]
            Up to 5 next-day preview signals.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        elements = []

        # PnL attribution
        if pnl_by_strategy:
            lines = [f"{k}: ${v:+,.0f}" for k, v in pnl_by_strategy.items()]
            elements.append(self._mk_field("PnL 按策略", "\n".join(lines[:8])))
        if pnl_by_symbol:
            lines = [f"{k}: ${v:+,.0f}" for k, v in pnl_by_symbol.items()]
            elements.append(self._mk_field("PnL 按标的", "\n".join(lines[:8])))

        # Positions
        if positions_data:
            lines = []
            for p in positions_data[:10]:
                upnl = p.get("unrealized_pnl", 0)
                lines.append(f"{p['symbol']} {p.get('shares',0)}股 "
                           f"@{p.get('avg_price',0):.2f} 浮盈${upnl:+,.0f}")
            elements.append(self._mk_field("当前持仓", "\n".join(lines)))

        # Risk events
        if risk_events:
            lines = []
            if risk_events.get("circuit_breaks", 0) > 0:
                lines.append(f"熔断: {risk_events['circuit_breaks']}次")
            if risk_events.get("slippage_exceeded", 0) > 0:
                lines.append(f"滑点超标: {risk_events['slippage_exceeded']}次")
            if risk_events.get("consecutive_losses", 0) > 0:
                lines.append(f"连亏: {risk_events['consecutive_losses']}笔")
            if lines:
                elements.append(self._mk_field("风控事件", "\n".join(lines)))

        # Signal preview
        if signals_preview:
            lines = []
            for s in signals_preview[:5]:
                sig_val = s.get("signal", 0)
                sig = "买入" if sig_val == 1 else ("卖出" if sig_val == -1 else "—")
                lines.append(f"{s.get('strategy','?'):<18s} {sig} {s['symbol']} "
                           f"@{s.get('price',0):.2f}")
            elements.append(self._mk_field("明日信号预览", "\n".join(lines)))

        if not elements:
            elements.append(self._mk_field("状态", "无异常，策略运行正常"))

        card = self._mk_card(
            title=f"日报 — {today}",
            color=COLOR["info"],
            elements=elements,
        )
        return self._send({"msg_type": "interactive", "card": card})

    def risk_alert_card(self, state) -> bool:
        """Send a RED risk-light alert card.

        ``state`` is an :class:`analysis.risk_monitor.RiskState`.  Renders
        title, reasons bullets, and current indicators.
        """
        ind = state.indicators or {}
        elements = []
        if state.reasons:
            elements.append(self._mk_field(
                "触发原因",
                "\n".join(f"• {r}" for r in state.reasons),
            ))
        if ind:
            lines = []
            if "spy_close" in ind:
                lines.append(f"SPY {ind['spy_close']} (MA200 {ind.get('spy_ma200','-')}, "
                             f"{ind.get('spy_vs_ma200_pct',0):+.2f}%)")
            if "spy_adx" in ind:
                lines.append(f"ADX {ind['spy_adx']}")
            if "spy_5d_return_pct" in ind:
                lines.append(f"SPY 5日 {ind['spy_5d_return_pct']:+.2f}%")
            if "vix" in ind:
                lines.append(f"VIX {ind['vix']}")
            if lines:
                elements.append(self._mk_field("当前指标", "\n".join(lines)))
        if "as_of" in ind:
            elements.append(self._mk_field("截止", str(ind["as_of"])))

        card = self._mk_card(
            title=f"{state.emoji} 风险灯转 RED — 减仓预警",
            color=COLOR["error"],
            elements=elements,
            footer=f"traderbridge 风险监控  |  {datetime.now().strftime('%H:%M')}",
        )
        return self.enqueue({"msg_type": "interactive", "card": card})

    def position_alert_card(self, position: dict, distance_pct: float) -> bool:
        """Send a position-approaching-stop alert card."""
        symbol = position.get("symbol", "?")
        current = position.get("current_price", 0)
        stop = position.get("stop_price", 0)
        shares = position.get("shares")
        strategy = position.get("strategy")

        elements = [
            self._mk_field("当前价", f"${current:.2f}"),
            self._mk_field("止损价", f"${stop:.2f}"),
            self._mk_field("距止损", f"{distance_pct:.2f}%"),
        ]
        if shares:
            elements.append(self._mk_field("持仓", f"{shares} 股"))
        if strategy:
            elements.append(self._mk_field("策略", self._strat_label(strategy)))

        card = self._mk_card(
            title=f"⚠ {symbol} 临近止损",
            color=COLOR["warn"],
            elements=elements,
            footer=f"traderbridge 风险监控  |  {datetime.now().strftime('%H:%M')}",
        )
        return self.enqueue({"msg_type": "interactive", "card": card})

    def kill_switch_card(
        self,
        reason: str,
        n_positions: int,
        orders: list,
        errors: list,
        dry_run: bool = False,
    ) -> bool:
        """Send a Kill Switch activation card.

        ``orders`` and ``errors`` are lists of dicts produced by
        :meth:`live.kill_switch.KillSwitch.trigger`.  Renders a RED card
        with reason + position count + per-symbol order status.
        """
        elements = [
            self._mk_field("触发原因", reason or "—"),
            self._mk_field("持仓数 / 已下单",
                           f"{n_positions} / {len(orders)} "
                           f"{'(DRY RUN)' if dry_run else ''}"),
        ]
        if orders:
            lines = []
            for o in orders[:15]:
                lines.append(
                    f"{o.get('symbol', '?'):<6} {o.get('side', '?'):<5} "
                    f"{o.get('qty', 0)} → {o.get('status', '?')}"
                )
            elements.append(self._mk_field("订单明细", "\n".join(lines)))
        if errors:
            lines = [
                f"{e.get('symbol', '?')}: {e.get('error', '?')[:80]}"
                for e in errors[:5]
            ]
            elements.append(self._mk_field("错误", "\n".join(lines)))
        title_prefix = "🚨 KILL SWITCH" if not dry_run else "🚨 KILL SWITCH (DRY RUN)"
        card = self._mk_card(
            title=f"{title_prefix} — 紧急平仓触发",
            color=COLOR["error"],
            elements=elements,
            footer=f"traderbridge 风险监控  |  {datetime.now().strftime('%H:%M:%S')}",
        )
        return self.enqueue({"msg_type": "interactive", "card": card})

    def vix_alert_card(self, vix_value: float, threshold: float) -> bool:
        """Send a VIX spike alert card."""
        elements = [
            self._mk_field("VIX 当前", f"{vix_value:.2f}"),
            self._mk_field("触发阈值", f"≥ {threshold:.0f}"),
            self._mk_field("含义", "极端恐慌区间, 考虑降仓 / 暂停加仓"),
        ]
        card = self._mk_card(
            title=f"VIX 突破 — {vix_value:.1f}",
            color=COLOR["error"],
            elements=elements,
            footer=f"traderbridge 风险监控  |  {datetime.now().strftime('%H:%M')}",
        )
        return self.enqueue({"msg_type": "interactive", "card": card})

    # ------------------------------------------------------------------
    # Internal — send
    # ------------------------------------------------------------------

    def _send(self, payload: dict) -> bool:
        if self.dry_run:
            logger.info("[DRY-RUN] 飞书通知: %s", json.dumps(payload, ensure_ascii=False)[:200])
            return True

        if self._mode == "webhook":
            return self._send_webhook(payload)
        elif self._mode == "app":
            return self._send_app(payload)
        return False

    def _send_webhook(self, payload: dict) -> bool:
        try:
            r = requests.post(self.url, json=payload, timeout=10)
            if r.status_code == 200:
                resp = r.json()
                if resp.get("code") == 0:
                    return True
                logger.warning("飞书 webhook 返回错误: %s", resp.get("msg", "unknown"))
            else:
                logger.warning("飞书 webhook HTTP %d: %s", r.status_code, r.text[:100])
        except Exception as e:
            logger.warning("飞书 webhook 发送失败: %s", e)
        return False

    def _send_app(self, payload: dict) -> bool:
        """Send via Feishu message API using app credentials."""
        token = self._get_app_token()
        if not token:
            return False
        if not self.chat_id:
            logger.warning("FEISHU_CHAT_ID 未设置，无法发送 app 模式消息")
            return False

        # Determine message type
        msg_type = payload.get("msg_type", "text")
        content = payload.get("content", payload.get("card"))

        body = {
            "receive_id": self.chat_id,
            "msg_type": msg_type,
            "content": json.dumps(content) if isinstance(content, dict) else content,
        }

        try:
            r = requests.post(
                "https://open.feishu.cn/open-apis/im/v1/messages",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                params={"receive_id_type": "chat_id"},
                json=body,
                timeout=10,
            )
            if r.status_code == 200:
                resp = r.json()
                if resp.get("code") == 0:
                    return True
                logger.warning("飞书 app 返回错误: %s", resp.get("msg", "unknown"))
            else:
                logger.warning("飞书 app HTTP %d: %s", r.status_code, r.text[:100])
        except Exception as e:
            logger.warning("飞书 app 发送失败: %s", e)
        return False

    def _get_app_token(self) -> Optional[str]:
        """Get or refresh tenant_access_token."""
        now = time.time()
        if self._token and now < self._token_expires - 60:
            return self._token

        try:
            r = requests.post(
                "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
                json={"app_id": self.app_id, "app_secret": self.app_secret},
                timeout=10,
            )
            if r.status_code == 200:
                data = r.json()
                if data.get("code") == 0:
                    self._token = data["tenant_access_token"]
                    self._token_expires = now + data.get("expire", 7200)
                    return self._token
                logger.warning("获取飞书 token 失败: %s", data.get("msg"))
            else:
                logger.warning("飞书 token HTTP %d", r.status_code)
        except Exception as e:
            logger.warning("飞书 token 请求异常: %s", e)
        return None

    @staticmethod
    def _mk_card(title: str, color: str, elements: List[dict],
                 footer: str = "") -> dict:
        card = {
            "header": {
                "title": {"tag": "plain_text", "content": title},
                "template": color,
            },
            "elements": elements,
        }
        if footer:
            card["elements"].append({
                "tag": "note",
                "elements": [{"tag": "plain_text", "content": footer}],
            })
        return card

    @staticmethod
    def _mk_field(label: str, value: str) -> dict:
        return {
            "tag": "div",
            "text": {"tag": "lark_md", "content": f"**{label}**\n{value}"},
        }


# ---------------------------------------------------------------------------
# Logging → Notification bridge
# ---------------------------------------------------------------------------


class NotifyLogHandler(_logging.Handler):
    """Bridge Python logging ERROR+ records to Notifier via async queue."""

    def __init__(self, notifier: Optional[Notifier] = None, level: int = _logging.ERROR):
        super().__init__(level=level)
        self._notifier = notifier or Notifier()
        self.setFormatter(_logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))

    def emit(self, record: _logging.LogRecord):
        try:
            msg = self.format(record)
            self._notifier.error(msg, context=record.name)
        except Exception:
            self.handleError(record)


def install_notify_log_handler(
    notifier: Optional[Notifier] = None,
    level: int = _logging.ERROR,
    fmt: str = "[%(levelname)s] %(name)s: %(message)s",
) -> None:
    """Install NotifyLogHandler on the root logger.  Idempotent."""
    notifier = notifier or Notifier()
    root = _logging.getLogger()
    for h in root.handlers:
        if isinstance(h, NotifyLogHandler):
            return
    handler = NotifyLogHandler(notifier, level=level)
    handler.setFormatter(_logging.Formatter(fmt))
    root.addHandler(handler)
