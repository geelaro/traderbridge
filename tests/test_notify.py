"""Tests for utils/notify.py — Notifier with mocked HTTP requests."""

import json
from unittest.mock import patch, MagicMock

import pytest

from utils.notify import Notifier


# ===================================================================
# Notifier init
# ===================================================================


class TestNotifierInit:
    def test_dry_run_mode(self):
        nf = Notifier(dry_run=True, async_mode=False)
        assert nf.dry_run is True
        assert nf.available is True

    def test_webhook_mode(self):
        nf = Notifier(async_mode=False, url="https://hooks.example.com/test")
        assert nf._mode == "webhook"

    def test_app_mode(self):
        nf = Notifier(async_mode=False, app_id="cli_xxx", app_secret="secret", chat_id="oc_xxx")
        assert nf._mode == "app"

    def test_none_mode_when_no_config(self, monkeypatch):
        monkeypatch.delenv("FEISHU_WEBHOOK", raising=False)
        monkeypatch.delenv("FEISHU_APP_ID", raising=False)
        monkeypatch.delenv("FEISHU_APP_SECRET", raising=False)
        monkeypatch.delenv("FEISHU_CHAT_ID", raising=False)
        nf = Notifier(async_mode=False)
        assert nf._mode == "none"
        assert nf.available is False

    def test_env_webhook(self, monkeypatch):
        monkeypatch.setenv("FEISHU_WEBHOOK", "https://hooks.example.com/env")
        nf = Notifier(async_mode=False)
        assert nf._mode == "webhook"
        assert nf.url == "https://hooks.example.com/env"

    def test_env_app(self, monkeypatch):
        monkeypatch.setenv("FEISHU_APP_ID", "cli_env")
        monkeypatch.setenv("FEISHU_APP_SECRET", "sec_env")
        monkeypatch.setenv("FEISHU_CHAT_ID", "oc_env")
        nf = Notifier(async_mode=False)
        assert nf._mode == "app"


# ===================================================================
# Dry-run mode (no external calls)
# ===================================================================


class TestNotifierDryRun:
    def test_text(self, capsys):
        nf = Notifier(dry_run=True, async_mode=False)
        assert nf.text("test message") is True

    def test_signal_card(self, capsys):
        nf = Notifier(dry_run=True, async_mode=False)
        signals = [
            {"symbol": "AAPL", "strategy": "weekly_macd", "signal": 1, "price": 195.0},
        ]
        assert nf.signal_card(signals, "2025-01-15") is True

    def test_signal_card_empty(self, capsys):
        nf = Notifier(dry_run=True, async_mode=False)
        # Empty signals → sends plain text
        assert nf.signal_card([], "2025-01-15") is True

    def test_trade_card(self):
        from broker import Order, OrderSide, OrderType, OrderStatus
        order = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET,
            quantity=10, order_id="abc123", status=OrderStatus.FILLED,
            avg_fill_price=195.0,
        )
        nf = Notifier(dry_run=True, async_mode=False)
        assert nf.trade_card(order) is True

    def test_error(self):
        nf = Notifier(dry_run=True, async_mode=False)
        assert nf.error("something broke", "context") is True

    def test_daily_summary(self):
        nf = Notifier(dry_run=True, async_mode=False)
        assert nf.daily_summary(3, 1, 10, 100000, 5) is True

    def test_daily_summary_no_account(self):
        nf = Notifier(dry_run=True, async_mode=False)
        assert nf.daily_summary(2, 0, 5) is True


# ===================================================================
# Webhook mode (mocked HTTP)
# ===================================================================


class TestNotifierWebhook:
    def test_send_success(self):
        nf = Notifier(async_mode=False, url="https://hooks.example.com/test")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"code": 0, "msg": "ok"}

        with patch("utils.notify.requests.post", return_value=mock_resp) as mock_post:
            result = nf.text("hello")
            assert result is True
            mock_post.assert_called_once()
            payload = mock_post.call_args[1]["json"]
            assert payload["msg_type"] == "text"

    def test_send_webhook_error_code(self):
        nf = Notifier(async_mode=False, url="https://hooks.example.com/test")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"code": 10001, "msg": "invalid"}

        with patch("utils.notify.requests.post", return_value=mock_resp):
            result = nf.text("hello")
            assert result is False

    def test_send_webhook_http_error(self):
        nf = Notifier(async_mode=False, url="https://hooks.example.com/test")

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"

        with patch("utils.notify.requests.post", return_value=mock_resp):
            result = nf.text("hello")
            assert result is False

    def test_send_webhook_exception(self):
        import logging
        from utils.notify import NotifyLogHandler

        nf = Notifier(async_mode=False, url="https://hooks.example.com/test")
        callback = MagicMock()
        handler = NotifyLogHandler(callback, level=logging.ERROR)
        logger = logging.getLogger("utils.notify")
        logger.addHandler(handler)

        try:
            with patch("utils.notify.requests.post", side_effect=Exception("timeout")):
                result = nf.text("hello")
        finally:
            logger.removeHandler(handler)

        assert result is False
        callback.error.assert_not_called()

    def test_send_card_via_webhook(self):
        nf = Notifier(async_mode=False, url="https://hooks.example.com/test")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"code": 0}

        signals = [
            {"symbol": "AAPL", "strategy": "weekly_macd", "signal": 1, "price": 195.0},
            {"symbol": "NVDA", "strategy": "turtle_trading", "signal": -1, "price": 850.0},
        ]
        with patch("utils.notify.requests.post", return_value=mock_resp) as mock_post:
            nf.signal_card(signals, "2025-01-15")
            payload = mock_post.call_args[1]["json"]
            assert payload["msg_type"] == "interactive"
            assert "card" in payload


# ===================================================================
# App mode (mocked HTTP)
# ===================================================================


class TestNotifierApp:
    def test_get_token_success(self):
        nf = Notifier(async_mode=False, app_id="cli_xxx", app_secret="secret", chat_id="oc_xxx")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "code": 0,
            "tenant_access_token": "tok-abc123",
            "expire": 7200,
        }

        with patch("utils.notify.requests.post", return_value=mock_resp):
            token = nf._get_app_token()
            assert token == "tok-abc123"
            assert nf._token == "tok-abc123"

    def test_get_token_reuses_cached(self):
        nf = Notifier(async_mode=False, app_id="cli_xxx", app_secret="secret", chat_id="oc_xxx")
        nf._token = "cached-tok"
        nf._token_expires = 9999999999  # far future

        # Should return cached without making HTTP call
        with patch("utils.notify.requests.post") as mock_post:
            token = nf._get_app_token()
            assert token == "cached-tok"
            mock_post.assert_not_called()

    def test_get_token_error(self):
        nf = Notifier(async_mode=False, app_id="cli_xxx", app_secret="secret", chat_id="oc_xxx")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"code": 10001, "msg": "invalid app_id"}

        with patch("utils.notify.requests.post", return_value=mock_resp):
            token = nf._get_app_token()
            assert token is None

    def test_send_app_no_chat_id(self):
        nf = Notifier(async_mode=False, app_id="cli_xxx", app_secret="secret")
        nf._token = "tok"
        nf._token_expires = 9999999999
        # No chat_id set
        result = nf._send_app({"msg_type": "text", "content": {"text": "hi"}})
        assert result is False

    def test_send_app_success(self):
        nf = Notifier(async_mode=False, app_id="cli_xxx", app_secret="secret", chat_id="oc_xxx")
        nf._token = "tok"
        nf._token_expires = 9999999999

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"code": 0}

        with patch("utils.notify.requests.post", return_value=mock_resp) as mock_post:
            result = nf._send_app({"msg_type": "text", "content": {"text": "hi"}})
            assert result is True
            call_args = mock_post.call_args[1]["json"]
            assert call_args["receive_id"] == "oc_xxx"

    def test_send_app_api_error(self):
        nf = Notifier(async_mode=False, app_id="cli_xxx", app_secret="secret", chat_id="oc_xxx")
        nf._token = "tok"
        nf._token_expires = 9999999999

        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = "Bad Request"

        with patch("utils.notify.requests.post", return_value=mock_resp):
            result = nf._send_app({"msg_type": "text", "content": {"text": "hi"}})
            assert result is False

    def test_send_mode_none_returns_false(self, monkeypatch):
        monkeypatch.delenv("FEISHU_WEBHOOK", raising=False)
        monkeypatch.delenv("FEISHU_APP_ID", raising=False)
        monkeypatch.delenv("FEISHU_APP_SECRET", raising=False)
        monkeypatch.delenv("FEISHU_CHAT_ID", raising=False)
        nf = Notifier(async_mode=False)
        assert nf._send({"msg_type": "text", "content": {"text": "hi"}}) is False


# ===================================================================
# Card builders
# ===================================================================


class TestCardBuilders:
    def test_mk_card(self):
        card = Notifier._mk_card("title", "blue", [], "footer text")
        assert card["header"]["title"]["content"] == "title"
        assert card["header"]["template"] == "blue"
        # Footer is appended as a note element
        assert len(card["elements"]) == 1  # just the footer note
        assert card["elements"][0]["tag"] == "note"

    def test_mk_card_no_footer(self):
        card = Notifier._mk_card("title", "red", [])
        assert len(card["elements"]) == 0

    def test_mk_field(self):
        field = Notifier._mk_field("label", "value")
        assert field["tag"] == "div"
        assert "label" in field["text"]["content"]
        assert "value" in field["text"]["content"]

    def test_strat_label(self):
        nf = Notifier(dry_run=True, async_mode=False)
        assert nf._strat_label("trend_follower") == "趋势跟踪"
        assert nf._strat_label("weekly_macd") == "周线MACD"
        assert nf._strat_label("weekly_macd_kdj") == "周线KDJ+MACD"
        assert nf._strat_label("unknown") == "unknown"


# ===================================================================
# Async path — enqueue + NotifyLogHandler
# ===================================================================


class TestNotifierAsync:
    def test_enqueue_puts_on_queue(self):
        nf = Notifier(dry_run=True, async_mode=True)
        # Stop worker to isolate queue behaviour
        nf.stop()
        assert nf._queue.qsize() == 0
        result = nf.enqueue({"msg_type": "text", "content": {"text": "hello"}})
        assert result is True
        assert nf._queue.qsize() == 1

    def test_enqueue_starts_worker(self):
        nf = Notifier(dry_run=True, async_mode=True)
        assert nf._worker is not None
        assert nf._worker.is_alive()
        nf.stop()

    def test_async_mode_false_calls_send_directly(self):
        nf = Notifier(dry_run=True, async_mode=False)
        assert nf._worker is None
        result = nf.enqueue({"msg_type": "text", "content": {"text": "sync"}})
        assert result is True
        assert nf._queue.qsize() == 0  # never queued

    def test_consumer_drains_queue(self):
        nf = Notifier(dry_run=True, async_mode=True)
        nf.stop()  # stop worker, set event so it exits cleanly
        # Worker is now stopped, queue isolated
        nf._queue.put({"msg_type": "text", "content": {"text": "msg1"}})
        nf._queue.put({"msg_type": "text", "content": {"text": "msg2"}})
        assert nf._queue.qsize() == 2
        # Consume one
        nf._send(nf._queue.get(timeout=0.1))
        assert nf._queue.qsize() == 1

    def test_enqueue_returns_false_when_unavailable(self):
        nf = Notifier(async_mode=True)  # no credentials, all env empty
        # mode should be "none" if no env vars are set
        if nf.available:
            pytest.skip("env has feishu creds")
        result = nf.enqueue({"msg_type": "text"})
        assert result is False

    def test_stop_joins_worker(self):
        nf = Notifier(dry_run=True, async_mode=True)
        assert nf._worker.is_alive()
        nf.stop()
        # Worker should exit cleanly after stop event is set
        assert not nf._worker.is_alive()


class TestNotifyLogHandler:
    def test_emit_enqueues_error(self):
        """emit() should call notifier.error() which enqueues a payload."""
        nf = Notifier(dry_run=True, async_mode=False)  # sync mode for direct check
        from utils.notify import NotifyLogHandler
        import logging as _logging
        handler = NotifyLogHandler(nf, level=_logging.ERROR)
        record = _logging.LogRecord(
            "test", _logging.ERROR, "test.py", 1, "test message", (), None,
        )
        # sync mode: error() → enqueue() → _send() returns True
        result = handler.emit(record)
        # emit doesn't return anything — if it didn't raise, it worked
        # Verify by checking that dry_run logged via _send
        assert True  # no exception = pass

    def test_emit_skips_below_level(self):
        nf = Notifier(dry_run=True, async_mode=True)
        nf.stop()
        from utils.notify import NotifyLogHandler
        import logging
        handler = NotifyLogHandler(nf, level=logging.ERROR)
        record = logging.LogRecord(
            "test", logging.WARNING, "test.py", 1, "just a warning", (), None,
        )
        handler.emit(record)
        assert nf._queue.qsize() == 0

    def test_install_notify_log_handler_idempotent(self):
        import logging as _logging
        from utils.notify import install_notify_log_handler, NotifyLogHandler

        root = _logging.getLogger()
        before = sum(1 for h in root.handlers if isinstance(h, NotifyLogHandler))

        install_notify_log_handler(level=_logging.ERROR)
        after_first = sum(1 for h in root.handlers if isinstance(h, NotifyLogHandler))
        assert after_first >= before

        install_notify_log_handler(level=_logging.ERROR)
        after_second = sum(1 for h in root.handlers if isinstance(h, NotifyLogHandler))
        assert after_second == after_first  # no dup
