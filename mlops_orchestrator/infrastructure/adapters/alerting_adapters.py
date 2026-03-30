"""Alerting adapters for Slack, PagerDuty, and email notifications."""
from __future__ import annotations

import asyncio
import json
import logging
import smtplib
import ssl
from email.message import EmailMessage
from typing import Any
from urllib.request import Request, urlopen

from mlops_orchestrator.domain.ports.alerting_port import Alert
from mlops_orchestrator.infrastructure.adapters.retry import with_retry

logger = logging.getLogger(__name__)


class SlackAlertAdapter:
    """Sends alerts to a Slack channel via incoming webhook. Implements AlertingPort."""

    def __init__(self, webhook_url: str) -> None:
        self._webhook_url = webhook_url

    @with_retry(max_attempts=3)
    async def send_alert(self, alert: Alert) -> bool:
        severity_emoji = {
            "info": ":information_source:",
            "warning": ":warning:",
            "critical": ":rotating_light:",
        }
        emoji = severity_emoji.get(alert.severity, ":bell:")
        color = {"info": "#36a64f", "warning": "#ff9900", "critical": "#ff0000"}.get(
            alert.severity, "#cccccc"
        )

        payload: dict[str, Any] = {
            "attachments": [
                {
                    "color": color,
                    "blocks": [
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"{emoji} *{alert.title}*\n{alert.message}",
                            },
                        },
                        {
                            "type": "context",
                            "elements": [
                                {
                                    "type": "mrkdwn",
                                    "text": f"Source: {alert.source} | Severity: {alert.severity}",
                                }
                            ],
                        },
                    ],
                }
            ]
        }
        if alert.metadata:
            fields_text = "\n".join(f"*{k}:* {v}" for k, v in alert.metadata.items())
            payload["attachments"][0]["blocks"].insert(
                1,
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": fields_text},
                },
            )

        data = json.dumps(payload).encode("utf-8")
        req = Request(
            self._webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            await asyncio.to_thread(urlopen, req, timeout=10)
            return True
        except Exception:
            logger.exception("Failed to send Slack alert")
            return False


class PagerDutyAlertAdapter:
    """Sends alerts to PagerDuty via the Events API v2. Implements AlertingPort."""

    _EVENTS_URL = "https://events.pagerduty.com/v2/enqueue"

    def __init__(self, routing_key: str) -> None:
        self._routing_key = routing_key

    @with_retry(max_attempts=3)
    async def send_alert(self, alert: Alert) -> bool:
        pd_severity = {
            "info": "info",
            "warning": "warning",
            "critical": "critical",
        }.get(alert.severity, "info")

        payload = {
            "routing_key": self._routing_key,
            "event_action": "trigger",
            "payload": {
                "summary": f"[{alert.source}] {alert.title}: {alert.message}",
                "severity": pd_severity,
                "source": alert.source,
                "custom_details": alert.metadata or {},
            },
        }

        data = json.dumps(payload).encode("utf-8")
        req = Request(
            self._EVENTS_URL,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            await asyncio.to_thread(urlopen, req, timeout=10)
            return True
        except Exception:
            logger.exception("Failed to send PagerDuty alert")
            return False


class EmailAlertAdapter:
    """Sends alerts via SMTP email. Implements AlertingPort."""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        sender: str,
        recipients: list[str],
        username: str = "",
        password: str = "",
        use_tls: bool = True,
    ) -> None:
        self._smtp_host = smtp_host
        self._smtp_port = smtp_port
        self._sender = sender
        self._recipients = recipients
        self._username = username
        self._password = password
        self._use_tls = use_tls

    @with_retry(max_attempts=3, base_delay=2.0)
    async def send_alert(self, alert: Alert) -> bool:
        msg = EmailMessage()
        msg["Subject"] = f"[{alert.severity.upper()}] {alert.title}"
        msg["From"] = self._sender
        msg["To"] = ", ".join(self._recipients)

        body = f"{alert.message}\n\nSource: {alert.source}\nSeverity: {alert.severity}"
        if alert.metadata:
            body += "\n\nDetails:\n"
            body += "\n".join(f"  {k}: {v}" for k, v in alert.metadata.items())
        msg.set_content(body)

        try:
            await asyncio.to_thread(self._send_smtp, msg)
            return True
        except Exception:
            logger.exception("Failed to send email alert")
            return False

    def _send_smtp(self, msg: EmailMessage) -> None:
        if self._use_tls:
            context = ssl.create_default_context()
            with smtplib.SMTP(self._smtp_host, self._smtp_port) as server:
                server.starttls(context=context)
                if self._username:
                    server.login(self._username, self._password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(self._smtp_host, self._smtp_port) as server:
                if self._username:
                    server.login(self._username, self._password)
                server.send_message(msg)


class CompositeAlertAdapter:
    """Fans out alerts to multiple adapters. Implements AlertingPort."""

    def __init__(self, adapters: list[Any]) -> None:
        self._adapters = adapters

    async def send_alert(self, alert: Alert) -> bool:
        if not self._adapters:
            return True
        results = await asyncio.gather(
            *(adapter.send_alert(alert) for adapter in self._adapters),
            return_exceptions=True,
        )
        return any(r is True for r in results)


class StubAlertAdapter:
    """In-memory alert adapter for testing. Implements AlertingPort."""

    def __init__(self) -> None:
        self._sent: list[Alert] = []

    async def send_alert(self, alert: Alert) -> bool:
        self._sent.append(alert)
        return True

    @property
    def sent_alerts(self) -> list[Alert]:
        return list(self._sent)

    def clear(self) -> None:
        self._sent.clear()
