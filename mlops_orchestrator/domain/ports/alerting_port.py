"""Port for alerting notifications (Slack, PagerDuty, email)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True)
class Alert:
    """An alert to send through one or more channels."""
    title: str
    message: str
    severity: Literal["info", "warning", "critical"]
    source: str = "mlops-orchestrator"
    metadata: dict[str, str] | None = None


class AlertingPort(Protocol):
    """Port for sending alerts through various channels."""

    async def send_alert(self, alert: Alert) -> bool:
        """Send an alert. Returns True if delivered successfully."""
        ...
