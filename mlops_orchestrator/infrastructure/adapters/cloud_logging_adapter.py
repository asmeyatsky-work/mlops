from __future__ import annotations
import asyncio
import re
from datetime import datetime, UTC

from mlops_orchestrator.infrastructure.adapters.retry import with_retry


class CloudLoggingAuditAdapter:
    """Real Cloud Logging audit adapter. Implements AuditLogPort."""

    def __init__(self, project: str) -> None:
        from google.cloud import logging as cloud_logging
        self._client = cloud_logging.Client(project=project)
        self._logger = self._client.logger("mlops-orchestrator-audit")

    @with_retry(max_attempts=3)
    async def log_action(
        self, action: str, resource_id: str, details: dict[str, str]
    ) -> None:
        await asyncio.to_thread(
            self._logger.log_struct,
            {
                "action": action,
                "resource_id": resource_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "details": details,
            },
        )

    @with_retry(max_attempts=3)
    async def get_audit_trail(self, resource_id: str) -> list[dict[str, str]]:
        # Sanitize resource_id to prevent log filter injection
        sanitized = re.sub(r'[^a-zA-Z0-9/_\-.]', '', resource_id)
        filter_str = (
            f'resource.type="global" AND '
            f'jsonPayload.resource_id="{sanitized}"'
        )
        entries = await asyncio.to_thread(
            lambda: list(self._client.list_entries(filter_=filter_str, max_results=100))
        )
        return [
            {
                "action": e.payload.get("action", ""),
                "resource_id": e.payload.get("resource_id", ""),
                "timestamp": e.payload.get("timestamp", ""),
                **e.payload.get("details", {}),
            }
            for e in entries
        ]
