from __future__ import annotations
import json
from datetime import datetime, UTC
from google.cloud import logging as cloud_logging


class CloudLoggingAuditAdapter:
    """Real Cloud Logging audit adapter. Implements AuditLogPort."""

    def __init__(self, project: str) -> None:
        self._client = cloud_logging.Client(project=project)
        self._logger = self._client.logger("mlops-orchestrator-audit")

    async def log_action(
        self, action: str, resource_id: str, details: dict[str, str]
    ) -> None:
        self._logger.log_struct({
            "action": action,
            "resource_id": resource_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "details": details,
        })

    async def get_audit_trail(self, resource_id: str) -> list[dict[str, str]]:
        filter_str = (
            f'resource.type="global" AND '
            f'jsonPayload.resource_id="{resource_id}"'
        )
        entries = list(self._client.list_entries(filter_=filter_str, max_results=100))
        return [
            {
                "action": e.payload.get("action", ""),
                "resource_id": e.payload.get("resource_id", ""),
                "timestamp": e.payload.get("timestamp", ""),
                **e.payload.get("details", {}),
            }
            for e in entries
        ]
