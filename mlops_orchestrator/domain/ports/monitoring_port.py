from __future__ import annotations
from typing import Protocol


class MonitoringPort(Protocol):
    """Port for Vertex Model Monitoring configuration."""

    async def configure_monitoring(
        self,
        endpoint_id: str,
        drift_threshold: float,
        skew_threshold: float,
    ) -> bool:
        """Configure model monitoring on an endpoint. Returns success."""
        ...

    async def get_monitoring_status(self, endpoint_id: str) -> dict[str, str]:
        """Get current monitoring configuration and status."""
        ...

    async def get_drift_alerts(self, endpoint_id: str) -> list[dict[str, float]]:
        """Retrieve recent drift detection alerts."""
        ...
