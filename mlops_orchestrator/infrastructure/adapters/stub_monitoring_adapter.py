from __future__ import annotations


class StubMonitoringAdapter:
    """In-memory monitoring adapter. Implements MonitoringPort."""

    def __init__(self) -> None:
        self._configs: dict[str, dict[str, float]] = {}
        self._synthetic_alerts: list[dict[str, float]] = []

    async def configure_monitoring(
        self, endpoint_id: str, drift_threshold: float, skew_threshold: float
    ) -> bool:
        self._configs[endpoint_id] = {
            "drift_threshold": drift_threshold,
            "skew_threshold": skew_threshold,
        }
        return True

    async def get_monitoring_status(self, endpoint_id: str) -> dict[str, str]:
        if endpoint_id in self._configs:
            return {"status": "ACTIVE", "endpoint_id": endpoint_id}
        return {"status": "NOT_CONFIGURED", "endpoint_id": endpoint_id}

    async def get_drift_alerts(self, endpoint_id: str) -> list[dict[str, float]]:
        return self._synthetic_alerts

    def inject_alerts(self, alerts: list[dict[str, float]]) -> None:
        """Test helper: inject synthetic drift alerts."""
        self._synthetic_alerts = alerts
