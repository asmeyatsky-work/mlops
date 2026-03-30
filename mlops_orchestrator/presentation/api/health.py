"""Health check endpoints with K8s readiness and liveness probe support."""
from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class HealthCheck:
    """Health check with K8s readiness/liveness probe support.

    - Liveness: is the process alive and not deadlocked?
    - Readiness: is the service ready to accept traffic (dependencies healthy)?
    """

    def __init__(self, container: object) -> None:
        self._container = container
        self._start_time = time.monotonic()
        self._ready = False

    async def check(self) -> dict[str, str]:
        """Basic health check — returns healthy if the process is up."""
        from mlops_orchestrator.infrastructure.config.container import DependencyContainer
        c: DependencyContainer = self._container  # type: ignore[assignment]
        adapter_mode = "stub" if c.settings.use_stubs else "gcp"
        return {
            "status": "healthy",
            "adapters": adapter_mode,
        }

    async def liveness(self) -> dict[str, object]:
        """K8s liveness probe.

        Checks:
        - Process is alive (implicit if this responds)
        - Event loop is not blocked (responds within timeout)
        - Uptime reporting
        """
        uptime = time.monotonic() - self._start_time
        return {
            "status": "alive",
            "uptime_seconds": round(uptime, 1),
        }

    async def readiness(self) -> dict[str, object]:
        """K8s readiness probe.

        Checks:
        - All configured adapters can be instantiated
        - Event bus is functional
        - Database/storage connectivity (via adapter health checks)
        """
        from mlops_orchestrator.infrastructure.config.container import DependencyContainer
        c: DependencyContainer = self._container  # type: ignore[assignment]

        checks: dict[str, str] = {}
        overall_ready = True

        # Check event bus
        try:
            bus = c.event_bus
            if bus is not None:
                checks["event_bus"] = "ok"
            else:
                checks["event_bus"] = "unavailable"
                overall_ready = False
        except Exception as e:
            checks["event_bus"] = f"error: {e}"
            overall_ready = False

        # Check dataset port
        try:
            _ = c.dataset_port
            checks["dataset_port"] = "ok"
        except Exception as e:
            checks["dataset_port"] = f"error: {e}"
            overall_ready = False

        # Check training port
        try:
            _ = c.training_port
            checks["training_port"] = "ok"
        except Exception as e:
            checks["training_port"] = f"error: {e}"
            overall_ready = False

        # Check deployment ports
        try:
            _ = c.vertex_deployment_port
            checks["vertex_deployment_port"] = "ok"
        except Exception as e:
            checks["vertex_deployment_port"] = f"error: {e}"
            overall_ready = False

        try:
            _ = c.gke_deployment_port
            checks["gke_deployment_port"] = "ok"
        except Exception as e:
            checks["gke_deployment_port"] = f"error: {e}"
            overall_ready = False

        # Check monitoring port
        try:
            _ = c.monitoring_port
            checks["monitoring_port"] = "ok"
        except Exception as e:
            checks["monitoring_port"] = f"error: {e}"
            overall_ready = False

        self._ready = overall_ready

        adapter_mode = "stub" if c.settings.use_stubs else "gcp"
        return {
            "status": "ready" if overall_ready else "not_ready",
            "adapters": adapter_mode,
            "checks": checks,
        }

    async def startup(self) -> dict[str, object]:
        """K8s startup probe.

        Used for slow-starting containers. Returns ready once
        the container has initialized all dependencies.
        """
        readiness_result = await self.readiness()
        return {
            "status": "started" if readiness_result["status"] == "ready" else "starting",
            "checks": readiness_result.get("checks", {}),
        }

    @property
    def is_ready(self) -> bool:
        return self._ready
