from __future__ import annotations


class HealthCheck:
    """Simple health check for the MLOps Orchestrator."""

    def __init__(self, container: object) -> None:
        self._container = container

    async def check(self) -> dict[str, str]:
        from mlops_orchestrator.infrastructure.config.container import DependencyContainer
        c: DependencyContainer = self._container  # type: ignore[assignment]
        adapter_mode = "stub" if c.settings.use_stubs else "gcp"
        return {
            "status": "healthy",
            "adapters": adapter_mode,
            "project": c.settings.gcp_project,
        }
