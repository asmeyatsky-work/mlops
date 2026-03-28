"""Tests for health check endpoint."""
from __future__ import annotations

from mlops_orchestrator.infrastructure.config.settings import Settings
from mlops_orchestrator.infrastructure.config.container import DependencyContainer
from mlops_orchestrator.presentation.api.health import HealthCheck


class TestHealthCheck:
    async def test_stub_mode(self):
        container = DependencyContainer(Settings(use_stubs=True))
        hc = HealthCheck(container)
        result = await hc.check()
        assert result["status"] == "healthy"
        assert result["adapters"] == "stub"

    async def test_no_project_id_in_response(self):
        """Health check should not leak the GCP project ID."""
        container = DependencyContainer(Settings(use_stubs=True))
        hc = HealthCheck(container)
        result = await hc.check()
        assert "project" not in result
