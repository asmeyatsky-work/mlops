"""Tests for K8s health check probes."""
from __future__ import annotations

from mlops_orchestrator.infrastructure.config.settings import Settings
from mlops_orchestrator.infrastructure.config.container import DependencyContainer
from mlops_orchestrator.presentation.api.health import HealthCheck


class TestLivenessProbe:
    async def test_liveness_returns_alive(self):
        container = DependencyContainer(Settings(use_stubs=True))
        hc = HealthCheck(container)
        result = await hc.liveness()
        assert result["status"] == "alive"
        assert "uptime_seconds" in result
        assert result["uptime_seconds"] >= 0

    async def test_liveness_uptime_increases(self):
        container = DependencyContainer(Settings(use_stubs=True))
        hc = HealthCheck(container)
        r1 = await hc.liveness()
        r2 = await hc.liveness()
        assert r2["uptime_seconds"] >= r1["uptime_seconds"]


class TestReadinessProbe:
    async def test_readiness_stub_mode(self):
        container = DependencyContainer(Settings(use_stubs=True))
        hc = HealthCheck(container)
        result = await hc.readiness()
        assert result["status"] == "ready"
        assert result["adapters"] == "stub"
        checks = result["checks"]
        assert checks["event_bus"] == "ok"
        assert checks["dataset_port"] == "ok"
        assert checks["training_port"] == "ok"
        assert checks["vertex_deployment_port"] == "ok"
        assert checks["gke_deployment_port"] == "ok"
        assert checks["monitoring_port"] == "ok"

    async def test_readiness_sets_is_ready_flag(self):
        container = DependencyContainer(Settings(use_stubs=True))
        hc = HealthCheck(container)
        assert hc.is_ready is False
        await hc.readiness()
        assert hc.is_ready is True


class TestStartupProbe:
    async def test_startup_returns_started(self):
        container = DependencyContainer(Settings(use_stubs=True))
        hc = HealthCheck(container)
        result = await hc.startup()
        assert result["status"] == "started"
        assert "checks" in result


class TestBasicHealthCheck:
    async def test_check_stub_mode(self):
        container = DependencyContainer(Settings(use_stubs=True))
        hc = HealthCheck(container)
        result = await hc.check()
        assert result["status"] == "healthy"
        assert result["adapters"] == "stub"

    async def test_no_project_id_in_response(self):
        container = DependencyContainer(Settings(use_stubs=True))
        hc = HealthCheck(container)
        result = await hc.check()
        assert "project" not in result
