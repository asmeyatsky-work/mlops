"""Tests for infrastructure configuration."""
from __future__ import annotations

from mlops_orchestrator.infrastructure.config.settings import Settings
from mlops_orchestrator.infrastructure.config.container import DependencyContainer


class TestSettings:
    def test_defaults(self):
        s = Settings()
        assert s.gcp_project == "mlops-491617"
        assert s.gcp_location == "us-central1"
        assert s.use_stubs is True
        assert s.transport == "stdio"


class TestDependencyContainer:
    def test_stub_wiring(self):
        container = DependencyContainer(Settings(use_stubs=True))
        assert container.dataset_port is not None
        assert container.training_port is not None
        assert container.vertex_deployment_port is not None
        assert container.gke_deployment_port is not None
        assert container.monitoring_port is not None
        assert container.event_bus is not None
        assert container.audit_log is not None

    def test_command_factories(self):
        container = DependencyContainer(Settings(use_stubs=True))
        assert container.create_dataset_command() is not None
        assert container.train_model_command() is not None
        assert container.deploy_vertex_command() is not None
        assert container.deploy_gke_command() is not None
        assert container.configure_monitoring_command() is not None

    def test_query_factories(self):
        container = DependencyContainer(Settings(use_stubs=True))
        assert container.job_status_query() is not None
        assert container.cost_query() is not None
