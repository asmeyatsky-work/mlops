"""Tests for infrastructure configuration."""
from __future__ import annotations

from mlops_orchestrator.infrastructure.config.settings import Settings
from mlops_orchestrator.infrastructure.config.container import DependencyContainer


class TestSettings:
    def test_defaults(self):
        s = Settings()
        assert s.gcp_project == ""
        assert s.gcp_location == "us-central1"
        assert s.use_stubs is False
        assert s.transport == "stdio"

    def test_settings_from_env_vars(self, monkeypatch):
        """Settings can be loaded from environment variables with MLOPS_ prefix."""
        monkeypatch.setenv("MLOPS_GCP_PROJECT", "my-test-project")
        monkeypatch.setenv("MLOPS_GCP_LOCATION", "europe-west1")
        monkeypatch.setenv("MLOPS_USE_STUBS", "true")
        monkeypatch.setenv("MLOPS_TRANSPORT", "sse")
        s = Settings()
        assert s.gcp_project == "my-test-project"
        assert s.gcp_location == "europe-west1"
        assert s.use_stubs is True
        assert s.transport == "sse"


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

    def test_container_creates_independent_command_instances(self):
        """Each call to a command factory returns a distinct instance."""
        container = DependencyContainer(Settings(use_stubs=True))
        cmd1 = container.create_dataset_command()
        cmd2 = container.create_dataset_command()
        assert cmd1 is not cmd2

        train1 = container.train_model_command()
        train2 = container.train_model_command()
        assert train1 is not train2

        deploy1 = container.deploy_vertex_command()
        deploy2 = container.deploy_vertex_command()
        assert deploy1 is not deploy2
