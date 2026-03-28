"""Tests for application commands."""
from __future__ import annotations

import pytest

from mlops_orchestrator.application.commands.create_dataset_command import CreateDatasetCommand
from mlops_orchestrator.application.commands.train_model_command import TrainModelCommand
from mlops_orchestrator.application.commands.deploy_vertex_command import DeployToVertexCommand
from mlops_orchestrator.application.commands.deploy_gke_command import DeployToGkeCommand
from mlops_orchestrator.application.commands.configure_monitoring_command import ConfigureMonitoringCommand
from mlops_orchestrator.application.dtos.dataset_dto import CreateDatasetRequest
from mlops_orchestrator.application.dtos.training_dto import TrainModelRequest
from mlops_orchestrator.application.dtos.deployment_dto import (
    DeployToVertexRequest, DeployToGkeRequest, MonitoringRequest,
)
from mlops_orchestrator.application.session.session_state import SessionState
from mlops_orchestrator.domain.events.dataset_events import DatasetCreatedEvent
from mlops_orchestrator.domain.events.training_events import TrainingJobStartedEvent
from mlops_orchestrator.domain.events.deployment_events import (
    ModelDeployedToVertexEvent, ModelDeployedToGkeEvent,
)
from mlops_orchestrator.domain.events.monitoring_events import MonitoringConfiguredEvent
from mlops_orchestrator.domain.value_objects.bq_source import BigQuerySource
from mlops_orchestrator.domain.value_objects.machine_spec import MachineSpec
from mlops_orchestrator.infrastructure.adapters.stub_dataset_adapter import StubDatasetAdapter
from mlops_orchestrator.infrastructure.adapters.stub_training_adapter import StubTrainingAdapter
from mlops_orchestrator.infrastructure.adapters.stub_deployment_adapter import (
    StubVertexDeploymentAdapter, StubGkeDeploymentAdapter,
)
from mlops_orchestrator.infrastructure.adapters.stub_monitoring_adapter import StubMonitoringAdapter
from mlops_orchestrator.infrastructure.adapters.stub_infrastructure_adapters import (
    InMemoryEventBus, StubAuditLogAdapter,
)


@pytest.fixture
def event_bus():
    return InMemoryEventBus()


@pytest.fixture
def audit_log():
    return StubAuditLogAdapter()


@pytest.fixture
def session():
    return SessionState()


# ── Failing port helpers ─────────────────────────────────────────────


class FailingDatasetPort:
    async def create_dataset(self, bq_source: BigQuerySource, display_name: str) -> str:
        raise RuntimeError("dataset port exploded")


class FailingTrainingPort:
    async def start_training(
        self, model_name: str, dataset_id: str, gcs_uri: str, train_image: str
    ) -> str:
        raise RuntimeError("training port exploded")


class FailingVertexDeploymentPort:
    async def create_endpoint_and_deploy(
        self, model_id: str, endpoint_name: str, machine_spec: MachineSpec
    ) -> str:
        raise RuntimeError("vertex deploy port exploded")


class FailingGkeDeploymentPort:
    async def deploy(
        self, model_id: str, cluster_name: str, replica_count: int
    ) -> dict[str, str]:
        raise RuntimeError("gke deploy port exploded")


class FailingMonitoringPort:
    async def configure_monitoring(
        self, endpoint_id: str, drift_threshold: float, skew_threshold: float
    ) -> bool:
        raise RuntimeError("monitoring port exploded")


# ── CreateDatasetCommand ─────────────────────────────────────────────


class TestCreateDatasetCommand:
    async def test_execute(self, event_bus, audit_log, session):
        cmd = CreateDatasetCommand(StubDatasetAdapter(), event_bus, audit_log)
        req = CreateDatasetRequest(bq_dataset="ds", bq_table="tbl", name="test")
        resp, new_session = await cmd.execute(req, session)
        assert resp.status == "REGISTERED"
        assert resp.display_name == "test"
        assert new_session.latest_dataset == resp.resource_name
        assert len(event_bus.published_events) > 0
        assert len(audit_log.all_entries) == 1
        assert audit_log.all_entries[0]["action"] == "create_dataset"

    async def test_validation_error(self, event_bus, audit_log, session):
        cmd = CreateDatasetCommand(StubDatasetAdapter(), event_bus, audit_log)
        req = CreateDatasetRequest(bq_dataset="bad.ds", bq_table="tbl", name="test")
        with pytest.raises(ValueError, match="Invalid BigQuery source"):
            await cmd.execute(req, session)

    async def test_port_failure_logs_and_reraises(self, event_bus, audit_log, session):
        """When the dataset port raises, the error is audit-logged and re-raised."""
        cmd = CreateDatasetCommand(FailingDatasetPort(), event_bus, audit_log)
        req = CreateDatasetRequest(bq_dataset="ds", bq_table="tbl", name="test")
        with pytest.raises(RuntimeError, match="dataset port exploded"):
            await cmd.execute(req, session)
        assert len(audit_log.all_entries) == 1
        assert "error" in audit_log.all_entries[0]

    async def test_session_state_unchanged_after_port_failure(self, event_bus, audit_log, session):
        """Session state must not change when the port raises."""
        original = session
        cmd = CreateDatasetCommand(FailingDatasetPort(), event_bus, audit_log)
        req = CreateDatasetRequest(bq_dataset="ds", bq_table="tbl", name="test")
        with pytest.raises(RuntimeError):
            await cmd.execute(req, session)
        assert session == original
        assert session.dataset_ids == ()

    async def test_event_type_is_dataset_created(self, event_bus, audit_log, session):
        """Published event is a DatasetCreatedEvent."""
        cmd = CreateDatasetCommand(StubDatasetAdapter(), event_bus, audit_log)
        req = CreateDatasetRequest(bq_dataset="ds", bq_table="tbl", name="test")
        await cmd.execute(req, session)
        assert any(isinstance(e, DatasetCreatedEvent) for e in event_bus.published_events)


# ── TrainModelCommand ────────────────────────────────────────────────


class TestTrainModelCommand:
    async def test_execute(self, event_bus, audit_log, session):
        cmd = TrainModelCommand(StubTrainingAdapter(), event_bus, audit_log)
        req = TrainModelRequest(model_name="my-model", dataset_id="ds-1")
        resp, new_session = await cmd.execute(req, session)
        assert resp.status == "RUNNING"
        assert resp.model_name == "my-model"
        assert new_session.latest_job == resp.job_resource_name
        assert len(event_bus.published_events) > 0
        assert audit_log.all_entries[0]["action"] == "train_model"

    async def test_port_failure_logs_and_reraises(self, event_bus, audit_log, session):
        """When the training port raises, the error is audit-logged and re-raised."""
        cmd = TrainModelCommand(FailingTrainingPort(), event_bus, audit_log)
        req = TrainModelRequest(model_name="m", dataset_id="ds-1")
        with pytest.raises(RuntimeError, match="training port exploded"):
            await cmd.execute(req, session)
        assert len(audit_log.all_entries) == 1
        assert "error" in audit_log.all_entries[0]

    async def test_event_type_is_training_started(self, event_bus, audit_log, session):
        """Published event is a TrainingJobStartedEvent."""
        cmd = TrainModelCommand(StubTrainingAdapter(), event_bus, audit_log)
        req = TrainModelRequest(model_name="m", dataset_id="ds-1")
        await cmd.execute(req, session)
        assert any(isinstance(e, TrainingJobStartedEvent) for e in event_bus.published_events)

    async def test_execute_with_gcs_uri(self, event_bus, audit_log, session):
        """TrainModelCommand works when given gcs_uri instead of dataset_id."""
        cmd = TrainModelCommand(StubTrainingAdapter(), event_bus, audit_log)
        req = TrainModelRequest(model_name="m", gcs_uri="gs://bucket/path")
        resp, new_session = await cmd.execute(req, session)
        assert resp.status == "RUNNING"
        assert new_session.latest_job == resp.job_resource_name


# ── DeployToVertexCommand ────────────────────────────────────────────


class TestDeployToVertexCommand:
    async def test_execute(self, event_bus, audit_log, session):
        cmd = DeployToVertexCommand(StubVertexDeploymentAdapter(), event_bus, audit_log)
        req = DeployToVertexRequest(model_id="m-1", endpoint_name="ep")
        resp, new_session = await cmd.execute(req, session)
        assert resp.status == "DEPLOYED"
        assert resp.target == "vertex"
        assert new_session.latest_endpoint == resp.resource_name
        assert audit_log.all_entries[0]["action"] == "deploy_to_vertex"

    async def test_port_failure_logs_and_reraises(self, event_bus, audit_log, session):
        """When the vertex deployment port raises, the error is audit-logged and re-raised."""
        cmd = DeployToVertexCommand(FailingVertexDeploymentPort(), event_bus, audit_log)
        req = DeployToVertexRequest(model_id="m-1", endpoint_name="ep")
        with pytest.raises(RuntimeError, match="vertex deploy port exploded"):
            await cmd.execute(req, session)
        assert len(audit_log.all_entries) == 1
        assert "error" in audit_log.all_entries[0]

    async def test_event_type_is_model_deployed_to_vertex(self, event_bus, audit_log, session):
        """Published event is a ModelDeployedToVertexEvent."""
        cmd = DeployToVertexCommand(StubVertexDeploymentAdapter(), event_bus, audit_log)
        req = DeployToVertexRequest(model_id="m-1", endpoint_name="ep")
        await cmd.execute(req, session)
        assert any(
            isinstance(e, ModelDeployedToVertexEvent) for e in event_bus.published_events
        )


# ── DeployToGkeCommand ───────────────────────────────────────────────


class TestDeployToGkeCommand:
    async def test_execute(self, event_bus, audit_log, session):
        cmd = DeployToGkeCommand(StubGkeDeploymentAdapter(), event_bus, audit_log)
        req = DeployToGkeRequest(model_id="m-1", cluster_name="cluster-1")
        resp, new_session = await cmd.execute(req, session)
        assert resp.status == "DEPLOYED"
        assert resp.target == "gke"
        assert "gke://" in new_session.latest_endpoint
        assert audit_log.all_entries[0]["action"] == "deploy_to_gke"

    async def test_port_failure_logs_and_reraises(self, event_bus, audit_log, session):
        """When the GKE deployment port raises, the error is audit-logged and re-raised."""
        cmd = DeployToGkeCommand(FailingGkeDeploymentPort(), event_bus, audit_log)
        req = DeployToGkeRequest(model_id="m-1", cluster_name="cluster-1")
        with pytest.raises(RuntimeError, match="gke deploy port exploded"):
            await cmd.execute(req, session)
        assert len(audit_log.all_entries) == 1
        assert "error" in audit_log.all_entries[0]

    async def test_event_type_is_model_deployed_to_gke(self, event_bus, audit_log, session):
        """Published event is a ModelDeployedToGkeEvent."""
        cmd = DeployToGkeCommand(StubGkeDeploymentAdapter(), event_bus, audit_log)
        req = DeployToGkeRequest(model_id="m-1", cluster_name="cluster-1")
        await cmd.execute(req, session)
        assert any(
            isinstance(e, ModelDeployedToGkeEvent) for e in event_bus.published_events
        )


# ── ConfigureMonitoringCommand ───────────────────────────────────────


class TestConfigureMonitoringCommand:
    async def test_execute_success(self, event_bus, audit_log, session):
        cmd = ConfigureMonitoringCommand(StubMonitoringAdapter(), event_bus, audit_log)
        req = MonitoringRequest(endpoint_id="ep-1")
        resp, new_session = await cmd.execute(req, session)
        assert resp.monitoring_enabled
        assert resp.status == "ACTIVE"
        assert new_session.metadata["monitoring_ep-1"] == "enabled"
        assert audit_log.all_entries[0]["action"] == "configure_monitoring"

    async def test_port_failure_logs_and_reraises(self, event_bus, audit_log, session):
        """When the monitoring port raises, the error is audit-logged and re-raised."""
        cmd = ConfigureMonitoringCommand(FailingMonitoringPort(), event_bus, audit_log)
        req = MonitoringRequest(endpoint_id="ep-1")
        with pytest.raises(RuntimeError, match="monitoring port exploded"):
            await cmd.execute(req, session)
        assert len(audit_log.all_entries) == 1
        assert "error" in audit_log.all_entries[0]
        assert audit_log.all_entries[0].get("status") == "error"

    async def test_auto_fail_stub_returns_failed(self, event_bus, audit_log, session):
        """StubMonitoringAdapter with auto_fail=True causes FAILED response."""
        cmd = ConfigureMonitoringCommand(
            StubMonitoringAdapter(auto_fail=True), event_bus, audit_log
        )
        req = MonitoringRequest(endpoint_id="ep-1")
        resp, new_session = await cmd.execute(req, session)
        assert not resp.monitoring_enabled
        assert resp.status == "FAILED"
        assert new_session.metadata["monitoring_ep-1"] == "failed"

    async def test_session_state_unchanged_after_port_failure(self, event_bus, audit_log, session):
        """Session state must not change when the port raises."""
        original = session
        cmd = ConfigureMonitoringCommand(FailingMonitoringPort(), event_bus, audit_log)
        req = MonitoringRequest(endpoint_id="ep-1")
        with pytest.raises(RuntimeError):
            await cmd.execute(req, session)
        assert session == original

    async def test_event_type_is_monitoring_configured(self, event_bus, audit_log, session):
        """Published event is a MonitoringConfiguredEvent."""
        cmd = ConfigureMonitoringCommand(StubMonitoringAdapter(), event_bus, audit_log)
        req = MonitoringRequest(endpoint_id="ep-1")
        await cmd.execute(req, session)
        assert any(
            isinstance(e, MonitoringConfiguredEvent) for e in event_bus.published_events
        )

    async def test_audit_log_includes_status_field(self, event_bus, audit_log, session):
        """Audit log entry includes a 'status' field."""
        cmd = ConfigureMonitoringCommand(StubMonitoringAdapter(), event_bus, audit_log)
        req = MonitoringRequest(endpoint_id="ep-1")
        await cmd.execute(req, session)
        entry = audit_log.all_entries[0]
        assert "status" in entry
        assert entry["status"] == "success"
