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


class TestDeployToVertexCommand:
    async def test_execute(self, event_bus, audit_log, session):
        cmd = DeployToVertexCommand(StubVertexDeploymentAdapter(), event_bus, audit_log)
        req = DeployToVertexRequest(model_id="m-1", endpoint_name="ep")
        resp, new_session = await cmd.execute(req, session)
        assert resp.status == "DEPLOYED"
        assert resp.target == "vertex"
        assert new_session.latest_endpoint == resp.resource_name
        assert audit_log.all_entries[0]["action"] == "deploy_to_vertex"


class TestDeployToGkeCommand:
    async def test_execute(self, event_bus, audit_log, session):
        cmd = DeployToGkeCommand(StubGkeDeploymentAdapter(), event_bus, audit_log)
        req = DeployToGkeRequest(model_id="m-1", cluster_name="cluster-1")
        resp, new_session = await cmd.execute(req, session)
        assert resp.status == "DEPLOYED"
        assert resp.target == "gke"
        assert "gke://" in new_session.latest_endpoint
        assert audit_log.all_entries[0]["action"] == "deploy_to_gke"


class TestConfigureMonitoringCommand:
    async def test_execute_success(self, event_bus, audit_log, session):
        cmd = ConfigureMonitoringCommand(StubMonitoringAdapter(), event_bus, audit_log)
        req = MonitoringRequest(endpoint_id="ep-1")
        resp, new_session = await cmd.execute(req, session)
        assert resp.monitoring_enabled
        assert resp.status == "ACTIVE"
        assert new_session.metadata["monitoring_ep-1"] == "enabled"
        assert audit_log.all_entries[0]["action"] == "configure_monitoring"
