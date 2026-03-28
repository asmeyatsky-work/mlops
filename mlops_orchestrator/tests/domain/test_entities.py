"""Tests for domain entities."""
from __future__ import annotations

import pytest

from mlops_orchestrator.domain.value_objects.bq_source import BigQuerySource
from mlops_orchestrator.domain.value_objects.machine_spec import MachineSpec
from mlops_orchestrator.domain.value_objects.drift_result import (
    DriftResult, DriftType, DriftSeverity,
)
from mlops_orchestrator.domain.entities.managed_dataset import ManagedDataset
from mlops_orchestrator.domain.entities.training_job import TrainingJob
from mlops_orchestrator.domain.entities.deployment import VertexDeployment, GkeDeployment
from mlops_orchestrator.domain.entities.monitoring_config import MonitoringConfig
from mlops_orchestrator.domain.entities.agent import Agent, AgentTask, AgentRole
from mlops_orchestrator.domain.events.dataset_events import (
    DatasetCreatedEvent, DatasetValidationFailedEvent,
)
from mlops_orchestrator.domain.events.training_events import (
    TrainingJobStartedEvent, TrainingJobCompletedEvent, TrainingJobFailedEvent,
)
from mlops_orchestrator.domain.events.deployment_events import (
    ModelDeployedToVertexEvent, ModelDeployedToGkeEvent, ModelUndeployedEvent,
)
from mlops_orchestrator.domain.events.monitoring_events import (
    MonitoringConfiguredEvent, DriftDetectedEvent, RemediationTriggeredEvent,
)


# ── ManagedDataset ────────────────────────────────────────────────────

class TestManagedDataset:
    def _make(self) -> ManagedDataset:
        return ManagedDataset.create(
            bq_source=BigQuerySource(dataset="ds", table="tbl"),
            display_name="test-dataset",
        )

    def test_create(self):
        ds = self._make()
        assert ds.status == "PENDING"
        assert ds.display_name == "test-dataset"
        assert ds.bq_source.dataset == "ds"
        assert ds.domain_events == ()

    def test_register_transitions_to_registered(self):
        ds = self._make().register("projects/p/locations/l/datasets/d")
        assert ds.status == "REGISTERED"
        assert ds.resource_name == "projects/p/locations/l/datasets/d"

    def test_register_emits_event(self):
        ds = self._make().register("rn")
        assert len(ds.domain_events) == 1
        event = ds.domain_events[0]
        assert isinstance(event, DatasetCreatedEvent)
        assert event.bq_dataset == "ds"
        assert event.bq_table == "tbl"

    def test_fail_validation(self):
        ds = self._make().fail_validation("bad data")
        assert ds.status == "VALIDATION_FAILED"
        assert len(ds.domain_events) == 1
        assert isinstance(ds.domain_events[0], DatasetValidationFailedEvent)
        assert ds.domain_events[0].reason == "bad data"

    def test_events_accumulate(self):
        ds = self._make().register("rn").fail_validation("reason")
        assert len(ds.domain_events) == 2


# ── TrainingJob ───────────────────────────────────────────────────────

class TestTrainingJob:
    def _make(self, **kw) -> TrainingJob:
        defaults = {"model_name": "my-model", "dataset_id": "ds-1"}
        defaults.update(kw)
        return TrainingJob.create(**defaults)

    def test_create_with_dataset(self):
        job = self._make()
        assert job.status == "PENDING"
        assert job.model_name == "my-model"

    def test_create_with_gcs(self):
        job = TrainingJob.create(model_name="m", gcs_uri="gs://b/data")
        assert job.gcs_uri == "gs://b/data"

    def test_create_requires_data_source(self):
        with pytest.raises(ValueError, match="Either dataset_id or gcs_uri"):
            TrainingJob.create(model_name="m")

    def test_start(self):
        job = self._make().start("job-rn")
        assert job.status == "RUNNING"
        assert job.job_resource_name == "job-rn"
        assert isinstance(job.domain_events[0], TrainingJobStartedEvent)

    def test_complete(self):
        job = self._make().start("j").complete("model-rn")
        assert job.status == "SUCCEEDED"
        assert job.model_resource_name == "model-rn"
        assert isinstance(job.domain_events[-1], TrainingJobCompletedEvent)

    def test_fail_from_running(self):
        job = self._make().start("j").fail("OOM")
        assert job.status == "FAILED"
        assert isinstance(job.domain_events[-1], TrainingJobFailedEvent)

    def test_fail_from_pending(self):
        job = self._make().fail("infra error")
        assert job.status == "FAILED"

    def test_invalid_transition_complete_from_pending(self):
        with pytest.raises(ValueError, match="Invalid state transition"):
            self._make().complete("rn")

    def test_invalid_transition_start_from_succeeded(self):
        job = self._make().start("j").complete("rn")
        with pytest.raises(ValueError, match="Invalid state transition"):
            job.start("j2")

    def test_is_terminal(self):
        assert not self._make().is_terminal
        assert self._make().start("j").complete("rn").is_terminal
        assert self._make().start("j").fail("err").is_terminal

    def test_is_active(self):
        assert not self._make().is_active
        assert self._make().start("j").is_active
        assert not self._make().start("j").complete("rn").is_active


# ── VertexDeployment ──────────────────────────────────────────────────

class TestVertexDeployment:
    def test_create_defaults(self):
        dep = VertexDeployment.create(model_id="m", endpoint_name="ep")
        assert dep.status == "PENDING"
        assert dep.machine_spec.machine_type == "n1-standard-4"
        assert not dep.monitoring_enabled

    def test_create_with_spec(self):
        spec = MachineSpec(machine_type="n1-highmem-8", accelerator_type="T4", accelerator_count=1)
        dep = VertexDeployment.create(model_id="m", endpoint_name="ep", machine_spec=spec)
        assert dep.machine_spec.has_gpu

    def test_deploy(self):
        dep = VertexDeployment.create("m", "ep").deploy("ep-rn")
        assert dep.status == "DEPLOYED"
        assert dep.endpoint_resource_name == "ep-rn"
        assert isinstance(dep.domain_events[0], ModelDeployedToVertexEvent)

    def test_enable_monitoring(self):
        dep = VertexDeployment.create("m", "ep").enable_monitoring()
        assert dep.monitoring_enabled

    def test_undeploy(self):
        dep = VertexDeployment.create("m", "ep").deploy("rn").undeploy("drift")
        assert dep.status == "UNDEPLOYED"
        assert isinstance(dep.domain_events[-1], ModelUndeployedEvent)


# ── GkeDeployment ─────────────────────────────────────────────────────

class TestGkeDeployment:
    def test_create_defaults(self):
        dep = GkeDeployment.create(model_id="m", cluster_name="c")
        assert dep.replica_count == 2
        assert dep.status == "PENDING"

    def test_mark_deployed(self):
        dep = GkeDeployment.create("m", "c").mark_deployed()
        assert dep.status == "DEPLOYED"
        event = dep.domain_events[0]
        assert isinstance(event, ModelDeployedToGkeEvent)
        assert event.deployment_status == "DEPLOYED"

    def test_mark_failed(self):
        dep = GkeDeployment.create("m", "c").mark_failed("timeout")
        assert dep.status == "FAILED"
        assert "FAILED: timeout" in dep.domain_events[0].deployment_status


# ── MonitoringConfig ──────────────────────────────────────────────────

class TestMonitoringConfig:
    def test_create_defaults(self):
        mc = MonitoringConfig.create("ep-1")
        assert not mc.enabled
        assert mc.drift_threshold == 0.05
        assert mc.drift_history == ()

    def test_enable(self):
        mc = MonitoringConfig.create("ep-1").enable()
        assert mc.enabled
        assert isinstance(mc.domain_events[0], MonitoringConfiguredEvent)

    def test_record_drift(self):
        dr = DriftResult.from_test("f", "ks", DriftType.DATA, 0.3, 0.01)
        mc = MonitoringConfig.create("ep-1").record_drift(dr)
        assert len(mc.drift_history) == 1
        assert isinstance(mc.domain_events[0], DriftDetectedEvent)
        assert mc.domain_events[0].severity == "critical"

    def test_trigger_remediation(self):
        mc = MonitoringConfig.create("ep-1").trigger_remediation("rollback", "critical drift")
        event = mc.domain_events[0]
        assert isinstance(event, RemediationTriggeredEvent)
        assert event.remediation_type == "rollback"


# ── AgentTask ─────────────────────────────────────────────────────────

class TestAgentTask:
    def test_lifecycle(self):
        t = AgentTask.create("do stuff")
        assert t.status == "PENDING"
        t = t.assign("agent-1")
        assert t.status == "ASSIGNED"
        assert t.assigned_agent_id == "agent-1"
        t = t.start()
        assert t.status == "IN_PROGRESS"
        t = t.complete("done")
        assert t.status == "COMPLETED"
        assert t.result == "done"

    def test_fail(self):
        t = AgentTask.create("task").assign("a").start().fail("err")
        assert t.status == "FAILED"
        assert t.result == "err"

    def test_depends_on(self):
        t = AgentTask.create("sub", depends_on=("t1", "t2"))
        assert t.depends_on == ("t1", "t2")


# ── Agent ─────────────────────────────────────────────────────────────

class TestAgent:
    def test_create(self):
        a = Agent.create(AgentRole.DATA_ENGINEER, ("ingest",), ("create_dataset",))
        assert a.role == AgentRole.DATA_ENGINEER
        assert a.status == "IDLE"

    def test_assign_task(self):
        a = Agent.create(AgentRole.DEPLOYMENT, ("deploy",), ("deploy_vertex",))
        a = a.assign_task("task-1")
        assert a.status == "BUSY"
        assert a.current_task_id == "task-1"

    def test_assign_when_busy_raises(self):
        a = Agent.create(AgentRole.FINOPS, ("cost",), ("cost_query",)).assign_task("t1")
        with pytest.raises(ValueError, match="not idle"):
            a.assign_task("t2")

    def test_complete_task(self):
        a = Agent.create(AgentRole.SECURITY, ("sec",), ("scan",)).assign_task("t").complete_task()
        assert a.status == "IDLE"
        assert a.current_task_id == ""

    def test_can_handle(self):
        a = Agent.create(AgentRole.ARCHITECT, ("design", "review"), ("read",))
        assert a.can_handle("design")
        assert not a.can_handle("deploy")
