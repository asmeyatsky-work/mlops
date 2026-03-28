"""Tests for infrastructure stub adapters."""
from __future__ import annotations

import pytest

from mlops_orchestrator.domain.value_objects.bq_source import BigQuerySource
from mlops_orchestrator.domain.value_objects.machine_spec import MachineSpec
from mlops_orchestrator.domain.value_objects.cost_metrics import CostMetrics, CostRecommendation
from mlops_orchestrator.domain.events.event_base import DomainEvent
from mlops_orchestrator.domain.events.dataset_events import DatasetCreatedEvent
from mlops_orchestrator.infrastructure.adapters.stub_dataset_adapter import StubDatasetAdapter
from mlops_orchestrator.infrastructure.adapters.stub_training_adapter import StubTrainingAdapter
from mlops_orchestrator.infrastructure.adapters.stub_deployment_adapter import (
    StubVertexDeploymentAdapter, StubGkeDeploymentAdapter,
)
from mlops_orchestrator.infrastructure.adapters.stub_monitoring_adapter import StubMonitoringAdapter
from mlops_orchestrator.infrastructure.adapters.stub_infrastructure_adapters import (
    InMemoryEventBus, StubAuditLogAdapter, StubCostAdapter, StubSecurityAdapter,
)


class TestStubDatasetAdapter:
    async def test_create_and_get(self):
        adapter = StubDatasetAdapter()
        bq = BigQuerySource(dataset="ds", table="tbl")
        rn = await adapter.create_dataset(bq, "test-ds")
        assert "stub-project" in rn
        ds = await adapter.get_dataset(rn)
        assert ds is not None
        assert ds.status == "REGISTERED"

    async def test_list_datasets(self):
        adapter = StubDatasetAdapter()
        bq = BigQuerySource(dataset="ds", table="tbl")
        await adapter.create_dataset(bq, "a")
        await adapter.create_dataset(bq, "b")
        all_ds = await adapter.list_datasets()
        assert len(all_ds) == 2

    async def test_get_nonexistent(self):
        adapter = StubDatasetAdapter()
        assert await adapter.get_dataset("nonexistent") is None


class TestStubTrainingAdapter:
    async def test_auto_succeed(self):
        adapter = StubTrainingAdapter(auto_succeed=True)
        rn = await adapter.start_training("m", "ds", "", "img")
        assert await adapter.get_job_status(rn) == "SUCCEEDED"
        assert await adapter.get_model_resource_name(rn) != ""

    async def test_running_mode(self):
        adapter = StubTrainingAdapter(auto_succeed=False)
        rn = await adapter.start_training("m", "ds", "", "img")
        assert await adapter.get_job_status(rn) == "RUNNING"

    async def test_cancel_job(self):
        adapter = StubTrainingAdapter(auto_succeed=False)
        rn = await adapter.start_training("m", "ds", "", "img")
        assert await adapter.cancel_job(rn)
        assert await adapter.get_job_status(rn) == "CANCELLED"

    async def test_cancel_nonexistent(self):
        adapter = StubTrainingAdapter()
        assert not await adapter.cancel_job("nope")

    async def test_status_unknown_job(self):
        adapter = StubTrainingAdapter()
        assert await adapter.get_job_status("nope") == "UNKNOWN"


class TestStubVertexDeploymentAdapter:
    async def test_create_and_get_status(self):
        adapter = StubVertexDeploymentAdapter()
        rn = await adapter.create_endpoint_and_deploy("m", "ep", MachineSpec())
        assert await adapter.get_endpoint_status(rn) == "DEPLOYED"

    async def test_undeploy(self):
        adapter = StubVertexDeploymentAdapter()
        rn = await adapter.create_endpoint_and_deploy("m", "ep", MachineSpec())
        await adapter.undeploy(rn)
        assert await adapter.get_endpoint_status(rn) == "UNDEPLOYED"

    async def test_unknown_endpoint(self):
        adapter = StubVertexDeploymentAdapter()
        assert await adapter.get_endpoint_status("nope") == "UNKNOWN"


class TestStubGkeDeploymentAdapter:
    async def test_deploy_and_status(self):
        adapter = StubGkeDeploymentAdapter()
        result = await adapter.deploy("models/my-model", "cluster-1", 2)
        assert result["status"] == "DEPLOYED"
        dep_name = result["deployment_name"]
        assert await adapter.get_deployment_status("cluster-1", dep_name) == "DEPLOYED"

    async def test_delete(self):
        adapter = StubGkeDeploymentAdapter()
        result = await adapter.deploy("models/m", "c", 2)
        await adapter.delete_deployment("c", result["deployment_name"])
        assert await adapter.get_deployment_status("c", result["deployment_name"]) == "UNKNOWN"


class TestStubMonitoringAdapter:
    async def test_configure_and_status(self):
        adapter = StubMonitoringAdapter()
        assert await adapter.configure_monitoring("ep-1", 0.05, 0.1)
        status = await adapter.get_monitoring_status("ep-1")
        assert status["status"] == "ACTIVE"

    async def test_not_configured(self):
        adapter = StubMonitoringAdapter()
        status = await adapter.get_monitoring_status("nope")
        assert status["status"] == "NOT_CONFIGURED"

    async def test_inject_alerts(self):
        adapter = StubMonitoringAdapter()
        alerts = [{"feature": "f1", "statistic": 0.5, "p_value": 0.01}]
        adapter.inject_alerts(alerts)
        result = await adapter.get_drift_alerts("ep")
        assert len(result) == 1

    async def test_auto_fail_mode(self):
        """StubMonitoringAdapter with auto_fail=True returns False from configure_monitoring."""
        adapter = StubMonitoringAdapter(auto_fail=True)
        success = await adapter.configure_monitoring("ep-1", 0.05, 0.1)
        assert success is False
        # auto_fail should not store the config
        status = await adapter.get_monitoring_status("ep-1")
        assert status["status"] == "NOT_CONFIGURED"


class TestInMemoryEventBus:
    async def test_publish_and_read(self):
        bus = InMemoryEventBus()
        event = DomainEvent(aggregate_id="agg-1")
        await bus.publish([event])
        assert len(bus.published_events) == 1
        assert bus.published_events[0].aggregate_id == "agg-1"

    async def test_subscribe_and_handle(self):
        bus = InMemoryEventBus()
        received = []

        async def handler(e):
            received.append(e)

        await bus.subscribe(DomainEvent, handler)
        await bus.publish([DomainEvent(aggregate_id="x")])
        assert len(received) == 1

    def test_clear(self):
        bus = InMemoryEventBus()
        # Sync clear test
        bus._published.append(DomainEvent(aggregate_id="x"))
        bus.clear()
        assert len(bus.published_events) == 0

    async def test_publish_empty_list(self):
        """Publishing an empty list should not raise and adds no events."""
        bus = InMemoryEventBus()
        await bus.publish([])
        assert len(bus.published_events) == 0

    async def test_subclass_event_dispatch(self):
        """Handler subscribed to DatasetCreatedEvent is called for that subclass,
        but handler subscribed to DomainEvent is NOT called for DatasetCreatedEvent
        (dispatch is by exact type, not by hierarchy)."""
        bus = InMemoryEventBus()
        base_received: list[DomainEvent] = []
        sub_received: list[DomainEvent] = []

        async def base_handler(e):
            base_received.append(e)

        async def sub_handler(e):
            sub_received.append(e)

        await bus.subscribe(DomainEvent, base_handler)
        await bus.subscribe(DatasetCreatedEvent, sub_handler)

        dataset_event = DatasetCreatedEvent(aggregate_id="ds-1", resource_name="rn")
        await bus.publish([dataset_event])

        # The subclass handler must be called
        assert len(sub_received) == 1
        # The base handler is keyed on DomainEvent (exact type), not on DatasetCreatedEvent
        assert len(base_received) == 0


class TestStubAuditLogAdapter:
    async def test_log_and_trail(self):
        adapter = StubAuditLogAdapter()
        await adapter.log_action("create", "r1", {"key": "val"})
        await adapter.log_action("delete", "r2", {})
        trail = await adapter.get_audit_trail("r1")
        assert len(trail) == 1
        assert trail[0]["action"] == "create"
        assert len(adapter.all_entries) == 2


class TestStubCostAdapter:
    async def test_defaults(self):
        adapter = StubCostAdapter()
        metrics = await adapter.get_project_metrics("p")
        assert metrics.cost_per_tb_scanned == 0.0

    async def test_custom_metrics(self):
        adapter = StubCostAdapter(metrics=CostMetrics(gpu_idle_pct=50.0))
        m = await adapter.get_project_metrics("p")
        assert m.gpu_idle_pct == 50.0

    async def test_resource_costs(self):
        adapter = StubCostAdapter()
        breakdown = await adapter.get_resource_costs("r", "2024-01-01", "2024-01-31")
        assert breakdown.total == 12.5


class TestStubSecurityAdapter:
    async def test_sanitize(self):
        adapter = StubSecurityAdapter()
        result = await adapter.sanitize_tool_metadata("tool", {"key": "v", "__secret": "x"})
        assert "key" in result
        assert "__secret" not in result

    async def test_validate_iam(self):
        adapter = StubSecurityAdapter()
        assert await adapter.validate_iam_permissions(["admin"])
