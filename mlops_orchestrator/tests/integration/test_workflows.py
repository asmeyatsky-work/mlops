"""Integration tests for end-to-end workflows."""
from __future__ import annotations

import pytest

from mlops_orchestrator.application.orchestration.ml_pipeline_workflow import MLPipelineWorkflow
from mlops_orchestrator.application.orchestration.dag_orchestrator import OrchestrationError
from mlops_orchestrator.application.orchestration.self_healing_workflow import SelfHealingWorkflow
from mlops_orchestrator.application.orchestration.swarm_coordinator import (
    SwarmCoordinator, OrchestrationPattern,
)
from mlops_orchestrator.application.orchestration.agent_registry import AgentRegistry
from mlops_orchestrator.application.commands.create_dataset_command import CreateDatasetCommand
from mlops_orchestrator.application.commands.train_model_command import TrainModelCommand
from mlops_orchestrator.application.commands.deploy_vertex_command import DeployToVertexCommand
from mlops_orchestrator.application.commands.configure_monitoring_command import ConfigureMonitoringCommand
from mlops_orchestrator.application.dtos.dataset_dto import CreateDatasetRequest
from mlops_orchestrator.application.dtos.training_dto import TrainModelRequest
from mlops_orchestrator.application.dtos.deployment_dto import DeployToVertexRequest, MonitoringRequest
from mlops_orchestrator.application.session.session_state import SessionState
from mlops_orchestrator.domain.entities.agent import AgentTask
from mlops_orchestrator.domain.value_objects.bq_source import BigQuerySource
from mlops_orchestrator.domain.value_objects.machine_spec import MachineSpec
from mlops_orchestrator.infrastructure.adapters.stub_dataset_adapter import StubDatasetAdapter
from mlops_orchestrator.infrastructure.adapters.stub_training_adapter import StubTrainingAdapter
from mlops_orchestrator.infrastructure.adapters.stub_deployment_adapter import (
    StubVertexDeploymentAdapter,
)
from mlops_orchestrator.infrastructure.adapters.stub_monitoring_adapter import StubMonitoringAdapter
from mlops_orchestrator.infrastructure.adapters.stub_infrastructure_adapters import (
    InMemoryEventBus, StubAuditLogAdapter,
)
from mlops_orchestrator.infrastructure.adapters.alerting_adapters import StubAlertAdapter


# ── Failing port helpers for pipeline failure tests ──────────────────


class FailingDatasetPort:
    async def create_dataset(self, bq_source: BigQuerySource, display_name: str) -> str:
        raise RuntimeError("data ingest failed")


class FailingTrainingPort:
    async def start_training(self, model_name, dataset_id, gcs_uri, train_image) -> str:
        raise RuntimeError("training failed")

    async def get_job_status(self, job_resource_name: str) -> str:
        return "UNKNOWN"

    async def get_model_resource_name(self, job_resource_name: str) -> str:
        return ""


class FailingVertexDeploymentPort:
    async def create_endpoint_and_deploy(self, model_id, endpoint_name, machine_spec) -> str:
        raise RuntimeError("deploy failed")


class FailingMonitoringPort:
    async def configure_monitoring(self, endpoint_id, drift_threshold, skew_threshold) -> bool:
        raise RuntimeError("monitoring failed")


# ── MLPipelineWorkflow ───────────────────────────────────────────────


class TestMLPipelineWorkflow:
    async def test_full_pipeline(self):
        workflow = MLPipelineWorkflow(
            dataset_port=StubDatasetAdapter(),
            training_port=StubTrainingAdapter(auto_succeed=True),
            deployment_port=StubVertexDeploymentAdapter(),
            monitoring_port=StubMonitoringAdapter(),
        )
        session = SessionState()
        result = await workflow.execute(
            bq_dataset="ds", bq_table="tbl",
            model_name="test-model", endpoint_name="test-ep",
            session=session,
        )
        assert "data_ingest" in result
        assert "train" in result
        assert "deploy" in result
        assert "monitor" in result
        assert result["monitor"] is True
        assert "stub-project" in result["data_ingest"]
        assert "stub-project" in result["train"]
        assert "stub-project" in result["deploy"]

    async def test_pipeline_failure_at_data_ingest(self):
        """Pipeline fails at data_ingest stage when dataset port raises."""
        workflow = MLPipelineWorkflow(
            dataset_port=FailingDatasetPort(),
            training_port=StubTrainingAdapter(auto_succeed=True),
            deployment_port=StubVertexDeploymentAdapter(),
            monitoring_port=StubMonitoringAdapter(),
        )
        with pytest.raises(OrchestrationError, match="data_ingest"):
            await workflow.execute(
                bq_dataset="ds", bq_table="tbl",
                model_name="m", endpoint_name="ep",
                session=SessionState(),
            )

    async def test_pipeline_failure_at_training(self):
        """Pipeline fails at train stage when training port raises."""
        workflow = MLPipelineWorkflow(
            dataset_port=StubDatasetAdapter(),
            training_port=FailingTrainingPort(),
            deployment_port=StubVertexDeploymentAdapter(),
            monitoring_port=StubMonitoringAdapter(),
        )
        with pytest.raises(OrchestrationError, match="train"):
            await workflow.execute(
                bq_dataset="ds", bq_table="tbl",
                model_name="m", endpoint_name="ep",
                session=SessionState(),
            )

    async def test_pipeline_failure_at_deploy(self):
        """Pipeline fails at deploy stage when deployment port raises."""
        workflow = MLPipelineWorkflow(
            dataset_port=StubDatasetAdapter(),
            training_port=StubTrainingAdapter(auto_succeed=True),
            deployment_port=FailingVertexDeploymentPort(),
            monitoring_port=StubMonitoringAdapter(),
        )
        with pytest.raises(OrchestrationError, match="deploy"):
            await workflow.execute(
                bq_dataset="ds", bq_table="tbl",
                model_name="m", endpoint_name="ep",
                session=SessionState(),
            )

    async def test_pipeline_failure_at_monitor(self):
        """Pipeline fails at monitor stage when monitoring port raises."""
        workflow = MLPipelineWorkflow(
            dataset_port=StubDatasetAdapter(),
            training_port=StubTrainingAdapter(auto_succeed=True),
            deployment_port=StubVertexDeploymentAdapter(),
            monitoring_port=FailingMonitoringPort(),
        )
        with pytest.raises(OrchestrationError, match="monitor"):
            await workflow.execute(
                bq_dataset="ds", bq_table="tbl",
                model_name="m", endpoint_name="ep",
                session=SessionState(),
            )


# ── SelfHealingWorkflow ──────────────────────────────────────────────


class TestSelfHealingWorkflow:
    async def test_no_drift(self):
        monitoring = StubMonitoringAdapter()
        training = StubTrainingAdapter()
        workflow = SelfHealingWorkflow(monitoring, training)
        result = await workflow.execute("ep-1")
        assert result["act"]["action"] == "none"

    async def test_critical_drift_triggers_rollback(self):
        monitoring = StubMonitoringAdapter()
        monitoring.inject_alerts([
            {"feature": "f1", "statistic": 0.5, "p_value": 0.001, "threshold": 0.05},
        ])
        deployment = StubVertexDeploymentAdapter()
        workflow = SelfHealingWorkflow(monitoring, StubTrainingAdapter(), deployment_port=deployment)
        result = await workflow.execute("ep-1")
        assert result["act"]["action"] == "rollback"
        assert result["act"]["executed"] == "true"

    async def test_moderate_drift_ensemble(self):
        monitoring = StubMonitoringAdapter()
        monitoring.inject_alerts([
            {"feature": "f1", "statistic": 0.15, "p_value": 0.01, "threshold": 0.05},
        ])
        workflow = SelfHealingWorkflow(monitoring, StubTrainingAdapter())
        result = await workflow.execute("ep-1")
        assert result["act"]["action"] == "ensemble_switching"

    async def test_high_data_drift_triggers_incremental_training(self):
        """High severity DATA drift triggers incremental_training and starts a job."""
        monitoring = StubMonitoringAdapter()
        monitoring.inject_alerts([
            {"feature": "f1", "statistic": 0.25, "p_value": 0.001, "threshold": 0.05},
        ])
        training = StubTrainingAdapter()
        workflow = SelfHealingWorkflow(monitoring, training)
        result = await workflow.execute("ep-1")
        assert result["act"]["action"] == "incremental_training"
        assert "job_resource_name" in result["act"]

    async def test_multiple_alerts(self):
        """Multiple drift alerts are all analyzed and influence the decision."""
        monitoring = StubMonitoringAdapter()
        monitoring.inject_alerts([
            {"feature": "f1", "statistic": 0.05, "p_value": 0.03, "threshold": 0.05},
            {"feature": "f2", "statistic": 0.15, "p_value": 0.02, "threshold": 0.05},
        ])
        workflow = SelfHealingWorkflow(monitoring, StubTrainingAdapter())
        result = await workflow.execute("ep-1")
        # analyze step should have processed both alerts
        assert len(result["analyze"]) == 2
        # The action should be determined by the most severe drift
        assert result["act"]["action"] != "none"

    async def test_drift_sends_alert(self):
        """When drift is detected, an alert is sent via the alerting port."""
        monitoring = StubMonitoringAdapter()
        monitoring.inject_alerts([
            {"feature": "f1", "statistic": 0.5, "p_value": 0.001, "threshold": 0.05},
        ])
        alerting = StubAlertAdapter()
        deployment = StubVertexDeploymentAdapter()
        workflow = SelfHealingWorkflow(
            monitoring, StubTrainingAdapter(),
            deployment_port=deployment,
            alerting_port=alerting,
        )
        await workflow.execute("ep-1")
        # Should have received drift detection alert + rollback alert
        assert len(alerting.sent_alerts) >= 1
        titles = [a.title for a in alerting.sent_alerts]
        assert "Model Drift Detected" in titles

    async def test_no_drift_no_alert(self):
        """When no drift is detected, no alert is sent."""
        monitoring = StubMonitoringAdapter()
        alerting = StubAlertAdapter()
        workflow = SelfHealingWorkflow(
            monitoring, StubTrainingAdapter(),
            alerting_port=alerting,
        )
        await workflow.execute("ep-1")
        assert len(alerting.sent_alerts) == 0

    async def test_rollback_sends_critical_alert(self):
        """Rollback action sends a critical alert."""
        monitoring = StubMonitoringAdapter()
        monitoring.inject_alerts([
            {"feature": "f1", "statistic": 0.5, "p_value": 0.001, "threshold": 0.05},
        ])
        alerting = StubAlertAdapter()
        deployment = StubVertexDeploymentAdapter()
        workflow = SelfHealingWorkflow(
            monitoring, StubTrainingAdapter(),
            deployment_port=deployment,
            alerting_port=alerting,
        )
        await workflow.execute("ep-1")
        rollback_alerts = [a for a in alerting.sent_alerts if a.title == "Model Rollback Executed"]
        assert len(rollback_alerts) == 1
        assert rollback_alerts[0].severity == "critical"


# ── SwarmIntegration ─────────────────────────────────────────────────


class TestSwarmIntegration:
    async def test_default_swarm_orchestrator_worker(self):
        registry = AgentRegistry.create_default_swarm()
        agents = registry.all_agents()
        tasks = [AgentTask.create(f"task-{i}") for i in range(3)]

        async def executor(agent, task):
            return f"{agent.role.value}:{task.description}"

        coord = SwarmCoordinator(agents, OrchestrationPattern.ORCHESTRATOR_WORKER)
        results = await coord.coordinate(tasks, executor)
        assert len(results) == 3
        for val in results.values():
            assert ":" in val

    async def test_default_swarm_pipeline(self):
        registry = AgentRegistry.create_default_swarm()
        agents = registry.all_agents()
        tasks = [AgentTask.create("ingest data")]

        async def executor(agent, task):
            return f"[{agent.role.value}] processed: {task.description}"

        coord = SwarmCoordinator(agents, OrchestrationPattern.PIPELINE)
        results = await coord.coordinate(tasks, executor)
        assert len(results) == 1


# ── End-to-end command chain ─────────────────────────────────────────


class TestEndToEndCommandChain:
    async def test_create_train_deploy_monitor_chain(self):
        """Full command chain: CreateDataset -> Train -> DeployVertex -> Monitor."""
        event_bus = InMemoryEventBus()
        audit_log = StubAuditLogAdapter()
        session = SessionState()

        # Step 1: Create dataset
        ds_cmd = CreateDatasetCommand(StubDatasetAdapter(), event_bus, audit_log)
        ds_req = CreateDatasetRequest(bq_dataset="ds", bq_table="tbl", name="chain-ds")
        ds_resp, session = await ds_cmd.execute(ds_req, session)
        assert ds_resp.status == "REGISTERED"
        assert session.latest_dataset == ds_resp.resource_name

        # Step 2: Train model (using the dataset from step 1)
        train_cmd = TrainModelCommand(StubTrainingAdapter(), event_bus, audit_log)
        train_req = TrainModelRequest(
            model_name="chain-model", dataset_id=session.latest_dataset
        )
        train_resp, session = await train_cmd.execute(train_req, session)
        assert train_resp.status == "RUNNING"
        assert session.latest_job == train_resp.job_resource_name

        # Step 3: Deploy to Vertex (using a model id)
        deploy_cmd = DeployToVertexCommand(
            StubVertexDeploymentAdapter(), event_bus, audit_log
        )
        deploy_req = DeployToVertexRequest(
            model_id="model-resource-1", endpoint_name="chain-ep"
        )
        deploy_resp, session = await deploy_cmd.execute(deploy_req, session)
        assert deploy_resp.status == "DEPLOYED"
        assert deploy_resp.target == "vertex"
        assert session.latest_endpoint == deploy_resp.resource_name

        # Step 4: Configure monitoring on the deployed endpoint
        mon_cmd = ConfigureMonitoringCommand(
            StubMonitoringAdapter(), event_bus, audit_log
        )
        mon_req = MonitoringRequest(endpoint_id=session.latest_endpoint)
        mon_resp, session = await mon_cmd.execute(mon_req, session)
        assert mon_resp.monitoring_enabled
        assert mon_resp.status == "ACTIVE"

        # Verify full chain state
        assert len(session.dataset_ids) == 1
        assert len(session.job_handles) == 1
        assert len(session.endpoint_names) == 1
        assert len(event_bus.published_events) > 0
        assert len(audit_log.all_entries) == 4
