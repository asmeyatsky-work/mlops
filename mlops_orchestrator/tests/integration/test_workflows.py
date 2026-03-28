"""Integration tests for end-to-end workflows."""
from __future__ import annotations

import pytest

from mlops_orchestrator.application.orchestration.ml_pipeline_workflow import MLPipelineWorkflow
from mlops_orchestrator.application.orchestration.self_healing_workflow import SelfHealingWorkflow
from mlops_orchestrator.application.orchestration.swarm_coordinator import (
    SwarmCoordinator, OrchestrationPattern,
)
from mlops_orchestrator.application.orchestration.agent_registry import AgentRegistry
from mlops_orchestrator.application.session.session_state import SessionState
from mlops_orchestrator.domain.entities.agent import AgentTask
from mlops_orchestrator.infrastructure.adapters.stub_dataset_adapter import StubDatasetAdapter
from mlops_orchestrator.infrastructure.adapters.stub_training_adapter import StubTrainingAdapter
from mlops_orchestrator.infrastructure.adapters.stub_deployment_adapter import (
    StubVertexDeploymentAdapter,
)
from mlops_orchestrator.infrastructure.adapters.stub_monitoring_adapter import StubMonitoringAdapter


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
        workflow = SelfHealingWorkflow(monitoring, StubTrainingAdapter())
        result = await workflow.execute("ep-1")
        assert result["act"]["action"] == "rollback"

    async def test_moderate_drift_ensemble(self):
        monitoring = StubMonitoringAdapter()
        monitoring.inject_alerts([
            {"feature": "f1", "statistic": 0.15, "p_value": 0.01, "threshold": 0.05},
        ])
        workflow = SelfHealingWorkflow(monitoring, StubTrainingAdapter())
        result = await workflow.execute("ep-1")
        assert result["act"]["action"] == "ensemble_switching"


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
