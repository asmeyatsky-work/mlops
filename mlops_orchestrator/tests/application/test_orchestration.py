"""Tests for application orchestration layer."""
from __future__ import annotations

import pytest

from mlops_orchestrator.application.orchestration.dag_orchestrator import (
    DAGOrchestrator, WorkflowStep, OrchestrationError,
)
from mlops_orchestrator.application.orchestration.agent_registry import AgentRegistry
from mlops_orchestrator.application.orchestration.swarm_coordinator import (
    SwarmCoordinator, OrchestrationPattern,
)
from mlops_orchestrator.domain.entities.agent import Agent, AgentRole, AgentTask


# ── DAGOrchestrator ───────────────────────────────────────────────────

class TestDAGOrchestrator:
    async def test_single_step(self):
        async def step_fn(ctx, completed):
            return "done"

        dag = DAGOrchestrator([WorkflowStep("a", step_fn)])
        result = await dag.execute({})
        assert result == {"a": "done"}

    async def test_sequential_steps(self):
        async def step_a(ctx, completed):
            return 1

        async def step_b(ctx, completed):
            return completed["a"] + 1

        dag = DAGOrchestrator([
            WorkflowStep("a", step_a),
            WorkflowStep("b", step_b, depends_on=("a",)),
        ])
        result = await dag.execute({})
        assert result == {"a": 1, "b": 2}

    async def test_parallel_steps(self):
        call_order = []

        async def step_a(ctx, completed):
            call_order.append("a")
            return "a"

        async def step_b(ctx, completed):
            call_order.append("b")
            return "b"

        dag = DAGOrchestrator([
            WorkflowStep("a", step_a),
            WorkflowStep("b", step_b),
        ])
        result = await dag.execute({})
        assert set(result.keys()) == {"a", "b"}

    async def test_diamond_dag(self):
        async def s1(ctx, c): return 1
        async def s2(ctx, c): return c["s1"] + 10
        async def s3(ctx, c): return c["s1"] + 20
        async def s4(ctx, c): return c["s2"] + c["s3"]

        dag = DAGOrchestrator([
            WorkflowStep("s1", s1),
            WorkflowStep("s2", s2, depends_on=("s1",)),
            WorkflowStep("s3", s3, depends_on=("s1",)),
            WorkflowStep("s4", s4, depends_on=("s2", "s3")),
        ])
        result = await dag.execute({})
        assert result["s4"] == 32  # (1+10) + (1+20)

    def test_cycle_detection(self):
        async def noop(ctx, c): return None
        with pytest.raises(OrchestrationError, match="Circular dependency"):
            DAGOrchestrator([
                WorkflowStep("a", noop, depends_on=("b",)),
                WorkflowStep("b", noop, depends_on=("a",)),
            ])

    def test_unknown_dependency(self):
        async def noop(ctx, c): return None
        with pytest.raises(OrchestrationError, match="Unknown dependency"):
            DAGOrchestrator([
                WorkflowStep("a", noop, depends_on=("nonexistent",)),
            ])

    async def test_step_failure_propagates(self):
        async def fail(ctx, c):
            raise RuntimeError("boom")

        dag = DAGOrchestrator([WorkflowStep("a", fail)])
        with pytest.raises(OrchestrationError, match="Step 'a' failed"):
            await dag.execute({})


# ── AgentRegistry ─────────────────────────────────────────────────────

class TestAgentRegistry:
    def test_register_and_get(self):
        registry = AgentRegistry()
        agent = Agent.create(AgentRole.DATA_ENGINEER, ("etl",), ("create_dataset",))
        registry.register(agent)
        assert registry.get_by_id(agent.id) == agent
        assert registry.get_by_id("nonexistent") is None

    def test_get_by_role(self):
        registry = AgentRegistry()
        a1 = Agent.create(AgentRole.DATA_ENGINEER, ("etl",), ())
        a2 = Agent.create(AgentRole.SECURITY, ("iam",), ())
        registry.register(a1)
        registry.register(a2)
        assert len(registry.get_by_role(AgentRole.DATA_ENGINEER)) == 1
        assert len(registry.get_by_role(AgentRole.FINOPS)) == 0

    def test_get_available(self):
        registry = AgentRegistry()
        a1 = Agent.create(AgentRole.DATA_ENGINEER, ("etl",), ())
        a2 = Agent.create(AgentRole.SECURITY, ("iam",), ()).assign_task("t")
        registry.register(a1)
        registry.register(a2)
        avail = registry.get_available()
        assert len(avail) == 1
        assert avail[0].id == a1.id

    def test_get_for_capability(self):
        registry = AgentRegistry()
        a1 = Agent.create(AgentRole.DATA_ENGINEER, ("etl", "sql"), ())
        a2 = Agent.create(AgentRole.SECURITY, ("iam",), ())
        registry.register(a1)
        registry.register(a2)
        assert len(registry.get_for_capability("sql")) == 1
        assert len(registry.get_for_capability("deploy")) == 0

    def test_create_default_swarm(self):
        registry = AgentRegistry.create_default_swarm()
        agents = registry.all_agents()
        assert len(agents) == 7
        roles = {a.role for a in agents}
        assert AgentRole.ORCHESTRATOR in roles
        assert AgentRole.DATA_ENGINEER in roles
        assert AgentRole.SECURITY in roles


# ── SwarmCoordinator ──────────────────────────────────────────────────

def _make_agents(n: int = 3) -> list[Agent]:
    roles = list(AgentRole)
    return [
        Agent.create(roles[i % len(roles)], ("general",), ())
        for i in range(n)
    ]


def _make_tasks(n: int = 2) -> list[AgentTask]:
    return [AgentTask.create(f"task-{i}") for i in range(n)]


async def _mock_executor(agent: Agent, task: AgentTask) -> str:
    return f"result-{task.id[:8]}"


class TestSwarmCoordinator:
    async def test_orchestrator_worker(self):
        agents = _make_agents(3)
        tasks = _make_tasks(2)
        coord = SwarmCoordinator(agents, OrchestrationPattern.ORCHESTRATOR_WORKER)
        results = await coord.coordinate(tasks, _mock_executor)
        assert len(results) == 2
        for tid in results:
            assert results[tid].startswith("result-")

    async def test_swarm_pattern(self):
        agents = _make_agents(2)
        tasks = _make_tasks(3)
        coord = SwarmCoordinator(agents, OrchestrationPattern.SWARM)
        results = await coord.coordinate(tasks, _mock_executor)
        assert len(results) == 3

    async def test_hierarchical_pattern(self):
        agents = _make_agents(3)
        tasks = _make_tasks(2)
        coord = SwarmCoordinator(agents, OrchestrationPattern.HIERARCHICAL)
        results = await coord.coordinate(tasks, _mock_executor)
        assert len(results) == 2

    async def test_mesh_pattern(self):
        agents = _make_agents(3)
        tasks = _make_tasks(1)
        coord = SwarmCoordinator(agents, OrchestrationPattern.MESH)
        results = await coord.coordinate(tasks, _mock_executor)
        assert len(results) == 1

    async def test_pipeline_pattern(self):
        agents = _make_agents(2)
        tasks = _make_tasks(1)
        coord = SwarmCoordinator(agents, OrchestrationPattern.PIPELINE)
        results = await coord.coordinate(tasks, _mock_executor)
        assert len(results) == 1

    async def test_orchestrator_worker_more_tasks_than_agents(self):
        agents = _make_agents(1)
        tasks = _make_tasks(3)
        coord = SwarmCoordinator(agents, OrchestrationPattern.ORCHESTRATOR_WORKER)
        results = await coord.coordinate(tasks, _mock_executor)
        assert len(results) == 1  # only 1 agent available

    async def test_executor_failure_captured(self):
        async def failing_executor(agent, task):
            raise RuntimeError("agent crashed")

        agents = _make_agents(2)
        tasks = _make_tasks(1)
        coord = SwarmCoordinator(agents, OrchestrationPattern.ORCHESTRATOR_WORKER)
        results = await coord.coordinate(tasks, failing_executor)
        assert "FAILED" in list(results.values())[0]
