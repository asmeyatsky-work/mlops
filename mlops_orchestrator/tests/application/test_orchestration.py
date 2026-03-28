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

    async def test_empty_dag(self):
        """DAG with no steps should return empty results."""
        dag = DAGOrchestrator([])
        result = await dag.execute({})
        assert result == {}

    def test_self_cycle_detection(self):
        """A step that depends on itself should raise OrchestrationError."""
        async def noop(ctx, c): return None
        with pytest.raises(OrchestrationError, match="Circular dependency"):
            DAGOrchestrator([
                WorkflowStep("a", noop, depends_on=("a",)),
            ])


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

    def test_duplicate_register_overwrites(self):
        """Registering an agent with the same id overwrites the previous one."""
        registry = AgentRegistry()
        agent_v1 = Agent.create(AgentRole.DATA_ENGINEER, ("etl",), ())
        # Manually create a second agent with the same id but different capabilities
        agent_v2 = Agent(
            id=agent_v1.id,
            role=AgentRole.DATA_ENGINEER,
            capabilities=("sql", "bigquery"),
            permitted_tools=(),
        )
        registry.register(agent_v1)
        registry.register(agent_v2)
        # Only one agent with that id
        assert len(registry.all_agents()) == 1
        retrieved = registry.get_by_id(agent_v1.id)
        assert retrieved is not None
        assert retrieved.capabilities == ("sql", "bigquery")

    def test_create_default_swarm_capabilities_verification(self):
        """Default swarm agents have expected capabilities for key roles."""
        registry = AgentRegistry.create_default_swarm()
        # Data engineer should have etl
        data_engineers = registry.get_by_role(AgentRole.DATA_ENGINEER)
        assert len(data_engineers) == 1
        assert "etl" in data_engineers[0].capabilities

        # Deployment agent should have kubernetes
        deployers = registry.get_by_role(AgentRole.DEPLOYMENT)
        assert len(deployers) == 1
        assert "kubernetes" in deployers[0].capabilities

        # Security agent should have iam
        sec_agents = registry.get_by_role(AgentRole.SECURITY)
        assert len(sec_agents) == 1
        assert "iam" in sec_agents[0].capabilities

        # Orchestrator should have coordination
        orchestrators = registry.get_by_role(AgentRole.ORCHESTRATOR)
        assert len(orchestrators) == 1
        assert "coordination" in orchestrators[0].capabilities


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
        """With 1 agent and 3 tasks, only 1 gets processed; the other 2 are dropped as FAILED."""
        agents = _make_agents(1)
        tasks = _make_tasks(3)
        coord = SwarmCoordinator(agents, OrchestrationPattern.ORCHESTRATOR_WORKER)
        results = await coord.coordinate(tasks, _mock_executor)
        assert len(results) == 3
        succeeded = [v for v in results.values() if v.startswith("result-")]
        failed = [v for v in results.values() if "FAILED" in v]
        assert len(succeeded) == 1
        assert len(failed) == 2

    async def test_executor_failure_captured(self):
        async def failing_executor(agent, task):
            raise RuntimeError("agent crashed")

        agents = _make_agents(2)
        tasks = _make_tasks(1)
        coord = SwarmCoordinator(agents, OrchestrationPattern.ORCHESTRATOR_WORKER)
        results = await coord.coordinate(tasks, failing_executor)
        assert "FAILED" in list(results.values())[0]

    async def test_no_agents_orchestrator_worker(self):
        """No agents in orchestrator-worker pattern: all tasks FAILED."""
        tasks = _make_tasks(2)
        coord = SwarmCoordinator([], OrchestrationPattern.ORCHESTRATOR_WORKER)
        results = await coord.coordinate(tasks, _mock_executor)
        assert len(results) == 2
        for v in results.values():
            assert "FAILED" in v

    async def test_no_agents_swarm(self):
        """No agents in swarm pattern: all tasks FAILED."""
        tasks = _make_tasks(2)
        coord = SwarmCoordinator([], OrchestrationPattern.SWARM)
        results = await coord.coordinate(tasks, _mock_executor)
        assert len(results) == 2
        for v in results.values():
            assert "FAILED" in v

    async def test_no_agents_hierarchical(self):
        """No agents in hierarchical pattern: all tasks FAILED."""
        tasks = _make_tasks(2)
        coord = SwarmCoordinator([], OrchestrationPattern.HIERARCHICAL)
        results = await coord.coordinate(tasks, _mock_executor)
        assert len(results) == 2
        for v in results.values():
            assert "FAILED" in v

    async def test_no_agents_mesh(self):
        """No agents in mesh pattern: all tasks FAILED."""
        tasks = _make_tasks(1)
        coord = SwarmCoordinator([], OrchestrationPattern.MESH)
        results = await coord.coordinate(tasks, _mock_executor)
        assert len(results) == 1
        for v in results.values():
            assert "FAILED" in v

    async def test_no_agents_pipeline(self):
        """No agents in pipeline pattern: all tasks FAILED."""
        tasks = _make_tasks(1)
        coord = SwarmCoordinator([], OrchestrationPattern.PIPELINE)
        results = await coord.coordinate(tasks, _mock_executor)
        assert len(results) == 1
        for v in results.values():
            assert "FAILED" in v

    async def test_no_tasks_returns_empty(self):
        """No tasks for any pattern returns empty results."""
        agents = _make_agents(2)
        for pattern in OrchestrationPattern:
            coord = SwarmCoordinator(agents, pattern)
            results = await coord.coordinate([], _mock_executor)
            assert results == {}

    async def test_hierarchical_executor_failure(self):
        """Hierarchical pattern captures executor failure as FAILED."""
        async def failing_executor(agent, task):
            raise RuntimeError("specialist crashed")

        agents = _make_agents(3)
        tasks = _make_tasks(2)
        coord = SwarmCoordinator(agents, OrchestrationPattern.HIERARCHICAL)
        results = await coord.coordinate(tasks, failing_executor)
        assert len(results) == 2
        for v in results.values():
            assert "FAILED" in v
