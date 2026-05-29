"""Tests for AgentExecutor routing (reasoning vs deterministic roles)."""
from __future__ import annotations

import pytest

from mlops_orchestrator.application.orchestration.agent_executor import (
    AgentExecutor,
    DEFERRED_PREFIX,
    DETERMINISTIC_ROLES,
    REASONING_ROLES,
)
from mlops_orchestrator.application.orchestration.swarm_coordinator import (
    OrchestrationPattern,
    SwarmCoordinator,
)
from mlops_orchestrator.domain.entities.agent import Agent, AgentRole, AgentTask
from mlops_orchestrator.domain.value_objects.plan import (
    Plan,
    PlannedTask,
    Recommendation,
    RecommendationKind,
)


class _RecordingReasoningPort:
    """In-memory ReasoningAgentPort double that records calls."""

    def __init__(self) -> None:
        self.advise_calls: list[tuple[AgentRole, str]] = []
        self.decompose_calls: list[tuple[str, tuple[str, ...]]] = []

    async def advise(self, role, task, context=None):
        self.advise_calls.append((role, task))
        return Recommendation(
            kind=RecommendationKind.GENERIC,
            summary=f"advice for {role.value}: {task}",
            confidence=0.9,
        )

    async def decompose(self, goal, available_capabilities):
        self.decompose_calls.append((goal, available_capabilities))
        return Plan(
            goal=goal,
            tasks=(PlannedTask(description=goal, capability=available_capabilities[0]),),
        )


def _agent(role: AgentRole) -> Agent:
    return Agent.create(role=role, capabilities=("x",), permitted_tools=())


class TestAgentExecutor:
    async def test_reasoning_role_calls_advise(self):
        port = _RecordingReasoningPort()
        ex = AgentExecutor(port)
        result = await ex.execute(_agent(AgentRole.FINOPS), AgentTask.create("cut GPU cost"))
        assert "advice for finops" in result
        assert port.advise_calls == [(AgentRole.FINOPS, "cut GPU cost")]

    @pytest.mark.parametrize("role", sorted(DETERMINISTIC_ROLES, key=lambda r: r.value))
    async def test_deterministic_roles_are_deferred_not_executed(self, role):
        port = _RecordingReasoningPort()
        ex = AgentExecutor(port)
        result = await ex.execute(_agent(role), AgentTask.create("do it"))
        assert result.startswith(DEFERRED_PREFIX)
        assert port.advise_calls == []  # LLM never invoked for these roles

    @pytest.mark.parametrize("role", sorted(REASONING_ROLES, key=lambda r: r.value))
    async def test_all_reasoning_roles_invoke_port(self, role):
        port = _RecordingReasoningPort()
        ex = AgentExecutor(port)
        await ex.execute(_agent(role), AgentTask.create("task"))
        assert len(port.advise_calls) == 1

    async def test_plan_delegates_to_decompose(self):
        port = _RecordingReasoningPort()
        ex = AgentExecutor(port)
        plan = await ex.plan("train a model", ("etl", "training"))
        assert plan.goal == "train a model"
        assert port.decompose_calls == [("train a model", ("etl", "training"))]

    async def test_wires_as_swarm_executor(self):
        """AgentExecutor.execute satisfies the SwarmCoordinator executor contract."""
        port = _RecordingReasoningPort()
        ex = AgentExecutor(port)
        agents = [_agent(AgentRole.ARCHITECT), _agent(AgentRole.FINOPS)]
        coord = SwarmCoordinator(agents, OrchestrationPattern.ORCHESTRATOR_WORKER)
        tasks = [AgentTask.create("a"), AgentTask.create("b")]
        results = await coord.coordinate(tasks, ex.execute)
        assert len(results) == 2
        assert all("advice for" in v for v in results.values())

    def test_reasoning_and_deterministic_roles_are_disjoint_and_total(self):
        assert REASONING_ROLES.isdisjoint(DETERMINISTIC_ROLES)
        assert REASONING_ROLES | DETERMINISTIC_ROLES == set(AgentRole)
