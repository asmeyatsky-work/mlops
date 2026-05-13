from __future__ import annotations
import asyncio
from enum import Enum
from typing import Any, Callable, Coroutine

from mlops_orchestrator.domain.entities.agent import Agent, AgentRole, AgentTask


class OrchestrationPattern(Enum):
    ORCHESTRATOR_WORKER = "orchestrator_worker"
    SWARM = "swarm"
    HIERARCHICAL = "hierarchical"
    MESH = "mesh"
    PIPELINE = "pipeline"


class SwarmCoordinator:
    """
    Coordinates multiple specialist agents using configurable interaction patterns.

    Patterns:
    - Orchestrator-Worker: Centralized fan-out/fan-in (routine decomposition)
    - Swarm: Shared blackboard / peer handoffs (exploration)
    - Hierarchical: Tree-structured delegation (multi-domain enterprise)
    - Mesh: Direct agent-to-agent chatter (iterative refinement)
    - Pipeline: Fixed sequential stages with I/O contracts (ETL)
    """

    def __init__(
        self,
        agents: list[Agent],
        pattern: OrchestrationPattern = OrchestrationPattern.ORCHESTRATOR_WORKER,
    ) -> None:
        self._agents = {a.id: a for a in agents}
        self._pattern = pattern

    async def coordinate(
        self,
        tasks: list[AgentTask],
        task_executor: Callable[[Agent, AgentTask], Coroutine[Any, Any, str]],
    ) -> dict[str, str]:
        executors = {
            OrchestrationPattern.ORCHESTRATOR_WORKER: self._orchestrator_worker,
            OrchestrationPattern.SWARM: self._swarm,
            OrchestrationPattern.HIERARCHICAL: self._hierarchical,
            OrchestrationPattern.MESH: self._mesh,
            OrchestrationPattern.PIPELINE: self._pipeline,
        }
        return await executors[self._pattern](tasks, task_executor)

    async def _orchestrator_worker(
        self,
        tasks: list[AgentTask],
        executor: Callable[[Agent, AgentTask], Coroutine[Any, Any, str]],
    ) -> dict[str, str]:
        """Fan-out tasks to available agents, fan-in results."""
        results: dict[str, str] = {}
        available = [a for a in self._agents.values() if a.status == "IDLE"]
        remaining = list(tasks)

        while remaining and available:
            batch_size = min(len(remaining), len(available))
            assignments = list(zip(available[:batch_size], remaining[:batch_size]))
            remaining = remaining[batch_size:]
            available = available[batch_size:]

            task_results = await asyncio.gather(
                *(executor(agent, task) for agent, task in assignments),
                return_exceptions=True,
            )

            for (agent, task), result in zip(assignments, task_results):
                if isinstance(result, BaseException):
                    results[task.id] = f"FAILED: {result}"
                else:
                    results[task.id] = result

        for task in remaining:
            results[task.id] = "FAILED: no available agents"

        return results

    async def _swarm(
        self,
        tasks: list[AgentTask],
        executor: Callable[[Agent, AgentTask], Coroutine[Any, Any, str]],
    ) -> dict[str, str]:
        """Shared blackboard — all agents can pick up any available task."""
        results: dict[str, str] = {}
        task_queue = list(tasks)
        agents = list(self._agents.values())

        if not agents:
            for task in task_queue:
                results[task.id] = "FAILED: no agents"
            return results

        while task_queue:
            batch = task_queue[:len(agents)]
            task_queue = task_queue[len(agents):]
            pairs = list(zip(agents[:len(batch)], batch))
            batch_results = await asyncio.gather(
                *(executor(a, t) for a, t in pairs),
                return_exceptions=True,
            )
            for (_, task), result in zip(pairs, batch_results):
                results[task.id] = str(result) if isinstance(result, BaseException) else result
        return results

    async def _hierarchical(
        self,
        tasks: list[AgentTask],
        executor: Callable[[Agent, AgentTask], Coroutine[Any, Any, str]],
    ) -> dict[str, str]:
        """Tree delegation: orchestrator delegates to specialists."""
        agents_list = list(self._agents.values())
        if not agents_list:
            return {t.id: "FAILED: no agents" for t in tasks}

        orchestrator = next(
            (a for a in agents_list if a.role == AgentRole.ORCHESTRATOR),
            agents_list[0],
        )
        results: dict[str, str] = {}
        for task in tasks:
            specialist = self._find_specialist(task)
            agent = specialist or orchestrator
            try:
                result = await executor(agent, task)
                results[task.id] = result
            except Exception as e:
                results[task.id] = f"FAILED: {e}"
        return results

    async def _mesh(
        self,
        tasks: list[AgentTask],
        executor: Callable[[Agent, AgentTask], Coroutine[Any, Any, str]],
    ) -> dict[str, str]:
        """Iterative refinement with direct agent-to-agent communication."""
        results: dict[str, str] = {}
        agents = list(self._agents.values())
        if not agents:
            return {t.id: "FAILED: no agents" for t in tasks}

        for task in tasks:
            current_result = ""
            for agent in agents[:3]:  # max 3 refinement passes
                refined_task = AgentTask.create(
                    description=f"{task.description}\nPrevious: {current_result}"
                ) if current_result else task
                try:
                    current_result = await executor(agent, refined_task)
                except Exception as e:
                    current_result = f"FAILED: {e}"
                    break
            results[task.id] = current_result
        return results

    async def _pipeline(
        self,
        tasks: list[AgentTask],
        executor: Callable[[Agent, AgentTask], Coroutine[Any, Any, str]],
    ) -> dict[str, str]:
        """Sequential stages with I/O contracts."""
        results: dict[str, str] = {}
        agents = list(self._agents.values())
        if not agents:
            return {t.id: "FAILED: no agents" for t in tasks}

        for task in tasks:
            stage_input = task.description
            for i, agent in enumerate(agents):
                stage_task = AgentTask.create(description=stage_input)
                try:
                    stage_input = await executor(agent, stage_task)
                except Exception as e:
                    stage_input = f"FAILED: {e}"
                    break
            results[task.id] = stage_input
        return results

    @staticmethod
    def _tokens(text: str) -> set[str]:
        """Whole-word tokens — defeats embedded-substring matching attacks
        like a crafted description that contains 'security' as part of a
        larger noun phrase to coerce specialist routing."""
        import re

        return {t for t in re.split(r"\W+", text.lower()) if t}

    def _find_specialist(self, task: AgentTask) -> Agent | None:
        task_tokens = self._tokens(task.description)
        for agent in self._agents.values():
            for cap in agent.capabilities:
                if cap.lower() in task_tokens:
                    return agent
        return None
