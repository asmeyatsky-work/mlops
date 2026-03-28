from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine


class OrchestrationError(Exception):
    """Raised when workflow orchestration fails."""


@dataclass(frozen=True)
class WorkflowStep:
    """A single step in a DAG workflow."""
    name: str
    execute: Callable[..., Coroutine[Any, Any, Any]]
    depends_on: tuple[str, ...] = ()


class DAGOrchestrator:
    """
    Executes workflow steps respecting dependency order,
    parallelizing independent steps automatically.

    Architectural Intent:
    - Parallelism-first design per skill2026.md Rule 7
    - Steps with no dependencies on each other run concurrently via asyncio.gather
    - Steps are topologically sorted by dependency level
    - Failure in any step propagates immediately
    """

    def __init__(self, steps: list[WorkflowStep]) -> None:
        self._steps = {s.name: s for s in steps}
        self._validate_no_cycles()

    def _validate_no_cycles(self) -> None:
        visited: set[str] = set()
        in_stack: set[str] = set()

        def dfs(name: str) -> None:
            if name in in_stack:
                raise OrchestrationError(f"Circular dependency detected at step: {name}")
            if name in visited:
                return
            in_stack.add(name)
            for dep in self._steps[name].depends_on:
                if dep not in self._steps:
                    raise OrchestrationError(f"Unknown dependency: {dep}")
                dfs(dep)
            in_stack.discard(name)
            visited.add(name)

        for step_name in self._steps:
            dfs(step_name)

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute all steps, parallelizing where dependencies allow."""
        completed: dict[str, Any] = {}
        pending = set(self._steps.keys())

        while pending:
            ready = [
                name for name in pending
                if all(dep in completed for dep in self._steps[name].depends_on)
            ]
            if not ready:
                raise OrchestrationError(
                    f"Deadlock: no steps ready. Pending: {pending}"
                )

            results = await asyncio.gather(
                *(self._steps[name].execute(context, completed) for name in ready),
                return_exceptions=True,
            )

            for name, result in zip(ready, results):
                if isinstance(result, BaseException):
                    raise OrchestrationError(
                        f"Step '{name}' failed: {result}"
                    ) from result
                completed[name] = result
                pending.discard(name)

        return completed
