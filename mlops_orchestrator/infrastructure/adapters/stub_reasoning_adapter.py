"""Deterministic ReasoningAgentPort adapter.

Layer: infrastructure (adapter). Implements ``ReasoningAgentPort`` with no
network and no LLM — it produces schema-valid ``Plan`` / ``Recommendation``
value objects so the reasoning seam can be exercised end-to-end in tests and in
``MLOPS_USE_STUBS`` mode without GCP/Gemini credentials. This is the in-memory
adapter the testing floor (§5) requires for the new port.
"""
from __future__ import annotations

from typing import Mapping

from mlops_orchestrator.domain.entities.agent import AgentRole
from mlops_orchestrator.domain.value_objects.plan import (
    Plan,
    PlannedTask,
    Recommendation,
    RecommendationKind,
)

_ROLE_KIND: dict[AgentRole, RecommendationKind] = {
    AgentRole.ARCHITECT: RecommendationKind.INFRASTRUCTURE,
    AgentRole.FINOPS: RecommendationKind.COST,
    AgentRole.VALIDATION: RecommendationKind.VALIDATION,
    AgentRole.DATA_ENGINEER: RecommendationKind.DATA,
    AgentRole.ORCHESTRATOR: RecommendationKind.GENERIC,
}


class StubReasoningAdapter:
    """Deterministic stand-in for a real reasoning engine."""

    async def decompose(
        self, goal: str, available_capabilities: tuple[str, ...]
    ) -> Plan:
        capabilities = available_capabilities or ("coordination",)
        tasks = tuple(
            PlannedTask(
                description=f"Handle goal via '{cap}': {goal}",
                capability=cap,
            )
            for cap in capabilities
        )
        return Plan(
            goal=goal,
            tasks=tasks,
            rationale="stub decomposition — one task per available capability",
        )

    async def advise(
        self,
        role: AgentRole,
        task: str,
        context: Mapping[str, str] | None = None,
    ) -> Recommendation:
        kind = _ROLE_KIND.get(role, RecommendationKind.GENERIC)
        return Recommendation(
            kind=kind,
            summary=f"[{role.value}] reviewed task: {task}",
            actions=(f"apply {role.value} best practice for: {task}",),
            confidence=0.5,
            rationale="stub advisory output — deterministic, no LLM call",
        )
