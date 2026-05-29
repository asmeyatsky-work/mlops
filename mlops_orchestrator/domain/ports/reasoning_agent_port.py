"""Port for LLM-backed reasoning agents.

Layer: domain (port). Adapters live in infrastructure and implement this
Protocol — the deterministic ``StubReasoningAdapter`` (tests / stub mode) and
the ``ADKReasoningAdapter`` (Google Agent Development Kit, live Gemini).

The domain ``Agent`` entity stays a DTO: it carries the *contract* (role,
capabilities, least-privilege ``permitted_tools``). This port is the seam where
that contract is handed to an actual reasoning engine. Reasoning is advisory —
the methods return validated value objects; they never perform side effects.
"""
from __future__ import annotations

from typing import Mapping, Protocol

from mlops_orchestrator.domain.entities.agent import AgentRole
from mlops_orchestrator.domain.value_objects.plan import Plan, Recommendation


class ReasoningAgentPort(Protocol):
    """LLM reasoning behind a port. Implementations must be schema-validating."""

    async def decompose(
        self, goal: str, available_capabilities: tuple[str, ...]
    ) -> Plan:
        """Decompose a natural-language goal into routed tasks (Orchestrator role).

        Each returned task's ``capability`` should be drawn from
        ``available_capabilities`` so the swarm can route it to a specialist.
        """
        ...

    async def advise(
        self,
        role: AgentRole,
        task: str,
        context: Mapping[str, str] | None = None,
    ) -> Recommendation:
        """Produce an advisory recommendation for ``task`` from ``role``'s perspective.

        Advisory only — the result is validated and returned, never actuated.
        """
        ...
