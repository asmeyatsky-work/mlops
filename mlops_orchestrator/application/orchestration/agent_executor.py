"""The real SwarmCoordinator task executor.

Layer: application. Imports only domain. This is the executor the
``SwarmCoordinator`` has always taken as a callback but never had a production
implementation for — until now it existed only as a mock in tests.

Routing policy (Architectural Rules §3.6, §4):
- Reasoning roles (Orchestrator, Architect, Data Engineer, Validation, FinOps)
  dispatch to the ``ReasoningAgentPort``. Their output is advisory and
  schema-validated; it never mutates state directly.
- Deterministic roles (Deployment, Security) are NEVER executed by the LLM.
  Deploys are irreversible side effects and security is policy enforcement, so
  the executor returns an explicit *deferral* marker — actuation must go through
  the typed commands and the EU AI Act compliance gate, not a reasoning agent.
"""
from __future__ import annotations

import logging

from mlops_orchestrator.domain.entities.agent import Agent, AgentRole, AgentTask
from mlops_orchestrator.domain.ports.reasoning_agent_port import ReasoningAgentPort
from mlops_orchestrator.domain.value_objects.plan import Plan, RecommendationKind

logger = logging.getLogger(__name__)

# Roles whose work is genuine reasoning — routed to the ReasoningAgentPort.
REASONING_ROLES: frozenset[AgentRole] = frozenset(
    {
        AgentRole.ORCHESTRATOR,
        AgentRole.ARCHITECT,
        AgentRole.DATA_ENGINEER,
        AgentRole.VALIDATION,
        AgentRole.FINOPS,
    }
)

# Roles that perform irreversible side effects or enforce policy — these stay
# deterministic DTOs and are actuated only by typed commands, never the LLM.
DETERMINISTIC_ROLES: frozenset[AgentRole] = frozenset(
    {AgentRole.DEPLOYMENT, AgentRole.SECURITY}
)

# Prefix the executor returns for deterministic roles so callers can recognise
# that the task was intentionally deferred (not failed).
DEFERRED_PREFIX = "DEFERRED"

_ROLE_KIND: dict[AgentRole, RecommendationKind] = {
    AgentRole.ARCHITECT: RecommendationKind.INFRASTRUCTURE,
    AgentRole.FINOPS: RecommendationKind.COST,
    AgentRole.VALIDATION: RecommendationKind.VALIDATION,
    AgentRole.DATA_ENGINEER: RecommendationKind.DATA,
    AgentRole.ORCHESTRATOR: RecommendationKind.GENERIC,
}


class AgentExecutor:
    """Bridges swarm tasks to reasoning (advisory) or deterministic execution."""

    def __init__(self, reasoning_port: ReasoningAgentPort) -> None:
        self._reasoning = reasoning_port

    async def execute(self, agent: Agent, task: AgentTask) -> str:
        """Execute a task for an agent. Matches ``SwarmCoordinator``'s executor signature."""
        if agent.role in DETERMINISTIC_ROLES:
            logger.info(
                "Deferring deterministic role %s to typed command path",
                agent.role.value,
                extra={"agent_id": agent.id},
            )
            return (
                f"{DEFERRED_PREFIX}: role '{agent.role.value}' performs irreversible "
                "actuation/policy enforcement and is not executed by reasoning agents; "
                "route through the deterministic command + compliance gate."
            )

        recommendation = await self._reasoning.advise(
            role=agent.role, task=task.description
        )
        logger.info(
            "Reasoning role %s produced %s recommendation (confidence=%.2f)",
            agent.role.value,
            recommendation.kind.value,
            recommendation.confidence,
            extra={"agent_id": agent.id},
        )
        return recommendation.summary

    async def plan(self, goal: str, available_capabilities: tuple[str, ...]) -> Plan:
        """Decompose a goal into routed tasks via the Orchestrator reasoning path."""
        return await self._reasoning.decompose(goal, available_capabilities)
