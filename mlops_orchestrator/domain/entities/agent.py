from __future__ import annotations
from dataclasses import dataclass, field, replace
from datetime import datetime, UTC
from enum import Enum
from uuid import uuid4

from mlops_orchestrator.domain.events.event_base import DomainEvent


class AgentRole(Enum):
    ORCHESTRATOR = "orchestrator"
    ARCHITECT = "architect"
    DATA_ENGINEER = "data_engineer"
    VALIDATION = "validation"
    DEPLOYMENT = "deployment"
    FINOPS = "finops"
    SECURITY = "security"


@dataclass(frozen=True)
class AgentTask:
    """A task assigned to a specialist agent."""
    id: str
    description: str
    assigned_agent_id: str = ""
    status: str = "PENDING"
    result: str = ""
    depends_on: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def create(cls, description: str, depends_on: tuple[str, ...] = ()) -> AgentTask:
        return cls(id=str(uuid4()), description=description, depends_on=depends_on)

    def assign(self, agent_id: str) -> AgentTask:
        return replace(self, assigned_agent_id=agent_id, status="ASSIGNED")

    def start(self) -> AgentTask:
        return replace(self, status="IN_PROGRESS")

    def complete(self, result: str) -> AgentTask:
        return replace(self, status="COMPLETED", result=result)

    def fail(self, error: str) -> AgentTask:
        return replace(self, status="FAILED", result=error)


@dataclass(frozen=True)
class Agent:
    """
    Specialist agent in the MLOps swarm.

    Architectural Intent:
    - Each agent has a narrow role with specific capabilities
    - Least-privilege: only permitted_tools are accessible
    - Agents are domain entities, orchestration is in application layer
    """
    id: str
    role: AgentRole
    capabilities: tuple[str, ...] = ()
    permitted_tools: tuple[str, ...] = ()
    status: str = "IDLE"
    current_task_id: str = ""
    domain_events: tuple[DomainEvent, ...] = ()

    @classmethod
    def create(cls, role: AgentRole, capabilities: tuple[str, ...], permitted_tools: tuple[str, ...]) -> Agent:
        return cls(
            id=str(uuid4()),
            role=role,
            capabilities=capabilities,
            permitted_tools=permitted_tools,
        )

    def assign_task(self, task_id: str) -> Agent:
        if self.status != "IDLE":
            raise ValueError(f"Agent {self.id} is not idle (status={self.status})")
        return replace(self, status="BUSY", current_task_id=task_id)

    def complete_task(self) -> Agent:
        return replace(self, status="IDLE", current_task_id="")

    def can_handle(self, capability: str) -> bool:
        return capability in self.capabilities
