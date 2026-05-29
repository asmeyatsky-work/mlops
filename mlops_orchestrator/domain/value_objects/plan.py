"""Validated outputs of reasoning (LLM) agents.

Layer: domain (value objects). Imports nothing from infrastructure,
presentation, or any SDK.

These are the *schema-validated* representations of what a reasoning agent
proposes — a decomposition (`Plan`) or an advisory (`Recommendation`). They are
immutable and enforce their invariants in ``__post_init__`` (factories never
setters), so a malformed LLM output can never become a domain object. The
infrastructure ADK adapter parses raw model JSON against a Pydantic schema and
maps it onto these types; if the mapping fails, no state is ever mutated
(Architectural Rules §4: "AI output that mutates state must be validated against
an explicit schema first").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4


class RecommendationKind(Enum):
    INFRASTRUCTURE = "infrastructure"
    COST = "cost"
    VALIDATION = "validation"
    DATA = "data"
    GENERIC = "generic"


@dataclass(frozen=True)
class PlannedTask:
    """A single unit of work proposed by a decomposition.

    ``capability`` is the capability a specialist must advertise to handle the
    task — it is how the swarm routes work, mirroring ``Agent.capabilities``.
    """

    description: str
    capability: str
    depends_on: tuple[str, ...] = ()
    id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        if not self.description.strip():
            raise ValueError("PlannedTask.description must be non-empty")
        if not self.capability.strip():
            raise ValueError("PlannedTask.capability must be non-empty")


@dataclass(frozen=True)
class Plan:
    """A goal decomposed into routed tasks — produced by the Orchestrator role."""

    goal: str
    tasks: tuple[PlannedTask, ...]
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.goal.strip():
            raise ValueError("Plan.goal must be non-empty")
        if not self.tasks:
            raise ValueError("Plan must contain at least one task")


@dataclass(frozen=True)
class Recommendation:
    """An advisory output — produced by Architect / FinOps / Validation / Data roles.

    Advisory only: a recommendation never mutates state. The deterministic
    command layer decides whether and how to act on it.
    """

    kind: RecommendationKind
    summary: str
    actions: tuple[str, ...] = ()
    confidence: float = 0.0
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.summary.strip():
            raise ValueError("Recommendation.summary must be non-empty")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Recommendation.confidence must be within [0.0, 1.0], got {self.confidence}"
            )
