from __future__ import annotations
from typing import Callable, Protocol

from mlops_orchestrator.domain.events.event_base import DomainEvent
from mlops_orchestrator.domain.value_objects.cost_metrics import (
    CostBreakdown,
    CostMetrics,
    CostRecommendation,
)


class EventBusPort(Protocol):
    """Port for domain event publishing and subscription."""

    async def publish(self, events: list[DomainEvent]) -> None:
        """Publish domain events."""
        ...

    async def subscribe(
        self, event_type: type[DomainEvent], handler: Callable
    ) -> None:
        """Subscribe a handler to a specific event type."""
        ...


class AuditLogPort(Protocol):
    """Port for immutable audit logging (Cloud Logging)."""

    async def log_action(
        self, action: str, resource_id: str, details: dict[str, str]
    ) -> None:
        """Log an auditable action."""
        ...

    async def get_audit_trail(self, resource_id: str) -> list[dict[str, str]]:
        """Retrieve audit trail for a resource."""
        ...


class CostManagementPort(Protocol):
    """Port for FinOps cost tracking and optimization."""

    async def get_resource_costs(
        self, resource_id: str, start_date: str, end_date: str
    ) -> CostBreakdown:
        """Get cost breakdown for a specific resource."""
        ...

    async def get_project_metrics(self, project_id: str) -> CostMetrics:
        """Get aggregated cost metrics for a project."""
        ...

    async def get_recommendations(self, project_id: str) -> list[CostRecommendation]:
        """Get FinOps optimization recommendations."""
        ...


class SecurityPort(Protocol):
    """Port for security operations (metadata sanitization, IAM)."""

    async def sanitize_tool_metadata(
        self, tool_name: str, metadata: dict[str, str]
    ) -> dict[str, str]:
        """Strip known injection patterns from tool metadata."""
        ...

    async def validate_iam_permissions(self, required_roles: list[str]) -> bool:
        """Verify the service account has required IAM roles."""
        ...
