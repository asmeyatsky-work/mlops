from __future__ import annotations
from typing import Callable

from mlops_orchestrator.domain.events.event_base import DomainEvent
from mlops_orchestrator.domain.value_objects.cost_metrics import (
    CostBreakdown,
    CostMetrics,
    CostRecommendation,
)


class InMemoryEventBus:
    """In-memory event bus for testing. Implements EventBusPort."""

    def __init__(self) -> None:
        self._published: list[DomainEvent] = []
        self._handlers: dict[type, list[Callable]] = {}

    async def publish(self, events: list[DomainEvent]) -> None:
        self._published.extend(events)
        for event in events:
            for handler in self._handlers.get(type(event), []):
                await handler(event)

    async def subscribe(
        self, event_type: type[DomainEvent], handler: Callable
    ) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    @property
    def published_events(self) -> list[DomainEvent]:
        return list(self._published)

    def clear(self) -> None:
        self._published.clear()


class StubAuditLogAdapter:
    """In-memory audit log for testing. Implements AuditLogPort."""

    def __init__(self) -> None:
        self._entries: list[dict[str, str]] = []

    async def log_action(
        self, action: str, resource_id: str, details: dict[str, str]
    ) -> None:
        self._entries.append({
            "action": action,
            "resource_id": resource_id,
            **details,
        })

    async def get_audit_trail(self, resource_id: str) -> list[dict[str, str]]:
        return [e for e in self._entries if e.get("resource_id") == resource_id]

    @property
    def all_entries(self) -> list[dict[str, str]]:
        return list(self._entries)


class StubCostAdapter:
    """In-memory cost adapter for testing. Implements CostManagementPort."""

    def __init__(
        self,
        metrics: CostMetrics | None = None,
        recommendations: list[CostRecommendation] | None = None,
    ) -> None:
        self._metrics = metrics or CostMetrics()
        self._recommendations = recommendations or []

    async def get_resource_costs(
        self, resource_id: str, start_date: str, end_date: str
    ) -> CostBreakdown:
        return CostBreakdown(compute_cost=10.0, storage_cost=2.0, network_cost=0.5)

    async def get_project_metrics(self, project_id: str) -> CostMetrics:
        return self._metrics

    async def get_recommendations(self, project_id: str) -> list[CostRecommendation]:
        return self._recommendations


class StubSecurityAdapter:
    """Pass-through security adapter for testing. Implements SecurityPort."""

    async def sanitize_tool_metadata(
        self, tool_name: str, metadata: dict[str, str]
    ) -> dict[str, str]:
        return {k: v for k, v in metadata.items() if not k.startswith("__")}

    async def validate_iam_permissions(self, required_roles: list[str]) -> bool:
        return True
