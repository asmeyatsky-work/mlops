from __future__ import annotations
from dataclasses import dataclass, field, replace
from datetime import datetime, UTC
from uuid import uuid4

from mlops_orchestrator.domain.events.event_base import DomainEvent
from mlops_orchestrator.domain.events.dataset_events import (
    DatasetCreatedEvent,
    DatasetValidationFailedEvent,
)
from mlops_orchestrator.domain.value_objects.bq_source import BigQuerySource


@dataclass(frozen=True)
class ManagedDataset:
    """
    Managed Dataset aggregate root.

    Architectural Intent:
    - Wraps a Vertex AI Managed Dataset backed by BigQuery
    - Tracks lifecycle from creation to registration
    - Emits domain events for cross-context communication
    """
    id: str
    display_name: str
    bq_source: BigQuerySource
    resource_name: str = ""
    status: str = "PENDING"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    domain_events: tuple[DomainEvent, ...] = ()

    @classmethod
    def create(cls, bq_source: BigQuerySource, display_name: str) -> ManagedDataset:
        dataset_id = str(uuid4())
        return cls(
            id=dataset_id,
            display_name=display_name,
            bq_source=bq_source,
        )

    def register(self, resource_name: str) -> ManagedDataset:
        return replace(
            self,
            resource_name=resource_name,
            status="REGISTERED",
            domain_events=self.domain_events + (
                DatasetCreatedEvent(
                    aggregate_id=self.id,
                    bq_dataset=self.bq_source.dataset,
                    bq_table=self.bq_source.table,
                    resource_name=resource_name,
                ),
            ),
        )

    def fail_validation(self, reason: str) -> ManagedDataset:
        return replace(
            self,
            status="VALIDATION_FAILED",
            domain_events=self.domain_events + (
                DatasetValidationFailedEvent(
                    aggregate_id=self.id,
                    reason=reason,
                ),
            ),
        )
