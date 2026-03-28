from __future__ import annotations
from dataclasses import dataclass
from mlops_orchestrator.domain.events.event_base import DomainEvent

@dataclass(frozen=True)
class DatasetCreatedEvent(DomainEvent):
    """Emitted when a managed dataset is registered."""
    bq_dataset: str = ""
    bq_table: str = ""
    resource_name: str = ""

@dataclass(frozen=True)
class DatasetValidationFailedEvent(DomainEvent):
    """Emitted when dataset validation fails."""
    reason: str = ""
