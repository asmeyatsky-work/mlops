from __future__ import annotations
from dataclasses import dataclass
from mlops_orchestrator.domain.events.event_base import DomainEvent

@dataclass(frozen=True)
class TrainingJobStartedEvent(DomainEvent):
    """Emitted when a training job begins execution."""
    model_name: str = ""
    dataset_id: str = ""

@dataclass(frozen=True)
class TrainingJobCompletedEvent(DomainEvent):
    """Emitted when training succeeds."""
    model_resource_name: str = ""

@dataclass(frozen=True)
class TrainingJobFailedEvent(DomainEvent):
    """Emitted when training fails."""
    error_message: str = ""
