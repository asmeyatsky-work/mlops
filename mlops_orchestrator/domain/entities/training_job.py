from __future__ import annotations
from dataclasses import dataclass, field, replace
from datetime import datetime, UTC
from uuid import uuid4

from mlops_orchestrator.domain.events.event_base import DomainEvent
from mlops_orchestrator.domain.events.training_events import (
    TrainingJobStartedEvent,
    TrainingJobCompletedEvent,
    TrainingJobFailedEvent,
)

TRAIN_IMAGE = "us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-12:latest"

_VALID_TRANSITIONS: dict[str, set[str]] = {
    "PENDING": {"RUNNING", "FAILED"},
    "RUNNING": {"SUCCEEDED", "FAILED"},
    "SUCCEEDED": set(),
    "FAILED": set(),
}


@dataclass(frozen=True)
class TrainingJob:
    """
    Training Job aggregate root.

    Architectural Intent:
    - Represents an async CustomTrainingJob on Vertex AI
    - Status state machine: PENDING -> RUNNING -> SUCCEEDED | FAILED
    - Stores job_resource_name for polling and model_resource_name on completion
    """
    id: str
    model_name: str
    dataset_id: str = ""
    gcs_uri: str = ""
    train_image: str = TRAIN_IMAGE
    status: str = "PENDING"
    job_resource_name: str = ""
    model_resource_name: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    domain_events: tuple[DomainEvent, ...] = ()

    @classmethod
    def create(
        cls,
        model_name: str,
        dataset_id: str = "",
        gcs_uri: str = "",
        train_image: str = TRAIN_IMAGE,
    ) -> TrainingJob:
        if not dataset_id and not gcs_uri:
            raise ValueError("Either dataset_id or gcs_uri must be provided")
        return cls(
            id=str(uuid4()),
            model_name=model_name,
            dataset_id=dataset_id,
            gcs_uri=gcs_uri,
            train_image=train_image,
        )

    def _validate_transition(self, target: str) -> None:
        allowed = _VALID_TRANSITIONS.get(self.status, set())
        if target not in allowed:
            raise ValueError(
                f"Invalid state transition: {self.status} -> {target}"
            )

    def start(self, job_resource_name: str) -> TrainingJob:
        self._validate_transition("RUNNING")
        return replace(
            self,
            status="RUNNING",
            job_resource_name=job_resource_name,
            domain_events=self.domain_events + (
                TrainingJobStartedEvent(
                    aggregate_id=self.id,
                    model_name=self.model_name,
                    dataset_id=self.dataset_id,
                ),
            ),
        )

    def complete(self, model_resource_name: str) -> TrainingJob:
        self._validate_transition("SUCCEEDED")
        return replace(
            self,
            status="SUCCEEDED",
            model_resource_name=model_resource_name,
            domain_events=self.domain_events + (
                TrainingJobCompletedEvent(
                    aggregate_id=self.id,
                    model_resource_name=model_resource_name,
                ),
            ),
        )

    def fail(self, error_message: str) -> TrainingJob:
        self._validate_transition("FAILED")
        return replace(
            self,
            status="FAILED",
            domain_events=self.domain_events + (
                TrainingJobFailedEvent(
                    aggregate_id=self.id,
                    error_message=error_message,
                ),
            ),
        )

    @property
    def is_terminal(self) -> bool:
        return self.status in {"SUCCEEDED", "FAILED"}

    @property
    def is_active(self) -> bool:
        return self.status == "RUNNING"
