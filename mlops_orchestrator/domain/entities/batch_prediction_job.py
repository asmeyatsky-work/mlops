"""Batch Prediction Job aggregate root."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, UTC
from uuid import uuid4

from mlops_orchestrator.domain.events.event_base import DomainEvent

_VALID_TRANSITIONS: dict[str, set[str]] = {
    "PENDING": {"RUNNING", "FAILED"},
    "RUNNING": {"SUCCEEDED", "FAILED"},
    "SUCCEEDED": set(),
    "FAILED": set(),
}


@dataclass(frozen=True)
class BatchPredictionJob:
    """
    Batch Prediction Job aggregate root.

    Tracks a Vertex AI BatchPredictionJob through its lifecycle.
    """
    id: str
    model_resource_name: str
    input_uri: str
    output_uri: str
    instance_type: str = "jsonl"
    status: str = "PENDING"
    job_resource_name: str = ""
    output_location: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    domain_events: tuple[DomainEvent, ...] = ()

    @classmethod
    def create(
        cls,
        model_resource_name: str,
        input_uri: str,
        output_uri: str,
        instance_type: str = "jsonl",
    ) -> BatchPredictionJob:
        return cls(
            id=str(uuid4()),
            model_resource_name=model_resource_name,
            input_uri=input_uri,
            output_uri=output_uri,
            instance_type=instance_type,
        )

    def _validate_transition(self, target: str) -> None:
        allowed = _VALID_TRANSITIONS.get(self.status, set())
        if target not in allowed:
            raise ValueError(f"Invalid state transition: {self.status} -> {target}")

    def start(self, job_resource_name: str) -> BatchPredictionJob:
        self._validate_transition("RUNNING")
        return replace(
            self,
            status="RUNNING",
            job_resource_name=job_resource_name,
        )

    def complete(self, output_location: str) -> BatchPredictionJob:
        self._validate_transition("SUCCEEDED")
        return replace(
            self,
            status="SUCCEEDED",
            output_location=output_location,
        )

    def fail(self) -> BatchPredictionJob:
        self._validate_transition("FAILED")
        return replace(self, status="FAILED")

    @property
    def is_terminal(self) -> bool:
        return self.status in {"SUCCEEDED", "FAILED"}
