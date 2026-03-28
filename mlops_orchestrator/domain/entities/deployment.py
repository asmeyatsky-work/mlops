from __future__ import annotations
from dataclasses import dataclass, field, replace
from datetime import datetime, UTC
from uuid import uuid4

from mlops_orchestrator.domain.events.event_base import DomainEvent
from mlops_orchestrator.domain.events.deployment_events import (
    ModelDeployedToVertexEvent,
    ModelDeployedToGkeEvent,
    GkeDeploymentFailedEvent,
    ModelUndeployedEvent,
)
from mlops_orchestrator.domain.value_objects.machine_spec import MachineSpec


_VERTEX_VALID_TRANSITIONS: dict[str, set[str]] = {
    "PENDING": {"DEPLOYED"},
    "DEPLOYED": {"UNDEPLOYED"},
    "UNDEPLOYED": set(),
}

_GKE_VALID_TRANSITIONS: dict[str, set[str]] = {
    "PENDING": {"DEPLOYED", "FAILED"},
    "DEPLOYED": set(),
    "FAILED": set(),
}


@dataclass(frozen=True)
class VertexDeployment:
    """
    Vertex AI Endpoint deployment aggregate root.

    Architectural Intent:
    - Manages model deployment to Vertex Endpoints (auto-scaling)
    - Tracks machine spec and monitoring state
    """
    id: str
    model_id: str
    endpoint_name: str
    endpoint_resource_name: str = ""
    machine_spec: MachineSpec = field(default_factory=MachineSpec)
    monitoring_enabled: bool = False
    status: str = "PENDING"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    domain_events: tuple[DomainEvent, ...] = ()

    @classmethod
    def create(
        cls,
        model_id: str,
        endpoint_name: str,
        machine_spec: MachineSpec | None = None,
    ) -> VertexDeployment:
        return cls(
            id=str(uuid4()),
            model_id=model_id,
            endpoint_name=endpoint_name,
            machine_spec=machine_spec or MachineSpec(),
        )

    def _validate_transition(self, target: str) -> None:
        allowed = _VERTEX_VALID_TRANSITIONS.get(self.status, set())
        if target not in allowed:
            raise ValueError(f"Invalid state transition: {self.status} -> {target}")

    def deploy(self, endpoint_resource_name: str) -> VertexDeployment:
        self._validate_transition("DEPLOYED")
        return replace(
            self,
            endpoint_resource_name=endpoint_resource_name,
            status="DEPLOYED",
            domain_events=self.domain_events + (
                ModelDeployedToVertexEvent(
                    aggregate_id=self.id,
                    endpoint_resource_name=endpoint_resource_name,
                ),
            ),
        )

    def enable_monitoring(self) -> VertexDeployment:
        return replace(self, monitoring_enabled=True)

    def undeploy(self, reason: str) -> VertexDeployment:
        self._validate_transition("UNDEPLOYED")
        return replace(
            self,
            status="UNDEPLOYED",
            domain_events=self.domain_events + (
                ModelUndeployedEvent(aggregate_id=self.id, reason=reason),
            ),
        )


@dataclass(frozen=True)
class GkeDeployment:
    """
    GKE deployment aggregate root.

    Architectural Intent:
    - Manages model deployment to GKE clusters (custom runtime)
    - Default 2 replicas per PRD spec
    """
    id: str
    model_id: str
    cluster_name: str
    replica_count: int = 2
    status: str = "PENDING"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    domain_events: tuple[DomainEvent, ...] = ()

    @classmethod
    def create(
        cls,
        model_id: str,
        cluster_name: str,
        replica_count: int = 2,
    ) -> GkeDeployment:
        return cls(
            id=str(uuid4()),
            model_id=model_id,
            cluster_name=cluster_name,
            replica_count=replica_count,
        )

    def _validate_transition(self, target: str) -> None:
        allowed = _GKE_VALID_TRANSITIONS.get(self.status, set())
        if target not in allowed:
            raise ValueError(f"Invalid state transition: {self.status} -> {target}")

    def mark_deployed(self) -> GkeDeployment:
        self._validate_transition("DEPLOYED")
        return replace(
            self,
            status="DEPLOYED",
            domain_events=self.domain_events + (
                ModelDeployedToGkeEvent(
                    aggregate_id=self.id,
                    cluster_name=self.cluster_name,
                    deployment_status="DEPLOYED",
                ),
            ),
        )

    def mark_failed(self, reason: str) -> GkeDeployment:
        self._validate_transition("FAILED")
        return replace(
            self,
            status="FAILED",
            domain_events=self.domain_events + (
                GkeDeploymentFailedEvent(
                    aggregate_id=self.id,
                    cluster_name=self.cluster_name,
                    reason=reason,
                ),
            ),
        )
