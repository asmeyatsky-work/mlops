from __future__ import annotations
from dataclasses import dataclass
from mlops_orchestrator.domain.events.event_base import DomainEvent

@dataclass(frozen=True)
class ModelDeployedToVertexEvent(DomainEvent):
    """Emitted when a model is deployed to a Vertex endpoint."""
    endpoint_resource_name: str = ""

@dataclass(frozen=True)
class ModelDeployedToGkeEvent(DomainEvent):
    """Emitted when a model is deployed to GKE."""
    cluster_name: str = ""
    deployment_status: str = ""

@dataclass(frozen=True)
class ModelUndeployedEvent(DomainEvent):
    """Emitted when a model is removed from serving."""
    reason: str = ""
