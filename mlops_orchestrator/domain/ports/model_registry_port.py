"""Port for model versioning and registry operations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ModelVersion:
    """A versioned model in the registry."""
    model_id: str
    version: int
    resource_name: str
    display_name: str
    description: str = ""
    labels: dict[str, str] | None = None
    stage: str = "development"  # development, staging, production, archived


class ModelRegistryPort(Protocol):
    """Port for model versioning and lifecycle management."""

    async def register_model(
        self,
        display_name: str,
        artifact_uri: str,
        serving_container_image: str,
        description: str = "",
        labels: dict[str, str] | None = None,
    ) -> ModelVersion:
        """Register a new model or create a new version of an existing model."""
        ...

    async def get_model_version(
        self, model_id: str, version: int | None = None
    ) -> ModelVersion | None:
        """Get a specific model version. If version is None, returns latest."""
        ...

    async def list_versions(self, model_id: str) -> list[ModelVersion]:
        """List all versions of a model."""
        ...

    async def promote_version(
        self, model_id: str, version: int, stage: str
    ) -> ModelVersion:
        """Promote a model version to a lifecycle stage (staging, production, archived)."""
        ...

    async def compare_versions(
        self, model_id: str, version_a: int, version_b: int
    ) -> dict[str, object]:
        """Compare two model versions (metadata, lineage, labels)."""
        ...
