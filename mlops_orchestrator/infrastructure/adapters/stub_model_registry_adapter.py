"""Stub model registry adapter for testing."""
from __future__ import annotations

from mlops_orchestrator.domain.ports.model_registry_port import ModelVersion


class StubModelRegistryAdapter:
    """In-memory model registry adapter. Implements ModelRegistryPort."""

    def __init__(self) -> None:
        self._models: dict[str, list[ModelVersion]] = {}

    async def register_model(
        self,
        display_name: str,
        artifact_uri: str,
        serving_container_image: str,
        description: str = "",
        labels: dict[str, str] | None = None,
    ) -> ModelVersion:
        model_id = f"projects/stub-project/locations/us-central1/models/{display_name}"
        versions = self._models.get(model_id, [])
        next_version = len(versions) + 1
        resource_name = f"{model_id}@{next_version}"

        version = ModelVersion(
            model_id=model_id,
            version=next_version,
            resource_name=resource_name,
            display_name=display_name,
            description=description,
            labels=labels,
            stage="development",
        )
        self._models.setdefault(model_id, []).append(version)
        return version

    async def get_model_version(
        self, model_id: str, version: int | None = None
    ) -> ModelVersion | None:
        versions = self._models.get(model_id, [])
        if not versions:
            return None
        if version is None:
            return versions[-1]
        for v in versions:
            if v.version == version:
                return v
        return None

    async def list_versions(self, model_id: str) -> list[ModelVersion]:
        return list(self._models.get(model_id, []))

    async def promote_version(
        self, model_id: str, version: int, stage: str
    ) -> ModelVersion:
        versions = self._models.get(model_id, [])
        for i, v in enumerate(versions):
            if v.version == version:
                promoted = ModelVersion(
                    model_id=v.model_id,
                    version=v.version,
                    resource_name=v.resource_name,
                    display_name=v.display_name,
                    description=v.description,
                    labels=v.labels,
                    stage=stage,
                )
                versions[i] = promoted
                return promoted
        raise ValueError(f"Version {version} not found for model {model_id}")

    async def compare_versions(
        self, model_id: str, version_a: int, version_b: int
    ) -> dict[str, object]:
        va = await self.get_model_version(model_id, version_a)
        vb = await self.get_model_version(model_id, version_b)
        return {
            "version_a": {"version": version_a, "labels": va.labels if va else None, "stage": va.stage if va else None},
            "version_b": {"version": version_b, "labels": vb.labels if vb else None, "stage": vb.stage if vb else None},
        }
