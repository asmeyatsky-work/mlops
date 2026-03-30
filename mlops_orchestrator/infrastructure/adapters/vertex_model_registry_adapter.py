"""Real Vertex AI Model Registry adapter with versioning support."""
from __future__ import annotations

import asyncio

from mlops_orchestrator.domain.ports.model_registry_port import ModelVersion
from mlops_orchestrator.infrastructure.adapters.retry import with_retry


class VertexModelRegistryAdapter:
    """Real Vertex AI model registry adapter. Implements ModelRegistryPort."""

    def __init__(self, project: str, location: str = "us-central1") -> None:
        self._project = project
        self._location = location
        from google.cloud import aiplatform
        aiplatform.init(project=project, location=location)

    @with_retry(max_attempts=3)
    async def register_model(
        self,
        display_name: str,
        artifact_uri: str,
        serving_container_image: str,
        description: str = "",
        labels: dict[str, str] | None = None,
    ) -> ModelVersion:
        from google.cloud import aiplatform

        # Upload model — if a model with the same display_name exists,
        # Vertex AI creates a new version automatically when parent_model is set.
        existing = await asyncio.to_thread(
            aiplatform.Model.list,
            filter=f'display_name="{display_name}"',
            order_by="create_time desc",
        )

        kwargs: dict = {
            "display_name": display_name,
            "artifact_uri": artifact_uri,
            "serving_container_image_uri": serving_container_image,
            "description": description,
            "labels": labels or {},
        }

        if existing:
            kwargs["parent_model"] = existing[0].resource_name

        model = await asyncio.to_thread(aiplatform.Model.upload, **kwargs)
        version_id = int(model.version_id) if hasattr(model, "version_id") and model.version_id else 1

        return ModelVersion(
            model_id=model.resource_name.rsplit("@", 1)[0] if "@" in model.resource_name else model.resource_name,
            version=version_id,
            resource_name=model.resource_name,
            display_name=display_name,
            description=description,
            labels=labels,
            stage="development",
        )

    @with_retry(max_attempts=3)
    async def get_model_version(
        self, model_id: str, version: int | None = None
    ) -> ModelVersion | None:
        from google.cloud import aiplatform

        try:
            version_str = f"@{version}" if version else ""
            model = await asyncio.to_thread(
                aiplatform.Model, f"{model_id}{version_str}"
            )
            vid = int(model.version_id) if hasattr(model, "version_id") and model.version_id else 1
            return ModelVersion(
                model_id=model_id,
                version=vid,
                resource_name=model.resource_name,
                display_name=model.display_name or "",
                description=model.description or "",
                labels=dict(model.labels) if model.labels else None,
            )
        except Exception:
            return None

    @with_retry(max_attempts=3)
    async def list_versions(self, model_id: str) -> list[ModelVersion]:
        from google.cloud import aiplatform

        try:
            model = await asyncio.to_thread(aiplatform.Model, model_id)
            versions = await asyncio.to_thread(model.list_versions)
            result = []
            for v in versions:
                vid = int(v.version_id) if hasattr(v, "version_id") and v.version_id else 1
                result.append(
                    ModelVersion(
                        model_id=model_id,
                        version=vid,
                        resource_name=v.resource_name,
                        display_name=v.display_name or "",
                        description=v.description or "",
                        labels=dict(v.labels) if v.labels else None,
                    )
                )
            return result
        except Exception:
            return []

    @with_retry(max_attempts=3)
    async def promote_version(
        self, model_id: str, version: int, stage: str
    ) -> ModelVersion:
        from google.cloud import aiplatform

        model = await asyncio.to_thread(aiplatform.Model, f"{model_id}@{version}")
        labels = dict(model.labels) if model.labels else {}
        labels["stage"] = stage
        await asyncio.to_thread(model.update, labels=labels)
        return ModelVersion(
            model_id=model_id,
            version=version,
            resource_name=model.resource_name,
            display_name=model.display_name or "",
            description=model.description or "",
            labels=labels,
            stage=stage,
        )

    @with_retry(max_attempts=3)
    async def compare_versions(
        self, model_id: str, version_a: int, version_b: int
    ) -> dict[str, object]:
        va = await self.get_model_version(model_id, version_a)
        vb = await self.get_model_version(model_id, version_b)
        return {
            "version_a": {"version": version_a, "labels": va.labels if va else None, "stage": va.stage if va else None},
            "version_b": {"version": version_b, "labels": vb.labels if vb else None, "stage": vb.stage if vb else None},
        }
