"""Command to register a model in the model registry."""
from __future__ import annotations

from mlops_orchestrator.domain.ports.infrastructure_ports import AuditLogPort
from mlops_orchestrator.domain.ports.model_registry_port import (
    ModelRegistryPort,
    ModelVersion,
)


class ModelRegistryCommand:
    """Register a model, creating a new version if it already exists."""

    def __init__(
        self,
        registry_port: ModelRegistryPort,
        audit_log: AuditLogPort,
    ) -> None:
        self._registry_port = registry_port
        self._audit_log = audit_log

    async def execute(
        self,
        display_name: str,
        artifact_uri: str,
        serving_container_image: str,
        description: str = "",
        labels: dict[str, str] | None = None,
    ) -> ModelVersion:
        version = await self._registry_port.register_model(
            display_name=display_name,
            artifact_uri=artifact_uri,
            serving_container_image=serving_container_image,
            description=description,
            labels=labels,
        )

        await self._audit_log.log_action(
            action="model_registered",
            resource_id=version.resource_name,
            details={
                "model_id": version.model_id,
                "version": str(version.version),
                "display_name": display_name,
            },
        )

        return version
