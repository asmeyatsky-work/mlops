from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, UTC

@dataclass(frozen=True)
class ModelArtifact:
    """Trained model artifact reference."""
    resource_name: str
    framework: str
    artifact_uri: str
    created_at: datetime

    @classmethod
    def create(cls, resource_name: str, framework: str, artifact_uri: str) -> ModelArtifact:
        return cls(
            resource_name=resource_name,
            framework=framework,
            artifact_uri=artifact_uri,
            created_at=datetime.now(UTC),
        )
