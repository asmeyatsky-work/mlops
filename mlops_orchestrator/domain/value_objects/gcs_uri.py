from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class GcsUri:
    """Google Cloud Storage URI value object."""
    uri: str

    def __post_init__(self) -> None:
        if not self.uri.startswith("gs://"):
            raise ValueError(f"GCS URI must start with 'gs://': {self.uri}")
        bucket = self.uri[5:].split("/")[0]
        if not bucket:
            raise ValueError(f"GCS URI must include a bucket name: {self.uri}")

    @property
    def bucket(self) -> str:
        path = self.uri[5:]  # strip gs://
        return path.split("/")[0]

    @property
    def path(self) -> str:
        path = self.uri[5:]
        parts = path.split("/", 1)
        return parts[1] if len(parts) > 1 else ""
