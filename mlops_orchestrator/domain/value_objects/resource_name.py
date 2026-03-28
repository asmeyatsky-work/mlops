from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class ResourceName:
    """GCP resource name value object."""
    project: str
    location: str
    resource_type: str
    resource_id: str

    @property
    def full_name(self) -> str:
        return f"projects/{self.project}/locations/{self.location}/{self.resource_type}/{self.resource_id}"

    @classmethod
    def from_string(cls, name: str) -> ResourceName:
        """Parse 'projects/{p}/locations/{l}/{type}/{id}' format."""
        parts = name.split("/")
        if len(parts) < 6 or parts[0] != "projects" or parts[2] != "locations":
            raise ValueError(f"Invalid resource name format: {name}")
        return cls(
            project=parts[1],
            location=parts[3],
            resource_type=parts[4],
            resource_id="/".join(parts[5:]),
        )

    def __str__(self) -> str:
        return self.full_name
