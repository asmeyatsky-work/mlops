from __future__ import annotations
from dataclasses import dataclass, field, replace
from types import MappingProxyType


@dataclass(frozen=True)
class SessionState:
    """
    Immutable session state for input/output stitching between MCP tools.

    Architectural Intent:
    - Stores GCP resource IDs accumulated during the ML lifecycle
    - Each tool call receives current state and returns updated state
    - Enables the agent to pass outputs from one GCP component as inputs to the next
    - Shields the user from "technical plumbing"
    """
    dataset_ids: tuple[str, ...] = ()
    model_uris: tuple[str, ...] = ()
    job_handles: tuple[str, ...] = ()
    endpoint_names: tuple[str, ...] = ()
    active_project: str = ""
    _metadata: dict[str, str] = field(default_factory=dict)

    @property
    def metadata(self) -> MappingProxyType[str, str]:
        return MappingProxyType(self._metadata)

    def add_dataset(self, resource_name: str) -> SessionState:
        return replace(self, dataset_ids=self.dataset_ids + (resource_name,))

    def add_model_uri(self, model_uri: str) -> SessionState:
        return replace(self, model_uris=self.model_uris + (model_uri,))

    def add_job_handle(self, job_handle: str) -> SessionState:
        return replace(self, job_handles=self.job_handles + (job_handle,))

    def add_endpoint(self, endpoint_name: str) -> SessionState:
        return replace(self, endpoint_names=self.endpoint_names + (endpoint_name,))

    def set_project(self, project: str) -> SessionState:
        return replace(self, active_project=project)

    def set_metadata(self, key: str, value: str) -> SessionState:
        new_meta = {**self._metadata, key: value}
        return replace(self, _metadata=new_meta)

    @property
    def latest_dataset(self) -> str:
        return self.dataset_ids[-1] if self.dataset_ids else ""

    @property
    def latest_model(self) -> str:
        return self.model_uris[-1] if self.model_uris else ""

    @property
    def latest_job(self) -> str:
        return self.job_handles[-1] if self.job_handles else ""

    @property
    def latest_endpoint(self) -> str:
        return self.endpoint_names[-1] if self.endpoint_names else ""

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_ids": list(self.dataset_ids),
            "model_uris": list(self.model_uris),
            "job_handles": list(self.job_handles),
            "endpoint_names": list(self.endpoint_names),
            "active_project": self.active_project,
            "metadata": dict(self._metadata),
        }
