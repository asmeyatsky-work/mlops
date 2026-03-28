from __future__ import annotations
from typing import Protocol

from mlops_orchestrator.domain.entities.managed_dataset import ManagedDataset
from mlops_orchestrator.domain.value_objects.bq_source import BigQuerySource


class DatasetPort(Protocol):
    """Port for managed dataset operations (Vertex AI / BigQuery)."""

    async def create_dataset(self, bq_source: BigQuerySource, display_name: str) -> str:
        """Create a managed dataset. Returns resource_name."""
        ...

    async def get_dataset(self, resource_name: str) -> ManagedDataset | None:
        """Retrieve a dataset by resource name."""
        ...

    async def list_datasets(self) -> list[ManagedDataset]:
        """List all managed datasets."""
        ...
