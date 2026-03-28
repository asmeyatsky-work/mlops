from __future__ import annotations
from uuid import uuid4

from mlops_orchestrator.domain.entities.managed_dataset import ManagedDataset
from mlops_orchestrator.domain.value_objects.bq_source import BigQuerySource


class StubDatasetAdapter:
    """In-memory dataset adapter for testing. Implements DatasetPort."""

    def __init__(self) -> None:
        self._datasets: dict[str, ManagedDataset] = {}

    async def create_dataset(self, bq_source: BigQuerySource, display_name: str) -> str:
        resource_name = f"projects/stub-project/locations/us-central1/datasets/{uuid4().hex[:8]}"
        dataset = ManagedDataset.create(bq_source=bq_source, display_name=display_name)
        dataset = dataset.register(resource_name)
        self._datasets[resource_name] = dataset
        return resource_name

    async def get_dataset(self, resource_name: str) -> ManagedDataset | None:
        return self._datasets.get(resource_name)

    async def list_datasets(self) -> list[ManagedDataset]:
        return list(self._datasets.values())
