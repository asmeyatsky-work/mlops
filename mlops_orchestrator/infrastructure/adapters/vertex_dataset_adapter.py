from __future__ import annotations
import asyncio

from mlops_orchestrator.domain.entities.managed_dataset import ManagedDataset
from mlops_orchestrator.domain.value_objects.bq_source import BigQuerySource


class VertexDatasetAdapter:
    """Real Vertex AI dataset adapter. Implements DatasetPort."""

    def __init__(self, project: str, location: str = "us-central1") -> None:
        self._project = project
        self._location = location
        from google.cloud import aiplatform
        aiplatform.init(project=project, location=location)

    async def create_dataset(self, bq_source: BigQuerySource, display_name: str) -> str:
        from google.cloud import aiplatform

        bq_uri = bq_source.to_uri()
        dataset = await asyncio.to_thread(
            aiplatform.TabularDataset.create,
            display_name=display_name,
            bq_source=bq_uri,
        )
        return dataset.resource_name

    async def get_dataset(self, resource_name: str) -> ManagedDataset | None:
        from google.cloud import aiplatform

        try:
            dataset = await asyncio.to_thread(aiplatform.TabularDataset, resource_name)
            display_name = dataset.display_name or "unknown"
            return ManagedDataset.create(
                bq_source=BigQuerySource(dataset="vertex_managed", table=display_name),
                display_name=display_name,
            ).register(resource_name)
        except Exception:
            return None

    async def list_datasets(self) -> list[ManagedDataset]:
        from google.cloud import aiplatform

        datasets = await asyncio.to_thread(aiplatform.TabularDataset.list)
        results = []
        for ds in datasets:
            display_name = ds.display_name or "unknown"
            entity = ManagedDataset.create(
                bq_source=BigQuerySource(dataset="vertex_managed", table=display_name),
                display_name=display_name,
            ).register(ds.resource_name)
            results.append(entity)
        return results
