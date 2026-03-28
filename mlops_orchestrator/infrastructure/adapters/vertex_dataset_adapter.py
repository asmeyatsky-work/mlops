from __future__ import annotations
from google.cloud import aiplatform

from mlops_orchestrator.domain.entities.managed_dataset import ManagedDataset
from mlops_orchestrator.domain.value_objects.bq_source import BigQuerySource


class VertexDatasetAdapter:
    """Real Vertex AI dataset adapter. Implements DatasetPort."""

    def __init__(self, project: str, location: str = "us-central1") -> None:
        self._project = project
        self._location = location
        aiplatform.init(project=project, location=location)

    async def create_dataset(self, bq_source: BigQuerySource, display_name: str) -> str:
        bq_uri = bq_source.to_uri()
        dataset = aiplatform.TabularDataset.create(
            display_name=display_name,
            bq_source=bq_uri,
        )
        return dataset.resource_name

    async def get_dataset(self, resource_name: str) -> ManagedDataset | None:
        try:
            dataset = aiplatform.TabularDataset(resource_name)
            return ManagedDataset(
                id=dataset.name,
                display_name=dataset.display_name,
                bq_source=BigQuerySource(dataset="", table=""),
                resource_name=dataset.resource_name,
                status="REGISTERED",
            )
        except Exception:
            return None

    async def list_datasets(self) -> list[ManagedDataset]:
        datasets = aiplatform.TabularDataset.list()
        return [
            ManagedDataset(
                id=ds.name,
                display_name=ds.display_name,
                bq_source=BigQuerySource(dataset="", table=""),
                resource_name=ds.resource_name,
                status="REGISTERED",
            )
            for ds in datasets
        ]
