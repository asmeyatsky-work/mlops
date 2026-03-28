from __future__ import annotations

from mlops_orchestrator.domain.entities.managed_dataset import ManagedDataset
from mlops_orchestrator.domain.value_objects.bq_source import BigQuerySource


class DatasetDomainService:
    """Pure domain logic for dataset management."""

    def create_managed_dataset(
        self, bq_dataset: str, bq_table: str, display_name: str, project: str = ""
    ) -> ManagedDataset:
        bq_source = BigQuerySource(dataset=bq_dataset, table=bq_table, project=project)
        return ManagedDataset.create(bq_source=bq_source, display_name=display_name)

    def validate_bq_source(self, bq_source: BigQuerySource) -> list[str]:
        errors: list[str] = []
        if not bq_source.dataset:
            errors.append("BigQuery dataset name is required")
        if not bq_source.table:
            errors.append("BigQuery table name is required")
        if "." in bq_source.dataset:
            errors.append("Dataset name must not contain dots")
        if "." in bq_source.table:
            errors.append("Table name must not contain dots")
        return errors
