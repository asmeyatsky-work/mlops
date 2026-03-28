from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class BigQuerySource:
    """BigQuery table reference value object."""
    dataset: str
    table: str
    project: str = ""

    def __post_init__(self) -> None:
        if not self.dataset:
            raise ValueError("BigQuery dataset name cannot be empty")
        if not self.table:
            raise ValueError("BigQuery table name cannot be empty")

    def to_uri(self) -> str:
        if self.project:
            return f"bq://{self.project}.{self.dataset}.{self.table}"
        return f"bq://{self.dataset}.{self.table}"
