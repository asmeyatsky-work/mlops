from __future__ import annotations
from pydantic import BaseModel, Field


class CreateDatasetRequest(BaseModel):
    """Input DTO for dataset creation."""
    bq_dataset: str = Field(..., min_length=1, description="BigQuery dataset name")
    bq_table: str = Field(..., min_length=1, description="BigQuery table name")
    name: str = Field(..., min_length=1, description="Display name for the managed dataset")


class DatasetResponse(BaseModel):
    """Output DTO for dataset operations."""
    resource_name: str
    display_name: str
    status: str
