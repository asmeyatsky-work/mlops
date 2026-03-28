from __future__ import annotations
from pydantic import BaseModel, Field


class CreateDatasetRequest(BaseModel):
    """Input DTO for dataset creation."""
    bq_dataset: str = Field(..., description="BigQuery dataset name")
    bq_table: str = Field(..., description="BigQuery table name")
    name: str = Field(..., description="Display name for the managed dataset")


class DatasetResponse(BaseModel):
    """Output DTO for dataset operations."""
    resource_name: str
    display_name: str
    status: str
