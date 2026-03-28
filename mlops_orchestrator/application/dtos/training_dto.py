from __future__ import annotations
from pydantic import BaseModel, Field, model_validator


class TrainModelRequest(BaseModel):
    """Input DTO for model training."""
    model_name: str = Field(..., min_length=1, description="Display name for the model")
    dataset_id: str = Field(default="", description="Managed dataset resource name")
    gcs_uri: str = Field(default="", description="GCS URI for raw training data")

    @model_validator(mode="after")
    def check_data_source(self) -> TrainModelRequest:
        if not self.dataset_id and not self.gcs_uri:
            raise ValueError("Either dataset_id or gcs_uri must be provided")
        return self


class TrainingResponse(BaseModel):
    """Output DTO for training operations."""
    job_resource_name: str
    model_name: str
    status: str
