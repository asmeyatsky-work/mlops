"""Port for batch prediction operations."""
from __future__ import annotations

from typing import Protocol


class BatchPredictionPort(Protocol):
    """Port for Vertex AI Batch Prediction Jobs."""

    async def start_batch_prediction(
        self,
        model_resource_name: str,
        input_uri: str,
        output_uri: str,
        instance_type: str,
    ) -> str:
        """Submit a batch prediction job. Returns job resource_name."""
        ...

    async def get_job_status(self, job_resource_name: str) -> str:
        """Get the current status of a batch prediction job."""
        ...

    async def get_job_output_uri(self, job_resource_name: str) -> str:
        """Get the output URI for a completed batch prediction job."""
        ...

    async def cancel_job(self, job_resource_name: str) -> bool:
        """Cancel a running batch prediction job. Returns success."""
        ...
