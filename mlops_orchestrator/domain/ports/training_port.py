from __future__ import annotations
from typing import Protocol


class TrainingPort(Protocol):
    """Port for model training operations (Vertex AI CustomTrainingJob)."""

    async def start_training(
        self,
        model_name: str,
        dataset_id: str,
        gcs_uri: str,
        train_image: str,
    ) -> str:
        """Submit a training job. Returns job resource_name."""
        ...

    async def get_job_status(self, job_resource_name: str) -> str:
        """Get the current status of a training job."""
        ...

    async def get_model_resource_name(self, job_resource_name: str) -> str:
        """Get the trained model resource_name from a completed job."""
        ...

    async def cancel_job(self, job_resource_name: str) -> bool:
        """Cancel a running training job. Returns success."""
        ...
