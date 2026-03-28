from __future__ import annotations
from uuid import uuid4


class StubTrainingAdapter:
    """In-memory training adapter for testing. Implements TrainingPort."""

    def __init__(self, auto_succeed: bool = True) -> None:
        self._jobs: dict[str, dict[str, str]] = {}
        self._auto_succeed = auto_succeed

    async def start_training(
        self, model_name: str, dataset_id: str, gcs_uri: str, train_image: str
    ) -> str:
        job_id = uuid4().hex[:8]
        resource_name = f"projects/stub-project/locations/us-central1/customJobs/{job_id}"
        model_resource = f"projects/stub-project/locations/us-central1/models/{model_name}-{job_id}"
        status = "SUCCEEDED" if self._auto_succeed else "RUNNING"
        self._jobs[resource_name] = {
            "status": status,
            "model_name": model_name,
            "model_resource_name": model_resource,
        }
        return resource_name

    async def get_job_status(self, job_resource_name: str) -> str:
        job = self._jobs.get(job_resource_name)
        return job["status"] if job else "UNKNOWN"

    async def get_model_resource_name(self, job_resource_name: str) -> str:
        job = self._jobs.get(job_resource_name)
        return job["model_resource_name"] if job else ""

    async def cancel_job(self, job_resource_name: str) -> bool:
        if job_resource_name in self._jobs:
            self._jobs[job_resource_name]["status"] = "CANCELLED"
            return True
        return False
