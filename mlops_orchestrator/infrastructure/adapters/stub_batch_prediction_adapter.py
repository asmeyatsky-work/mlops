"""Stub batch prediction adapter for testing."""
from __future__ import annotations

from uuid import uuid4


class StubBatchPredictionAdapter:
    """In-memory batch prediction adapter. Implements BatchPredictionPort."""

    def __init__(self, auto_succeed: bool = True) -> None:
        self._auto_succeed = auto_succeed
        self._jobs: dict[str, dict[str, str]] = {}

    async def start_batch_prediction(
        self,
        model_resource_name: str,
        input_uri: str,
        output_uri: str,
        instance_type: str,
    ) -> str:
        job_rn = f"projects/stub-project/locations/us-central1/batchPredictionJobs/{uuid4()}"
        status = "SUCCEEDED" if self._auto_succeed else "RUNNING"
        self._jobs[job_rn] = {
            "status": status,
            "model": model_resource_name,
            "input_uri": input_uri,
            "output_uri": output_uri,
            "instance_type": instance_type,
        }
        return job_rn

    async def get_job_status(self, job_resource_name: str) -> str:
        job = self._jobs.get(job_resource_name)
        return job["status"] if job else "UNKNOWN"

    async def get_job_output_uri(self, job_resource_name: str) -> str:
        job = self._jobs.get(job_resource_name)
        return job["output_uri"] if job else ""

    async def cancel_job(self, job_resource_name: str) -> bool:
        if job_resource_name in self._jobs:
            self._jobs[job_resource_name]["status"] = "CANCELLED"
            return True
        return False
