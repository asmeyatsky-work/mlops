from __future__ import annotations
import asyncio
from mlops_orchestrator.domain.ports.training_port import TrainingPort


class JobStatusQuery:
    """Query the status of an async training job."""

    def __init__(self, training_port: TrainingPort) -> None:
        self._training_port = training_port

    async def execute(self, job_resource_name: str) -> dict[str, str]:
        status = await self._training_port.get_job_status(job_resource_name)
        return {"job_resource_name": job_resource_name, "status": status}

    async def poll_until_complete(
        self, job_resource_name: str, interval_seconds: int = 60, timeout_seconds: int = 86400
    ) -> dict[str, str]:
        """Poll a training job until it reaches a terminal state."""
        elapsed = 0
        while elapsed < timeout_seconds:
            status = await self._training_port.get_job_status(job_resource_name)
            if status in {"SUCCEEDED", "FAILED", "CANCELLED"}:
                result = {"job_resource_name": job_resource_name, "status": status}
                if status == "SUCCEEDED":
                    model_name = await self._training_port.get_model_resource_name(
                        job_resource_name
                    )
                    result["model_resource_name"] = model_name
                return result
            await asyncio.sleep(interval_seconds)
            elapsed += interval_seconds
        return {"job_resource_name": job_resource_name, "status": "TIMEOUT"}
