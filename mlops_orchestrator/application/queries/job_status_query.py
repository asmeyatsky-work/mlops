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
        self,
        job_resource_name: str,
        interval_seconds: int = 60,
        timeout_seconds: int = 86400,
    ) -> dict[str, str]:
        """Poll until terminal state with hard wall-clock timeout via wait_for."""

        async def _poll() -> dict[str, str]:
            while True:
                status = await self._training_port.get_job_status(job_resource_name)
                if status in {"SUCCEEDED", "FAILED", "CANCELLED"}:
                    result: dict[str, str] = {
                        "job_resource_name": job_resource_name,
                        "status": status,
                    }
                    if status == "SUCCEEDED":
                        result["model_resource_name"] = (
                            await self._training_port.get_model_resource_name(
                                job_resource_name
                            )
                        )
                    return result
                await asyncio.sleep(interval_seconds)

        try:
            return await asyncio.wait_for(_poll(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            return {"job_resource_name": job_resource_name, "status": "TIMEOUT"}
