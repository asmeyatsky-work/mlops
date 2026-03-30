"""Real Vertex AI batch prediction adapter."""
from __future__ import annotations

import asyncio

from mlops_orchestrator.infrastructure.adapters.retry import with_retry


class VertexBatchPredictionAdapter:
    """Real Vertex AI batch prediction adapter. Implements BatchPredictionPort."""

    def __init__(self, project: str, location: str = "us-central1") -> None:
        self._project = project
        self._location = location
        from google.cloud import aiplatform
        aiplatform.init(project=project, location=location)

    @with_retry(max_attempts=3)
    async def start_batch_prediction(
        self,
        model_resource_name: str,
        input_uri: str,
        output_uri: str,
        instance_type: str,
    ) -> str:
        from google.cloud import aiplatform

        model = aiplatform.Model(model_resource_name)
        job = await asyncio.to_thread(
            model.batch_predict,
            job_display_name=f"batch-{model_resource_name.split('/')[-1]}",
            instances_format=instance_type,
            gcs_source=input_uri,
            gcs_destination_output_uri_prefix=output_uri,
            sync=False,
        )
        return job.resource_name

    @with_retry(max_attempts=3)
    async def get_job_status(self, job_resource_name: str) -> str:
        from google.cloud import aiplatform

        job = await asyncio.to_thread(
            aiplatform.BatchPredictionJob.get, job_resource_name
        )
        state_map = {
            "JOB_STATE_QUEUED": "PENDING",
            "JOB_STATE_PENDING": "PENDING",
            "JOB_STATE_RUNNING": "RUNNING",
            "JOB_STATE_SUCCEEDED": "SUCCEEDED",
            "JOB_STATE_FAILED": "FAILED",
            "JOB_STATE_CANCELLED": "CANCELLED",
        }
        return state_map.get(str(job.state), "UNKNOWN")

    @with_retry(max_attempts=3)
    async def get_job_output_uri(self, job_resource_name: str) -> str:
        from google.cloud import aiplatform

        job = await asyncio.to_thread(
            aiplatform.BatchPredictionJob.get, job_resource_name
        )
        return job.output_info.gcs_output_directory if job.output_info else ""

    @with_retry(max_attempts=3)
    async def cancel_job(self, job_resource_name: str) -> bool:
        from google.cloud import aiplatform

        try:
            job = await asyncio.to_thread(
                aiplatform.BatchPredictionJob.get, job_resource_name
            )
            await asyncio.to_thread(job.cancel)
            return True
        except Exception:
            return False
