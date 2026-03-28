from __future__ import annotations
from google.cloud import aiplatform


class VertexTrainingAdapter:
    """Real Vertex AI training adapter. Implements TrainingPort."""

    def __init__(self, project: str, location: str = "us-central1") -> None:
        self._project = project
        self._location = location
        aiplatform.init(project=project, location=location)

    async def start_training(
        self, model_name: str, dataset_id: str, gcs_uri: str, train_image: str
    ) -> str:
        job = aiplatform.CustomTrainingJob(
            display_name=model_name,
            script_path="train.py",
            container_uri=train_image,
        )
        kwargs: dict = {"model_display_name": model_name}
        if dataset_id:
            dataset = aiplatform.TabularDataset(dataset_id)
            kwargs["dataset"] = dataset
        model = job.run(**kwargs, sync=False)
        return job.resource_name

    async def get_job_status(self, job_resource_name: str) -> str:
        job = aiplatform.CustomJob.get(job_resource_name)
        state_map = {
            "JOB_STATE_QUEUED": "PENDING",
            "JOB_STATE_PENDING": "PENDING",
            "JOB_STATE_RUNNING": "RUNNING",
            "JOB_STATE_SUCCEEDED": "SUCCEEDED",
            "JOB_STATE_FAILED": "FAILED",
            "JOB_STATE_CANCELLED": "CANCELLED",
        }
        return state_map.get(str(job.state), "UNKNOWN")

    async def get_model_resource_name(self, job_resource_name: str) -> str:
        job = aiplatform.CustomJob.get(job_resource_name)
        if hasattr(job, "output") and job.output:
            return str(job.output.get("model", ""))
        models = aiplatform.Model.list(
            filter=f'display_name="{job.display_name}"',
            order_by="create_time desc",
        )
        return models[0].resource_name if models else ""

    async def cancel_job(self, job_resource_name: str) -> bool:
        try:
            job = aiplatform.CustomJob.get(job_resource_name)
            job.cancel()
            return True
        except Exception:
            return False
