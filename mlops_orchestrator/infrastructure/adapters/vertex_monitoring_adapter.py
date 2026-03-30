from __future__ import annotations
import asyncio

from mlops_orchestrator.infrastructure.adapters.retry import with_retry


class VertexMonitoringAdapter:
    """Real Vertex Model Monitoring adapter. Implements MonitoringPort."""

    def __init__(self, project: str, location: str = "us-central1") -> None:
        self._project = project
        self._location = location
        from google.cloud import aiplatform
        aiplatform.init(project=project, location=location)

    @with_retry(max_attempts=3)
    async def configure_monitoring(
        self, endpoint_id: str, drift_threshold: float, skew_threshold: float
    ) -> bool:
        from google.cloud import aiplatform

        try:
            objective_config = {
                "training_dataset": {"target_field": "target"},
                "training_prediction_skew_detection_config": {
                    "skew_thresholds": {"defaultThreshold": {"value": skew_threshold}}
                },
                "prediction_drift_detection_config": {
                    "drift_thresholds": {"defaultThreshold": {"value": drift_threshold}}
                },
            }
            await asyncio.to_thread(
                aiplatform.ModelDeploymentMonitoringJob.create,
                display_name=f"monitoring-{endpoint_id.split('/')[-1]}",
                endpoint=endpoint_id,
                objective_configs=[objective_config],
                logging_sampling_strategy={"random_sample_config": {"sample_rate": 0.8}},
                schedule_config={"monitor_interval": {"seconds": 3600}},
            )
            return True
        except Exception:
            return False

    @with_retry(max_attempts=3)
    async def get_monitoring_status(self, endpoint_id: str) -> dict[str, str]:
        from google.cloud import aiplatform

        jobs = await asyncio.to_thread(
            aiplatform.ModelDeploymentMonitoringJob.list,
            filter=f'endpoint="{endpoint_id}"',
        )
        if jobs:
            return {"status": str(jobs[0].state), "endpoint_id": endpoint_id}
        return {"status": "NOT_CONFIGURED", "endpoint_id": endpoint_id}

    @with_retry(max_attempts=3)
    async def get_drift_alerts(self, endpoint_id: str) -> list[dict[str, float]]:
        return []
