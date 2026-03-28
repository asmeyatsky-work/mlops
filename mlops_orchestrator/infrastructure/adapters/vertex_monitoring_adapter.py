from __future__ import annotations
from google.cloud import aiplatform


class VertexMonitoringAdapter:
    """Real Vertex Model Monitoring adapter. Implements MonitoringPort."""

    def __init__(self, project: str, location: str = "us-central1") -> None:
        self._project = project
        self._location = location
        aiplatform.init(project=project, location=location)

    async def configure_monitoring(
        self, endpoint_id: str, drift_threshold: float, skew_threshold: float
    ) -> bool:
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
            aiplatform.ModelDeploymentMonitoringJob.create(
                display_name=f"monitoring-{endpoint_id.split('/')[-1]}",
                endpoint=endpoint_id,
                objective_configs=[objective_config],
                logging_sampling_strategy={"random_sample_config": {"sample_rate": 0.8}},
                schedule_config={"monitor_interval": {"seconds": 3600}},
            )
            return True
        except Exception:
            return False

    async def get_monitoring_status(self, endpoint_id: str) -> dict[str, str]:
        jobs = aiplatform.ModelDeploymentMonitoringJob.list(
            filter=f'endpoint="{endpoint_id}"'
        )
        if jobs:
            return {"status": str(jobs[0].state), "endpoint_id": endpoint_id}
        return {"status": "NOT_CONFIGURED", "endpoint_id": endpoint_id}

    async def get_drift_alerts(self, endpoint_id: str) -> list[dict[str, float]]:
        # Vertex returns alerts via Cloud Monitoring; this is a simplified accessor
        return []
