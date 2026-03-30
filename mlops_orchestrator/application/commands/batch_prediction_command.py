"""Command to submit a Vertex AI batch prediction job."""
from __future__ import annotations

from mlops_orchestrator.application.session.session_state import SessionState
from mlops_orchestrator.domain.entities.batch_prediction_job import BatchPredictionJob
from mlops_orchestrator.domain.ports.batch_prediction_port import BatchPredictionPort
from mlops_orchestrator.domain.ports.infrastructure_ports import (
    AuditLogPort,
    EventBusPort,
)


class BatchPredictionCommand:
    """Submit a batch prediction job to Vertex AI."""

    def __init__(
        self,
        batch_port: BatchPredictionPort,
        event_bus: EventBusPort,
        audit_log: AuditLogPort,
    ) -> None:
        self._batch_port = batch_port
        self._event_bus = event_bus
        self._audit_log = audit_log

    async def execute(
        self,
        model_resource_name: str,
        input_uri: str,
        output_uri: str,
        instance_type: str,
        session: SessionState,
    ) -> str:
        entity = BatchPredictionJob.create(
            model_resource_name=model_resource_name,
            input_uri=input_uri,
            output_uri=output_uri,
            instance_type=instance_type,
        )

        job_rn = await self._batch_port.start_batch_prediction(
            model_resource_name=model_resource_name,
            input_uri=input_uri,
            output_uri=output_uri,
            instance_type=instance_type,
        )
        entity = entity.start(job_rn)

        await self._audit_log.log_action(
            action="batch_prediction_submitted",
            resource_id=job_rn,
            details={
                "model": model_resource_name,
                "input_uri": input_uri,
                "output_uri": output_uri,
            },
        )

        return job_rn
