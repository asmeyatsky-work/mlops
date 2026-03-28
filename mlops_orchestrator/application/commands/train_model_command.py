from __future__ import annotations

from mlops_orchestrator.application.dtos.training_dto import (
    TrainModelRequest,
    TrainingResponse,
)
from mlops_orchestrator.application.session.session_state import SessionState
from mlops_orchestrator.domain.entities.training_job import TRAIN_IMAGE, TrainingJob
from mlops_orchestrator.domain.ports.infrastructure_ports import (
    AuditLogPort,
    EventBusPort,
)
from mlops_orchestrator.domain.ports.training_port import TrainingPort


class TrainModelCommand:
    """
    Use case: Submit a CustomTrainingJob to Vertex AI.

    Returns immediately with a job handle for async polling.
    The agent uses the job handle to monitor progress across sessions.
    """

    def __init__(
        self,
        training_port: TrainingPort,
        event_bus: EventBusPort,
        audit_log: AuditLogPort,
    ) -> None:
        self._training_port = training_port
        self._event_bus = event_bus
        self._audit_log = audit_log

    async def execute(
        self, request: TrainModelRequest, session: SessionState
    ) -> tuple[TrainingResponse, SessionState]:
        job = TrainingJob.create(
            model_name=request.model_name,
            dataset_id=request.dataset_id,
            gcs_uri=request.gcs_uri,
            train_image=TRAIN_IMAGE,
        )

        job_resource_name = await self._training_port.start_training(
            model_name=request.model_name,
            dataset_id=request.dataset_id,
            gcs_uri=request.gcs_uri,
            train_image=job.train_image,
        )

        job = job.start(job_resource_name)
        await self._event_bus.publish(list(job.domain_events))
        await self._audit_log.log_action(
            action="train_model",
            resource_id=job_resource_name,
            details={"model_name": request.model_name, "dataset_id": request.dataset_id},
        )

        response = TrainingResponse(
            job_resource_name=job_resource_name,
            model_name=request.model_name,
            status=job.status,
        )
        updated_session = session.add_job_handle(job_resource_name)
        return response, updated_session
