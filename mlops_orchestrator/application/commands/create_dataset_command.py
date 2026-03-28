from __future__ import annotations

from mlops_orchestrator.application.dtos.dataset_dto import (
    CreateDatasetRequest,
    DatasetResponse,
)
from mlops_orchestrator.application.session.session_state import SessionState
from mlops_orchestrator.domain.ports.dataset_port import DatasetPort
from mlops_orchestrator.domain.ports.infrastructure_ports import (
    AuditLogPort,
    EventBusPort,
)
from mlops_orchestrator.domain.services.dataset_service import DatasetDomainService
from mlops_orchestrator.domain.value_objects.bq_source import BigQuerySource


class CreateDatasetCommand:
    """
    Use case: Create a Vertex Managed Dataset from BigQuery.

    Orchestrates domain service, infrastructure port, event publishing, and audit logging.
    Returns the dataset resource_name and updated session state for stitching.
    """

    def __init__(
        self,
        dataset_port: DatasetPort,
        event_bus: EventBusPort,
        audit_log: AuditLogPort,
    ) -> None:
        self._dataset_port = dataset_port
        self._event_bus = event_bus
        self._audit_log = audit_log
        self._domain_service = DatasetDomainService()

    async def execute(
        self, request: CreateDatasetRequest, session: SessionState
    ) -> tuple[DatasetResponse, SessionState]:
        bq_source = BigQuerySource(dataset=request.bq_dataset, table=request.bq_table)
        errors = self._domain_service.validate_bq_source(bq_source)
        if errors:
            raise ValueError(f"Invalid BigQuery source: {'; '.join(errors)}")

        dataset = self._domain_service.create_managed_dataset(
            bq_dataset=request.bq_dataset,
            bq_table=request.bq_table,
            display_name=request.name,
        )

        try:
            resource_name = await self._dataset_port.create_dataset(
                bq_source=bq_source, display_name=request.name
            )
        except Exception as e:
            await self._audit_log.log_action(
                action="create_dataset",
                resource_id="unknown",
                details={"error": str(e), "bq_dataset": request.bq_dataset},
            )
            raise

        dataset = dataset.register(resource_name)

        await self._event_bus.publish(list(dataset.domain_events))
        await self._audit_log.log_action(
            action="create_dataset",
            resource_id=resource_name,
            details={"bq_dataset": request.bq_dataset, "bq_table": request.bq_table},
        )

        response = DatasetResponse(
            resource_name=resource_name,
            display_name=dataset.display_name,
            status=dataset.status,
        )
        updated_session = session.add_dataset(resource_name)
        return response, updated_session
