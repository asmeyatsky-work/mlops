from __future__ import annotations

from mlops_orchestrator.application.dtos.deployment_dto import (
    MonitoringRequest,
    MonitoringResponse,
)
from mlops_orchestrator.application.session.session_state import SessionState
from mlops_orchestrator.domain.entities.monitoring_config import MonitoringConfig
from mlops_orchestrator.domain.ports.infrastructure_ports import (
    AuditLogPort,
    EventBusPort,
)
from mlops_orchestrator.domain.ports.monitoring_port import MonitoringPort


class ConfigureMonitoringCommand:
    """
    Use case: Configure Vertex Model Monitoring on a deployed endpoint.

    Proactively sets up drift detection and prediction skew tracking.
    """

    def __init__(
        self,
        monitoring_port: MonitoringPort,
        event_bus: EventBusPort,
        audit_log: AuditLogPort,
    ) -> None:
        self._monitoring_port = monitoring_port
        self._event_bus = event_bus
        self._audit_log = audit_log

    async def execute(
        self, request: MonitoringRequest, session: SessionState
    ) -> tuple[MonitoringResponse, SessionState]:
        config = MonitoringConfig.create(
            endpoint_id=request.endpoint_id,
            drift_threshold=request.drift_threshold,
            skew_threshold=request.skew_threshold,
        )

        try:
            success = await self._monitoring_port.configure_monitoring(
                endpoint_id=request.endpoint_id,
                drift_threshold=request.drift_threshold,
                skew_threshold=request.skew_threshold,
            )
        except Exception as e:
            await self._audit_log.log_action(
                action="configure_monitoring",
                resource_id=request.endpoint_id,
                details={"error": str(e), "status": "error"},
            )
            raise

        if success:
            config = config.enable()

        await self._event_bus.publish(list(config.domain_events))
        await self._audit_log.log_action(
            action="configure_monitoring",
            resource_id=request.endpoint_id,
            details={
                "drift_threshold": str(request.drift_threshold),
                "skew_threshold": str(request.skew_threshold),
                "status": "success" if success else "failed",
            },
        )

        response = MonitoringResponse(
            endpoint_id=request.endpoint_id,
            monitoring_enabled=config.enabled,
            status="ACTIVE" if config.enabled else "FAILED",
        )
        updated_session = session.set_metadata(
            f"monitoring_{request.endpoint_id}", "enabled" if config.enabled else "failed"
        )
        return response, updated_session
