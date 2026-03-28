from __future__ import annotations

from mlops_orchestrator.application.dtos.deployment_dto import (
    DeployToGkeRequest,
    DeploymentResponse,
)
from mlops_orchestrator.application.session.session_state import SessionState
from mlops_orchestrator.domain.entities.deployment import GkeDeployment
from mlops_orchestrator.domain.ports.deployment_port import GkeDeploymentPort
from mlops_orchestrator.domain.ports.infrastructure_ports import (
    AuditLogPort,
    EventBusPort,
)


class DeployToGkeCommand:
    """
    Use case: Deploy a model to a GKE cluster.

    Creates a V1Deployment with 2 replicas (PRD default).
    """

    def __init__(
        self,
        deployment_port: GkeDeploymentPort,
        event_bus: EventBusPort,
        audit_log: AuditLogPort,
    ) -> None:
        self._deployment_port = deployment_port
        self._event_bus = event_bus
        self._audit_log = audit_log

    async def execute(
        self, request: DeployToGkeRequest, session: SessionState
    ) -> tuple[DeploymentResponse, SessionState]:
        gke_deployment = GkeDeployment.create(
            model_id=request.model_id,
            cluster_name=request.cluster_name,
            replica_count=2,
        )

        try:
            result = await self._deployment_port.deploy(
                model_id=request.model_id,
                cluster_name=request.cluster_name,
                replica_count=2,
            )
        except Exception as e:
            await self._audit_log.log_action(
                action="deploy_to_gke",
                resource_id=request.cluster_name,
                details={"error": str(e), "model_id": request.model_id},
            )
            raise

        deployment_name = result.get("deployment_name")
        if not deployment_name:
            raise ValueError("GKE deployment port did not return deployment_name")

        gke_deployment = gke_deployment.mark_deployed()
        await self._event_bus.publish(list(gke_deployment.domain_events))
        await self._audit_log.log_action(
            action="deploy_to_gke",
            resource_id=request.cluster_name,
            details={"model_id": request.model_id, "replicas": "2"},
        )

        response = DeploymentResponse(
            resource_name=deployment_name,
            status=gke_deployment.status,
            target="gke",
        )
        updated_session = session.add_endpoint(
            f"gke://{request.cluster_name}/{deployment_name}"
        )
        return response, updated_session
