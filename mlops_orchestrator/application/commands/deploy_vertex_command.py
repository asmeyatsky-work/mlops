from __future__ import annotations

from mlops_orchestrator.application.dtos.deployment_dto import (
    DeployToVertexRequest,
    DeploymentResponse,
)
from mlops_orchestrator.application.session.session_state import SessionState
from mlops_orchestrator.domain.entities.deployment import VertexDeployment
from mlops_orchestrator.domain.ports.deployment_port import VertexDeploymentPort
from mlops_orchestrator.domain.ports.infrastructure_ports import (
    AuditLogPort,
    EventBusPort,
)
from mlops_orchestrator.domain.value_objects.machine_spec import MachineSpec


class DeployToVertexCommand:
    """
    Use case: Deploy a trained model to a Vertex AI Endpoint.

    Creates endpoint with n1-standard-4 (PRD default) and deploys the model.
    """

    def __init__(
        self,
        deployment_port: VertexDeploymentPort,
        event_bus: EventBusPort,
        audit_log: AuditLogPort,
    ) -> None:
        self._deployment_port = deployment_port
        self._event_bus = event_bus
        self._audit_log = audit_log

    async def execute(
        self, request: DeployToVertexRequest, session: SessionState
    ) -> tuple[DeploymentResponse, SessionState]:
        machine_spec = MachineSpec(machine_type="n1-standard-4")
        deployment = VertexDeployment.create(
            model_id=request.model_id,
            endpoint_name=request.endpoint_name,
            machine_spec=machine_spec,
        )

        try:
            endpoint_resource_name = await self._deployment_port.create_endpoint_and_deploy(
                model_id=request.model_id,
                endpoint_name=request.endpoint_name,
                machine_spec=machine_spec,
            )
        except Exception as e:
            await self._audit_log.log_action(
                action="deploy_to_vertex",
                resource_id="unknown",
                details={"error": str(e), "model_id": request.model_id},
            )
            raise

        deployment = deployment.deploy(endpoint_resource_name)
        await self._event_bus.publish(list(deployment.domain_events))
        await self._audit_log.log_action(
            action="deploy_to_vertex",
            resource_id=endpoint_resource_name,
            details={"model_id": request.model_id, "machine_type": "n1-standard-4"},
        )

        response = DeploymentResponse(
            resource_name=endpoint_resource_name,
            status=deployment.status,
            target="vertex",
        )
        updated_session = session.add_endpoint(endpoint_resource_name)
        return response, updated_session
