from __future__ import annotations

from mlops_orchestrator.application.dtos.deployment_dto import (
    DeployToVertexRequest,
    DeploymentResponse,
)
from mlops_orchestrator.application.session.session_state import SessionState
from mlops_orchestrator.domain.entities.deployment import VertexDeployment
from mlops_orchestrator.domain.ports.deployment_port import VertexDeploymentPort
from mlops_orchestrator.domain.ports.governance_port import ModelGovernancePort
from mlops_orchestrator.domain.ports.infrastructure_ports import (
    AuditLogPort,
    EventBusPort,
)
from mlops_orchestrator.domain.services.compliance_service import (
    ComplianceGateError,
    ComplianceGateService,
)
from mlops_orchestrator.domain.value_objects.machine_spec import MachineSpec


class DeployToVertexCommand:
    """
    Use case: Deploy a trained model to a Vertex AI Endpoint.

    Creates endpoint with n1-standard-4 (PRD default) and deploys the model.
    When ``governance_port`` is supplied the EU AI Act compliance gate runs
    first: deployment is rejected for PROHIBITED models and for HIGH-risk
    models that lack a complete model card or declared required controls.
    """

    def __init__(
        self,
        deployment_port: VertexDeploymentPort,
        event_bus: EventBusPort,
        audit_log: AuditLogPort,
        governance_port: ModelGovernancePort | None = None,
    ) -> None:
        self._deployment_port = deployment_port
        self._event_bus = event_bus
        self._audit_log = audit_log
        self._governance_port = governance_port
        self._gate = ComplianceGateService()

    async def _enforce_compliance(self, model_id: str) -> None:
        if self._governance_port is None:
            return
        risk = await self._governance_port.get_risk_classification(model_id)
        card = await self._governance_port.get_model_card(model_id)
        try:
            self._gate.enforce(risk, card)
        except ComplianceGateError as exc:
            await self._audit_log.log_action(
                action="deploy_to_vertex",
                resource_id=model_id,
                details={"compliance_block": str(exc), "model_id": model_id},
            )
            raise

    async def execute(
        self, request: DeployToVertexRequest, session: SessionState
    ) -> tuple[DeploymentResponse, SessionState]:
        await self._enforce_compliance(request.model_id)

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
