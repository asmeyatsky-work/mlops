from __future__ import annotations
from google.cloud import aiplatform

from mlops_orchestrator.domain.value_objects.machine_spec import MachineSpec


class VertexEndpointAdapter:
    """Real Vertex AI endpoint adapter. Implements VertexDeploymentPort."""

    def __init__(self, project: str, location: str = "us-central1") -> None:
        self._project = project
        self._location = location
        aiplatform.init(project=project, location=location)

    async def create_endpoint_and_deploy(
        self, model_id: str, endpoint_name: str, machine_spec: MachineSpec
    ) -> str:
        endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)
        model = aiplatform.Model(model_id)
        model.deploy(
            endpoint=endpoint,
            machine_type=machine_spec.machine_type,
            min_replica_count=machine_spec.replica_count,
            max_replica_count=machine_spec.replica_count,
        )
        return endpoint.resource_name

    async def undeploy(self, endpoint_resource_name: str) -> None:
        endpoint = aiplatform.Endpoint(endpoint_resource_name)
        endpoint.undeploy_all()

    async def get_endpoint_status(self, endpoint_resource_name: str) -> str:
        try:
            endpoint = aiplatform.Endpoint(endpoint_resource_name)
            deployed = endpoint.list_models()
            return "DEPLOYED" if deployed else "EMPTY"
        except Exception:
            return "UNKNOWN"
