from __future__ import annotations
import asyncio

from mlops_orchestrator.domain.value_objects.machine_spec import MachineSpec


class VertexEndpointAdapter:
    """Real Vertex AI endpoint adapter. Implements VertexDeploymentPort."""

    def __init__(self, project: str, location: str = "us-central1") -> None:
        self._project = project
        self._location = location
        from google.cloud import aiplatform
        aiplatform.init(project=project, location=location)

    async def create_endpoint_and_deploy(
        self, model_id: str, endpoint_name: str, machine_spec: MachineSpec
    ) -> str:
        from google.cloud import aiplatform

        endpoint = await asyncio.to_thread(
            aiplatform.Endpoint.create, display_name=endpoint_name
        )
        model = aiplatform.Model(model_id)
        await asyncio.to_thread(
            model.deploy,
            endpoint=endpoint,
            machine_type=machine_spec.machine_type,
            min_replica_count=machine_spec.replica_count,
            max_replica_count=machine_spec.replica_count,
        )
        return endpoint.resource_name

    async def undeploy(self, endpoint_resource_name: str) -> None:
        from google.cloud import aiplatform

        endpoint = aiplatform.Endpoint(endpoint_resource_name)
        await asyncio.to_thread(endpoint.undeploy_all)

    async def get_endpoint_status(self, endpoint_resource_name: str) -> str:
        from google.cloud import aiplatform

        try:
            endpoint = aiplatform.Endpoint(endpoint_resource_name)
            deployed = await asyncio.to_thread(endpoint.list_models)
            return "DEPLOYED" if deployed else "EMPTY"
        except Exception:
            return "UNKNOWN"
