from __future__ import annotations
from uuid import uuid4

from mlops_orchestrator.domain.value_objects.machine_spec import MachineSpec


class StubVertexDeploymentAdapter:
    """In-memory Vertex deployment adapter. Implements VertexDeploymentPort."""

    def __init__(self) -> None:
        self._endpoints: dict[str, dict[str, str]] = {}

    async def create_endpoint_and_deploy(
        self, model_id: str, endpoint_name: str, machine_spec: MachineSpec
    ) -> str:
        endpoint_id = uuid4().hex[:8]
        resource_name = f"projects/stub-project/locations/us-central1/endpoints/{endpoint_id}"
        self._endpoints[resource_name] = {
            "model_id": model_id,
            "endpoint_name": endpoint_name,
            "machine_type": machine_spec.machine_type,
            "status": "DEPLOYED",
        }
        return resource_name

    async def undeploy(self, endpoint_resource_name: str) -> None:
        if endpoint_resource_name in self._endpoints:
            self._endpoints[endpoint_resource_name]["status"] = "UNDEPLOYED"

    async def get_endpoint_status(self, endpoint_resource_name: str) -> str:
        ep = self._endpoints.get(endpoint_resource_name)
        return ep["status"] if ep else "UNKNOWN"


class StubGkeDeploymentAdapter:
    """In-memory GKE deployment adapter. Implements GkeDeploymentPort."""

    def __init__(self) -> None:
        self._deployments: dict[str, dict[str, str]] = {}

    async def deploy(
        self, model_id: str, cluster_name: str, replica_count: int
    ) -> dict[str, str]:
        deployment_name = f"{model_id.split('/')[-1]}-serving"
        key = f"{cluster_name}/{deployment_name}"
        self._deployments[key] = {
            "deployment_name": deployment_name,
            "cluster_name": cluster_name,
            "replicas": str(replica_count),
            "status": "DEPLOYED",
        }
        return {"deployment_name": deployment_name, "status": "DEPLOYED"}

    async def get_deployment_status(
        self, cluster_name: str, deployment_name: str
    ) -> str:
        key = f"{cluster_name}/{deployment_name}"
        dep = self._deployments.get(key)
        return dep["status"] if dep else "UNKNOWN"

    async def delete_deployment(
        self, cluster_name: str, deployment_name: str
    ) -> None:
        key = f"{cluster_name}/{deployment_name}"
        self._deployments.pop(key, None)
