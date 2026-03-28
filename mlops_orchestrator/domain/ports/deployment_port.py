from __future__ import annotations
from typing import Protocol

from mlops_orchestrator.domain.value_objects.machine_spec import MachineSpec


class VertexDeploymentPort(Protocol):
    """Port for Vertex AI Endpoint deployment."""

    async def create_endpoint_and_deploy(
        self,
        model_id: str,
        endpoint_name: str,
        machine_spec: MachineSpec,
    ) -> str:
        """Create endpoint and deploy model. Returns endpoint resource_name."""
        ...

    async def undeploy(self, endpoint_resource_name: str) -> None:
        """Remove all deployed models from endpoint."""
        ...

    async def get_endpoint_status(self, endpoint_resource_name: str) -> str:
        """Get endpoint deployment status."""
        ...


class GkeDeploymentPort(Protocol):
    """Port for GKE Kubernetes deployment."""

    async def deploy(
        self,
        model_id: str,
        cluster_name: str,
        replica_count: int,
    ) -> dict[str, str]:
        """Create K8s deployment. Returns status dict with deployment info."""
        ...

    async def get_deployment_status(
        self, cluster_name: str, deployment_name: str
    ) -> str:
        """Get deployment status."""
        ...

    async def delete_deployment(
        self, cluster_name: str, deployment_name: str
    ) -> None:
        """Delete a deployment."""
        ...
