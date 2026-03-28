from __future__ import annotations
from kubernetes import client, config


class GkeDeploymentAdapter:
    """Real GKE Kubernetes deployment adapter. Implements GkeDeploymentPort."""

    def __init__(self) -> None:
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()
        self._apps_api = client.AppsV1Api()

    async def deploy(
        self, model_id: str, cluster_name: str, replica_count: int
    ) -> dict[str, str]:
        model_short = model_id.split("/")[-1] if "/" in model_id else model_id
        deployment_name = f"{model_short}-serving"
        container = client.V1Container(
            name="model-server",
            image="tensorflow/serving:latest",
            ports=[client.V1ContainerPort(container_port=8501)],
            env=[
                client.V1EnvVar(name="MODEL_NAME", value=model_short),
            ],
        )
        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels={"app": deployment_name}),
            spec=client.V1PodSpec(containers=[container]),
        )
        spec = client.V1DeploymentSpec(
            replicas=replica_count,
            selector=client.V1LabelSelector(
                match_labels={"app": deployment_name}
            ),
            template=template,
        )
        deployment = client.V1Deployment(
            api_version="apps/v1",
            kind="Deployment",
            metadata=client.V1ObjectMeta(name=deployment_name),
            spec=spec,
        )
        self._apps_api.create_namespaced_deployment(
            namespace="default", body=deployment
        )
        return {"deployment_name": deployment_name, "status": "DEPLOYED"}

    async def get_deployment_status(
        self, cluster_name: str, deployment_name: str
    ) -> str:
        try:
            dep = self._apps_api.read_namespaced_deployment(
                name=deployment_name, namespace="default"
            )
            if dep.status.available_replicas and dep.status.available_replicas > 0:
                return "DEPLOYED"
            return "PENDING"
        except client.ApiException:
            return "NOT_FOUND"

    async def delete_deployment(
        self, cluster_name: str, deployment_name: str
    ) -> None:
        self._apps_api.delete_namespaced_deployment(
            name=deployment_name, namespace="default"
        )
