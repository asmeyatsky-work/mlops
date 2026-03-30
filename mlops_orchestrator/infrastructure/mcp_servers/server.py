from __future__ import annotations
import asyncio
import json
import logging

from mcp.server.fastmcp import FastMCP

from mlops_orchestrator.application.dtos.dataset_dto import CreateDatasetRequest
from mlops_orchestrator.application.dtos.deployment_dto import (
    DeployToGkeRequest,
    DeployToVertexRequest,
    MonitoringRequest,
)
from mlops_orchestrator.application.dtos.training_dto import TrainModelRequest
from mlops_orchestrator.application.session.session_state import SessionState
from mlops_orchestrator.infrastructure.auth.auth_middleware import (
    AuthConfig,
    AuthMiddleware,
    NoOpAuthMiddleware,
)

logger = logging.getLogger(__name__)


def create_mlops_server(
    container: object,
    auth_config: AuthConfig | None = None,
) -> FastMCP:
    """
    Create the MLOps Orchestrator MCP server.

    Each bounded context has exactly one MCP server.
    Tools = write operations (commands), Resources = read operations (queries).
    """
    mcp = FastMCP("mlops-orchestrator")

    # Mutable holder for immutable session state, with lock for concurrent access
    state = [SessionState()]
    state_lock = asyncio.Lock()

    # Auth middleware
    if auth_config and auth_config.enabled:
        auth = AuthMiddleware(auth_config)
        logger.info("MCP server authentication enabled")
    else:
        auth = NoOpAuthMiddleware()  # type: ignore[assignment]

    from mlops_orchestrator.infrastructure.config.container import DependencyContainer
    c: DependencyContainer = container  # type: ignore[assignment]

    # ─── TOOLS (write operations / commands) ───

    @mcp.tool()
    async def create_dataset(bq_dataset: str, bq_table: str, name: str) -> dict:
        """Create a Vertex Managed Dataset from a BigQuery table.
        Returns the dataset resource_name for use in subsequent training steps."""
        try:
            cmd = c.create_dataset_command()
            request = CreateDatasetRequest(bq_dataset=bq_dataset, bq_table=bq_table, name=name)
            async with state_lock:
                response, new_state = await cmd.execute(request, state[0])
                state[0] = new_state
            return response.model_dump()
        except Exception as e:
            logger.exception("create_dataset failed")
            return {"error": str(e), "isError": True}

    @mcp.tool()
    async def train_model(model_name: str, dataset_id: str = "", gcs_uri: str = "") -> dict:
        """Submit a CustomTrainingJob to Vertex AI.
        Returns job_resource_name for async monitoring."""
        try:
            cmd = c.train_model_command()
            request = TrainModelRequest(model_name=model_name, dataset_id=dataset_id, gcs_uri=gcs_uri)
            async with state_lock:
                response, new_state = await cmd.execute(request, state[0])
                state[0] = new_state
            return response.model_dump()
        except Exception as e:
            logger.exception("train_model failed")
            return {"error": str(e), "isError": True}

    @mcp.tool()
    async def deploy_to_vertex(model_id: str, endpoint_name: str) -> dict:
        """Deploy a trained model to a Vertex AI Endpoint with n1-standard-4.
        Returns endpoint resource_name."""
        try:
            cmd = c.deploy_vertex_command()
            request = DeployToVertexRequest(model_id=model_id, endpoint_name=endpoint_name)
            async with state_lock:
                response, new_state = await cmd.execute(request, state[0])
                state[0] = new_state
            return response.model_dump()
        except Exception as e:
            logger.exception("deploy_to_vertex failed")
            return {"error": str(e), "isError": True}

    @mcp.tool()
    async def deploy_to_gke(model_id: str, cluster_name: str) -> dict:
        """Deploy a model to a GKE cluster as a V1Deployment with 2 replicas.
        Returns deployment status."""
        try:
            cmd = c.deploy_gke_command()
            request = DeployToGkeRequest(model_id=model_id, cluster_name=cluster_name)
            async with state_lock:
                response, new_state = await cmd.execute(request, state[0])
                state[0] = new_state
            return response.model_dump()
        except Exception as e:
            logger.exception("deploy_to_gke failed")
            return {"error": str(e), "isError": True}

    @mcp.tool()
    async def configure_monitoring(endpoint_id: str) -> dict:
        """Configure Vertex Model Monitoring for drift and prediction skew detection."""
        try:
            cmd = c.configure_monitoring_command()
            request = MonitoringRequest(endpoint_id=endpoint_id)
            async with state_lock:
                response, new_state = await cmd.execute(request, state[0])
                state[0] = new_state
            return response.model_dump()
        except Exception as e:
            logger.exception("configure_monitoring failed")
            return {"error": str(e), "isError": True}

    @mcp.tool()
    async def batch_predict(
        model_resource_name: str,
        input_uri: str,
        output_uri: str,
        instance_type: str = "jsonl",
    ) -> dict:
        """Submit a Vertex AI BatchPredictionJob.
        Returns job resource_name for async monitoring."""
        try:
            cmd = c.batch_prediction_command()
            async with state_lock:
                job_rn = await cmd.execute(
                    model_resource_name=model_resource_name,
                    input_uri=input_uri,
                    output_uri=output_uri,
                    instance_type=instance_type,
                    session=state[0],
                )
                state[0] = state[0].add_job_handle(job_rn)
            return {"job_resource_name": job_rn, "status": "SUBMITTED"}
        except Exception as e:
            logger.exception("batch_predict failed")
            return {"error": str(e), "isError": True}

    @mcp.tool()
    async def register_model(
        display_name: str,
        artifact_uri: str,
        serving_container_image: str = "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest",
        description: str = "",
    ) -> dict:
        """Register a model in the model registry, creating a new version if it already exists."""
        try:
            cmd = c.model_registry_command()
            version = await cmd.execute(
                display_name=display_name,
                artifact_uri=artifact_uri,
                serving_container_image=serving_container_image,
                description=description,
            )
            async with state_lock:
                state[0] = state[0].add_model_uri(version.resource_name)
            return {
                "model_id": version.model_id,
                "version": version.version,
                "resource_name": version.resource_name,
                "stage": version.stage,
            }
        except Exception as e:
            logger.exception("register_model failed")
            return {"error": str(e), "isError": True}

    @mcp.tool()
    async def promote_model(model_id: str, version: int, stage: str) -> dict:
        """Promote a model version to a lifecycle stage (staging, production, archived)."""
        try:
            registry = c.model_registry_port
            promoted = await registry.promote_version(model_id, version, stage)
            return {
                "model_id": promoted.model_id,
                "version": promoted.version,
                "stage": promoted.stage,
            }
        except Exception as e:
            logger.exception("promote_model failed")
            return {"error": str(e), "isError": True}

    # ─── RESOURCES (read operations / queries) ───

    @mcp.resource("mlops://session")
    async def get_session() -> str:
        """Current session state: accumulated dataset IDs, model URIs, job handles, endpoints."""
        return json.dumps(state[0].to_dict(), indent=2)

    @mcp.resource("mlops://jobs/{job_id}")
    async def get_job_status(job_id: str) -> str:
        """Status of a training job by resource name."""
        query = c.job_status_query()
        result = await query.execute(job_id)
        return json.dumps(result)

    @mcp.resource("mlops://costs/{project_id}")
    async def get_costs(project_id: str) -> str:
        """FinOps cost metrics for a project."""
        query = c.cost_query()
        metrics = await query.get_project_metrics(project_id)
        recommendations = await query.get_recommendations(project_id)
        return json.dumps({"metrics": metrics, "recommendations": recommendations}, indent=2)

    @mcp.resource("mlops://models/{model_id}")
    async def get_model_versions(model_id: str) -> str:
        """List all versions of a model in the registry."""
        registry = c.model_registry_port
        versions = await registry.list_versions(model_id)
        return json.dumps(
            [
                {
                    "model_id": v.model_id,
                    "version": v.version,
                    "resource_name": v.resource_name,
                    "display_name": v.display_name,
                    "stage": v.stage,
                }
                for v in versions
            ],
            indent=2,
        )

    # ─── PROMPTS ───

    @mcp.prompt()
    async def audit_compliance(model_name: str) -> str:
        """Generate an EU AI Act compliance audit prompt for a model."""
        return (
            f"Please perform a compliance audit for model '{model_name}'.\n\n"
            "Check the following EU AI Act requirements:\n"
            "1. Risk Classification (Article 6): What risk tier does this model fall into?\n"
            "2. Data Governance (Article 10): Are data sources documented with provenance tracking?\n"
            "3. Technical Documentation (Article 11): Is the model card complete?\n"
            "4. Accuracy & Robustness (Article 15): Are accuracy metrics declared with adversarial testing?\n\n"
            "For each requirement, indicate PASS/FAIL with justification."
        )

    @mcp.prompt()
    async def deployment_summary(endpoint_name: str) -> str:
        """Generate a deployment summary prompt."""
        session_data = state[0].to_dict()
        return (
            f"Summarize the deployment status for endpoint '{endpoint_name}'.\n\n"
            f"Current session state:\n{json.dumps(session_data, indent=2)}\n\n"
            "Include: deployment target, model version, monitoring status, and any drift alerts."
        )

    return mcp
