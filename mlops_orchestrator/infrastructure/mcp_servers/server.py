from __future__ import annotations
import asyncio
import json
import logging
import time
from typing import Any, Awaitable, Callable

from mcp.server.fastmcp import FastMCP

from mlops_orchestrator.application.dtos.dataset_dto import CreateDatasetRequest
from mlops_orchestrator.application.dtos.deployment_dto import (
    DeployToGkeRequest,
    DeployToVertexRequest,
    MonitoringRequest,
)
from mlops_orchestrator.application.dtos.training_dto import TrainModelRequest
from mlops_orchestrator.application.session.session_state import SessionState
from mlops_orchestrator.infrastructure.auth.auth_context import (
    ToolAuthzError,
    enforce_tool_authz,
)
from mlops_orchestrator.infrastructure.auth.auth_middleware import (
    AuthConfig,
    AuthMiddleware,
    NoOpAuthMiddleware,
)
from mlops_orchestrator.infrastructure.observability.tracing import (
    correlation_id_var,
    new_correlation_id,
    start_span,
)

logger = logging.getLogger(__name__)


def _tool_call(
    name: str,
    coro_factory: Callable[[], Awaitable[Any]],
) -> Callable[[], Awaitable[dict]]:
    """Wraps a tool body with: authz check, correlation id, span, structured
    error envelope, and duration logging. Coroutine is built fresh inside so
    awaiting twice does not reuse a consumed coroutine."""

    async def runner() -> dict:
        cid = new_correlation_id()
        token = correlation_id_var.set(cid)
        started = time.perf_counter()
        try:
            try:
                enforce_tool_authz(name)
            except ToolAuthzError as e:
                logger.warning("tool authz denied: %s", e, extra={"action": name})
                return {"error": str(e), "isError": True, "correlation_id": cid}

            try:
                async with start_span(f"mcp.tool.{name}"):
                    result = await coro_factory()
            except Exception as e:
                logger.exception("%s failed", name, extra={"action": name})
                return {"error": str(e), "isError": True, "correlation_id": cid}

            duration_ms = int((time.perf_counter() - started) * 1000)
            logger.info(
                "%s ok",
                name,
                extra={"action": name, "duration_ms": duration_ms},
            )
            if isinstance(result, dict):
                result.setdefault("correlation_id", cid)
            return result  # type: ignore[return-value]
        finally:
            correlation_id_var.reset(token)

    return runner


def create_mlops_server(
    container: object,
    auth_config: AuthConfig | None = None,
) -> FastMCP:
    """Create the MLOps Orchestrator MCP server.

    Each bounded context has exactly one MCP server.
    Tools = write operations (commands), Resources = read operations (queries).

    Concurrency invariant: ``state_lock`` is only held to snapshot the current
    SessionState and to swap in the new one — never across an awaited I/O
    call. This keeps concurrent tool invocations from serializing on long
    GCP operations.
    """
    mcp = FastMCP("mlops-orchestrator")

    state: list[SessionState] = [SessionState()]
    state_lock = asyncio.Lock()

    if auth_config and auth_config.enabled:
        auth = AuthMiddleware(auth_config)
        logger.info("MCP server authentication enabled")
    else:
        auth = NoOpAuthMiddleware()  # type: ignore[assignment]

    from mlops_orchestrator.infrastructure.config.container import DependencyContainer

    c: DependencyContainer = container  # type: ignore[assignment]

    async def _snapshot() -> SessionState:
        async with state_lock:
            return state[0]

    async def _apply(new_state: SessionState) -> None:
        async with state_lock:
            state[0] = new_state

    # ─── TOOLS ───

    @mcp.tool()
    async def create_dataset(bq_dataset: str, bq_table: str, name: str) -> dict:
        """Create a Vertex Managed Dataset from a BigQuery table."""

        async def body() -> dict:
            cmd = c.create_dataset_command()
            request = CreateDatasetRequest(
                bq_dataset=bq_dataset, bq_table=bq_table, name=name
            )
            snapshot = await _snapshot()
            response, new_state = await cmd.execute(request, snapshot)
            await _apply(new_state)
            return response.model_dump()

        return await _tool_call("create_dataset", body)()

    @mcp.tool()
    async def train_model(model_name: str, dataset_id: str = "", gcs_uri: str = "") -> dict:
        """Submit a CustomTrainingJob to Vertex AI."""

        async def body() -> dict:
            cmd = c.train_model_command()
            request = TrainModelRequest(
                model_name=model_name, dataset_id=dataset_id, gcs_uri=gcs_uri
            )
            snapshot = await _snapshot()
            response, new_state = await cmd.execute(request, snapshot)
            await _apply(new_state)
            return response.model_dump()

        return await _tool_call("train_model", body)()

    @mcp.tool()
    async def deploy_to_vertex(model_id: str, endpoint_name: str) -> dict:
        """Deploy a trained model to a Vertex AI Endpoint."""

        async def body() -> dict:
            cmd = c.deploy_vertex_command()
            request = DeployToVertexRequest(
                model_id=model_id, endpoint_name=endpoint_name
            )
            snapshot = await _snapshot()
            response, new_state = await cmd.execute(request, snapshot)
            await _apply(new_state)
            return response.model_dump()

        return await _tool_call("deploy_to_vertex", body)()

    @mcp.tool()
    async def deploy_to_gke(model_id: str, cluster_name: str) -> dict:
        """Deploy a model to a GKE cluster."""

        async def body() -> dict:
            cmd = c.deploy_gke_command()
            request = DeployToGkeRequest(model_id=model_id, cluster_name=cluster_name)
            snapshot = await _snapshot()
            response, new_state = await cmd.execute(request, snapshot)
            await _apply(new_state)
            return response.model_dump()

        return await _tool_call("deploy_to_gke", body)()

    @mcp.tool()
    async def configure_monitoring(endpoint_id: str) -> dict:
        """Configure Vertex Model Monitoring for drift and prediction skew detection."""

        async def body() -> dict:
            cmd = c.configure_monitoring_command()
            request = MonitoringRequest(endpoint_id=endpoint_id)
            snapshot = await _snapshot()
            response, new_state = await cmd.execute(request, snapshot)
            await _apply(new_state)
            return response.model_dump()

        return await _tool_call("configure_monitoring", body)()

    @mcp.tool()
    async def batch_predict(
        model_resource_name: str,
        input_uri: str,
        output_uri: str,
        instance_type: str = "jsonl",
    ) -> dict:
        """Submit a Vertex AI BatchPredictionJob."""

        async def body() -> dict:
            cmd = c.batch_prediction_command()
            snapshot = await _snapshot()
            job_rn = await cmd.execute(
                model_resource_name=model_resource_name,
                input_uri=input_uri,
                output_uri=output_uri,
                instance_type=instance_type,
                session=snapshot,
            )
            await _apply(snapshot.add_job_handle(job_rn))
            return {"job_resource_name": job_rn, "status": "SUBMITTED"}

        return await _tool_call("batch_predict", body)()

    @mcp.tool()
    async def register_model(
        display_name: str,
        artifact_uri: str,
        serving_container_image: str = "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest",
        description: str = "",
    ) -> dict:
        """Register a model in the model registry."""

        async def body() -> dict:
            cmd = c.model_registry_command()
            version = await cmd.execute(
                display_name=display_name,
                artifact_uri=artifact_uri,
                serving_container_image=serving_container_image,
                description=description,
            )
            snapshot = await _snapshot()
            await _apply(snapshot.add_model_uri(version.resource_name))
            return {
                "model_id": version.model_id,
                "version": version.version,
                "resource_name": version.resource_name,
                "stage": version.stage,
            }

        return await _tool_call("register_model", body)()

    @mcp.tool()
    async def promote_model(model_id: str, version: int, stage: str) -> dict:
        """Promote a model version to a lifecycle stage."""

        async def body() -> dict:
            registry = c.model_registry_port
            promoted = await registry.promote_version(model_id, version, stage)
            return {
                "model_id": promoted.model_id,
                "version": promoted.version,
                "stage": promoted.stage,
            }

        return await _tool_call("promote_model", body)()

    # ─── RESOURCES ───

    @mcp.resource("mlops://session")
    async def get_session() -> str:
        snapshot = await _snapshot()
        return json.dumps(snapshot.to_dict(), indent=2)

    @mcp.resource("mlops://jobs/{job_id}")
    async def get_job_status(job_id: str) -> str:
        query = c.job_status_query()
        result = await query.execute(job_id)
        return json.dumps(result)

    @mcp.resource("mlops://costs/{project_id}")
    async def get_costs(project_id: str) -> str:
        query = c.cost_query()
        metrics = await query.get_project_metrics(project_id)
        recommendations = await query.get_recommendations(project_id)
        return json.dumps({"metrics": metrics, "recommendations": recommendations}, indent=2)

    @mcp.resource("mlops://models/{model_id}")
    async def get_model_versions(model_id: str) -> str:
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
        snapshot = await _snapshot()
        return (
            f"Summarize the deployment status for endpoint '{endpoint_name}'.\n\n"
            f"Current session state:\n{json.dumps(snapshot.to_dict(), indent=2)}\n\n"
            "Include: deployment target, model version, monitoring status, and any drift alerts."
        )

    return mcp
