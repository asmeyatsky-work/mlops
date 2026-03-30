"""Tests for MCP server tool and resource registration."""
from __future__ import annotations

import pytest

from mlops_orchestrator.infrastructure.config.settings import Settings
from mlops_orchestrator.infrastructure.config.container import DependencyContainer
from mlops_orchestrator.infrastructure.mcp_servers.server import create_mlops_server
from mlops_orchestrator.infrastructure.auth.auth_middleware import AuthConfig


@pytest.fixture
def container():
    return DependencyContainer(Settings(use_stubs=True))


class TestMcpServerCreation:
    def test_server_created(self, container):
        mcp = create_mlops_server(container)
        assert mcp is not None
        assert mcp.name == "mlops-orchestrator"

    def test_server_with_auth_config(self, container):
        auth = AuthConfig(enabled=True, api_keys=("test-key",))
        mcp = create_mlops_server(container, auth_config=auth)
        assert mcp is not None

    def test_server_without_auth_config(self, container):
        mcp = create_mlops_server(container, auth_config=None)
        assert mcp is not None


class TestMcpTools:
    async def test_tools_registered(self, container):
        mcp = create_mlops_server(container)
        tools = {t.name: t for t in mcp._tool_manager.list_tools()}
        assert "create_dataset" in tools
        assert "train_model" in tools
        assert "deploy_to_vertex" in tools
        assert "deploy_to_gke" in tools
        assert "configure_monitoring" in tools
        assert "batch_predict" in tools
        assert "register_model" in tools
        assert "promote_model" in tools

    async def test_create_dataset_executes(self, container):
        mcp = create_mlops_server(container)
        result = await mcp._tool_manager.call_tool(
            "create_dataset",
            {"bq_dataset": "ds", "bq_table": "tbl", "name": "test"},
        )
        assert result["status"] == "REGISTERED"
        assert result["display_name"] == "test"

    async def test_train_model_executes(self, container):
        mcp = create_mlops_server(container)
        result = await mcp._tool_manager.call_tool(
            "train_model",
            {"model_name": "my-model", "dataset_id": "ds-1"},
        )
        assert result["status"] == "RUNNING"

    async def test_deploy_to_vertex_executes(self, container):
        mcp = create_mlops_server(container)
        result = await mcp._tool_manager.call_tool(
            "deploy_to_vertex",
            {"model_id": "m-1", "endpoint_name": "ep"},
        )
        assert result["status"] == "DEPLOYED"
        assert result["target"] == "vertex"

    async def test_deploy_to_gke_executes(self, container):
        mcp = create_mlops_server(container)
        result = await mcp._tool_manager.call_tool(
            "deploy_to_gke",
            {"model_id": "m-1", "cluster_name": "c-1"},
        )
        assert result["status"] == "DEPLOYED"
        assert result["target"] == "gke"

    async def test_configure_monitoring_executes(self, container):
        mcp = create_mlops_server(container)
        result = await mcp._tool_manager.call_tool(
            "configure_monitoring",
            {"endpoint_id": "ep-1"},
        )
        assert result["monitoring_enabled"] is True

    async def test_batch_predict_executes(self, container):
        mcp = create_mlops_server(container)
        result = await mcp._tool_manager.call_tool(
            "batch_predict",
            {
                "model_resource_name": "projects/p/models/m",
                "input_uri": "gs://bucket/input",
                "output_uri": "gs://bucket/output",
            },
        )
        assert result["status"] == "SUBMITTED"
        assert "job_resource_name" in result

    async def test_register_model_executes(self, container):
        mcp = create_mlops_server(container)
        result = await mcp._tool_manager.call_tool(
            "register_model",
            {
                "display_name": "test-model",
                "artifact_uri": "gs://bucket/model",
            },
        )
        assert result["version"] == 1
        assert result["stage"] == "development"

    async def test_promote_model_executes(self, container):
        mcp = create_mlops_server(container)
        # First register a model
        await mcp._tool_manager.call_tool(
            "register_model",
            {"display_name": "test-model", "artifact_uri": "gs://bucket/model"},
        )
        # Then promote it
        result = await mcp._tool_manager.call_tool(
            "promote_model",
            {
                "model_id": "projects/stub-project/locations/us-central1/models/test-model",
                "version": 1,
                "stage": "production",
            },
        )
        assert result["stage"] == "production"

    async def test_tool_error_returns_error_dict(self, container):
        """Passing invalid args should return error dict, not crash."""
        mcp = create_mlops_server(container)
        result = await mcp._tool_manager.call_tool(
            "create_dataset",
            {"bq_dataset": "", "bq_table": "tbl", "name": "test"},
        )
        assert result.get("isError") is True or "error" in result

    async def test_session_state_stitching(self, container):
        """Tools should accumulate session state across calls."""
        mcp = create_mlops_server(container)
        await mcp._tool_manager.call_tool(
            "create_dataset",
            {"bq_dataset": "ds", "bq_table": "tbl", "name": "test"},
        )
        await mcp._tool_manager.call_tool(
            "train_model",
            {"model_name": "m", "dataset_id": "ds-1"},
        )
        resources = mcp._resource_manager.list_resources()
        session_resource = next((r for r in resources if "session" in str(r.uri)), None)
        assert session_resource is not None


class TestMcpResources:
    async def test_resources_registered(self, container):
        mcp = create_mlops_server(container)
        resources = mcp._resource_manager.list_resources()
        resource_uris = [str(r.uri) for r in resources]
        assert any("session" in u for u in resource_uris)
