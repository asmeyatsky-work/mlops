"""MCP contract tests — guarantee tool surface stays stable for the agent.

The agent and any MCP client builds requests from these tool names and
schemas, so a rename or accidental removal is a breaking change.
"""
from __future__ import annotations

import pytest

from mlops_orchestrator.infrastructure.config.container import DependencyContainer
from mlops_orchestrator.infrastructure.config.settings import Settings
from mlops_orchestrator.infrastructure.mcp_servers.server import create_mlops_server


EXPECTED_TOOLS = {
    "create_dataset",
    "train_model",
    "deploy_to_vertex",
    "deploy_to_gke",
    "configure_monitoring",
    "batch_predict",
    "register_model",
    "promote_model",
}

EXPECTED_RESOURCE_PREFIXES = {
    "mlops://session",
    "mlops://jobs/",
    "mlops://costs/",
    "mlops://models/",
}


@pytest.fixture
def server():
    settings = Settings(use_stubs=True)
    return create_mlops_server(DependencyContainer(settings))


class TestMcpContract:
    async def test_all_expected_tools_registered(self, server):
        tools = await server.list_tools()
        names = {t.name for t in tools}
        assert EXPECTED_TOOLS.issubset(names), (
            f"missing: {EXPECTED_TOOLS - names}, unexpected: {names - EXPECTED_TOOLS}"
        )

    async def test_tool_responses_include_correlation_id(self, server):
        result = await server._tool_manager.call_tool(
            "create_dataset",
            {"bq_dataset": "d", "bq_table": "t", "name": "n"},
        )
        assert "correlation_id" in result
        assert result["correlation_id"]

    async def test_tool_error_envelope_shape(self, server):
        # GKE stub returns deployment_name; force an error by promoting an
        # unknown model version. promote_model surfaces a domain ValueError.
        result = await server._tool_manager.call_tool(
            "promote_model",
            {"model_id": "does-not-exist", "version": 99, "stage": "production"},
        )
        assert result.get("isError") is True
        assert "error" in result
        assert "correlation_id" in result
