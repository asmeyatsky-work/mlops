"""
MLOps Orchestrator — MCP Server Entry Point.

Usage:
    python -m mlops_orchestrator.presentation.cli.main

Environment Variables:
    MLOPS_GCP_PROJECT  — GCP project ID (default: mlops-491617)
    MLOPS_GCP_LOCATION — GCP region (default: us-central1)
    MLOPS_USE_STUBS    — Use stub adapters (default: true)
    MLOPS_TRANSPORT    — MCP transport: stdio or sse (default: stdio)
"""
from __future__ import annotations


def main() -> None:
    from mlops_orchestrator.infrastructure.config.settings import Settings
    from mlops_orchestrator.infrastructure.config.container import DependencyContainer
    from mlops_orchestrator.infrastructure.mcp_servers.server import create_mlops_server

    settings = Settings()
    container = DependencyContainer(settings)
    mcp = create_mlops_server(container)
    mcp.run(transport=settings.transport)


if __name__ == "__main__":
    main()
