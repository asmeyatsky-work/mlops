"""
MLOps Orchestrator — MCP Server Entry Point.

Usage:
    python -m mlops_orchestrator.presentation.cli.main

Environment Variables:
    MLOPS_GCP_PROJECT  — GCP project ID
    MLOPS_GCP_LOCATION — GCP region (default: us-central1)
    MLOPS_USE_STUBS    — Use stub adapters (default: false)
    MLOPS_TRANSPORT    — MCP transport: stdio or sse (default: stdio)
"""
from __future__ import annotations
import sys


def main() -> None:
    try:
        from mlops_orchestrator.infrastructure.config.settings import Settings
        from mlops_orchestrator.infrastructure.config.container import DependencyContainer
        from mlops_orchestrator.infrastructure.mcp_servers.server import create_mlops_server
        from mlops_orchestrator.infrastructure.logging_config import configure_logging

        settings = Settings()
        configure_logging(json_output=True)

        if settings.use_stubs:
            print("WARNING: Running with stub adapters (MLOPS_USE_STUBS=true). "
                  "GCP operations will be simulated.", file=sys.stderr)
        container = DependencyContainer(settings)
        mcp = create_mlops_server(container)
        mcp.run(transport=settings.transport)
    except Exception as e:
        print(f"Error starting MLOps Orchestrator: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
