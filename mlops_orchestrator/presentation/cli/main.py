"""
MLOps Orchestrator — MCP Server Entry Point.

Usage:
    python -m mlops_orchestrator.presentation.cli.main

Environment Variables:
    MLOPS_GCP_PROJECT  — GCP project ID
    MLOPS_GCP_LOCATION — GCP region (default: us-central1)
    MLOPS_USE_STUBS    — Use stub adapters (default: false)
    MLOPS_TRANSPORT    — MCP transport: stdio or sse (default: stdio)
    MLOPS_AUTH_ENABLED — Enable API key / JWT authentication (default: false)
    MLOPS_AUTH_API_KEYS — Comma-separated API keys
    MLOPS_AUTH_JWT_SECRET — JWT signing secret (HS256)
"""
from __future__ import annotations
import sys


def main() -> None:
    try:
        from mlops_orchestrator.infrastructure.config.settings import Settings
        from mlops_orchestrator.infrastructure.config.container import DependencyContainer
        from mlops_orchestrator.infrastructure.mcp_servers.server import create_mlops_server
        from mlops_orchestrator.infrastructure.logging_config import configure_logging
        from mlops_orchestrator.infrastructure.auth.auth_middleware import AuthConfig

        settings = Settings()
        configure_logging(json_output=True)

        if settings.use_stubs:
            print("WARNING: Running with stub adapters (MLOPS_USE_STUBS=true). "
                  "GCP operations will be simulated.", file=sys.stderr)

        # Build auth config from settings
        auth_config = AuthConfig(
            enabled=settings.auth_enabled,
            api_keys=tuple(k.strip() for k in settings.auth_api_keys.split(",") if k.strip()),
            jwt_secret=settings.auth_jwt_secret,
        )

        container = DependencyContainer(settings)
        mcp = create_mlops_server(container, auth_config=auth_config)
        mcp.run(transport=settings.transport)
    except Exception as e:
        print(f"Error starting MLOps Orchestrator: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
