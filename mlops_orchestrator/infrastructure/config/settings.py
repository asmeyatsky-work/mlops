from __future__ import annotations
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables with MLOPS_ prefix."""

    gcp_project: str = Field(default="mlops-491617", description="GCP project ID")
    gcp_location: str = Field(default="us-central1", description="GCP region")
    train_image: str = Field(
        default="us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-12:latest",
        description="Default training container image",
    )
    use_stubs: bool = Field(
        default=True,
        description="Use stub adapters (True) or real GCP adapters (False)",
    )
    transport: str = Field(
        default="stdio",
        description="MCP transport: stdio or sse",
    )

    model_config = {"env_prefix": "MLOPS_"}
