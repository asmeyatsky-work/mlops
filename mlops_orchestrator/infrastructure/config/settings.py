from __future__ import annotations
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables with MLOPS_ prefix."""

    gcp_project: str = Field(default="", description="GCP project ID (required for GCP mode)")
    gcp_location: str = Field(default="us-central1", description="GCP region")
    train_image: str = Field(
        default="us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-12:latest",
        description="Default training container image",
    )
    use_stubs: bool = Field(
        default=False,
        description="Use stub adapters (True) or real GCP adapters (False)",
    )
    transport: Literal["stdio", "sse"] = Field(
        default="stdio",
        description="MCP transport: stdio or sse",
    )

    # Billing / Cost
    billing_table: str = Field(
        default="",
        description="Fully-qualified BigQuery billing export table",
    )

    # Alerting
    slack_webhook_url: str = Field(default="", description="Slack incoming webhook URL")
    pagerduty_routing_key: str = Field(default="", description="PagerDuty Events API v2 routing key")
    alert_email_smtp_host: str = Field(default="", description="SMTP host for email alerts")
    alert_email_smtp_port: int = Field(default=587, description="SMTP port")
    alert_email_sender: str = Field(default="", description="Email alert sender address")
    alert_email_recipients: str = Field(default="", description="Comma-separated alert recipient emails")
    alert_email_username: str = Field(default="", description="SMTP username")
    alert_email_password: str = Field(default="", description="SMTP password")

    # Auth
    auth_enabled: bool = Field(default=False, description="Enable API key / JWT auth on MCP server")
    auth_api_keys: str = Field(default="", description="Comma-separated API keys")
    auth_jwt_secret: str = Field(default="", description="JWT signing secret (HS256)")

    model_config = {"env_prefix": "MLOPS_"}
