from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class DeployToVertexRequest(BaseModel):
    """Input DTO for Vertex AI deployment."""
    model_id: str = Field(..., min_length=1, description="Model resource name to deploy")
    endpoint_name: str = Field(..., min_length=1, description="Display name for the endpoint")


class DeployToGkeRequest(BaseModel):
    """Input DTO for GKE deployment."""
    model_id: str = Field(..., min_length=1, description="Model resource name to deploy")
    cluster_name: str = Field(..., min_length=1, description="Target GKE cluster name")


class DeploymentResponse(BaseModel):
    """Output DTO for deployment operations."""
    resource_name: str
    status: str
    target: Literal["vertex", "gke"]


class MonitoringRequest(BaseModel):
    """Input DTO for monitoring configuration."""
    endpoint_id: str = Field(..., min_length=1, description="Endpoint resource name to monitor")
    drift_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    skew_threshold: float = Field(default=0.1, ge=0.0, le=1.0)


class MonitoringResponse(BaseModel):
    """Output DTO for monitoring configuration."""
    endpoint_id: str
    monitoring_enabled: bool
    status: str
