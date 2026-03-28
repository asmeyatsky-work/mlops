"""Tests for application DTOs."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from mlops_orchestrator.application.dtos.dataset_dto import (
    CreateDatasetRequest, DatasetResponse,
)
from mlops_orchestrator.application.dtos.training_dto import (
    TrainModelRequest, TrainingResponse,
)
from mlops_orchestrator.application.dtos.deployment_dto import (
    DeployToVertexRequest, DeployToGkeRequest, DeploymentResponse,
    MonitoringRequest, MonitoringResponse,
)


class TestCreateDatasetRequest:
    def test_valid(self):
        req = CreateDatasetRequest(bq_dataset="ds", bq_table="tbl", name="test")
        assert req.bq_dataset == "ds"

    def test_missing_field(self):
        with pytest.raises(ValidationError):
            CreateDatasetRequest(bq_dataset="ds", bq_table="tbl")  # type: ignore[call-arg]


class TestTrainModelRequest:
    def test_with_dataset_id(self):
        req = TrainModelRequest(model_name="m", dataset_id="ds-1")
        assert req.dataset_id == "ds-1"

    def test_with_gcs_uri(self):
        req = TrainModelRequest(model_name="m", gcs_uri="gs://b/d")
        assert req.gcs_uri == "gs://b/d"

    def test_neither_raises(self):
        with pytest.raises(ValidationError, match="dataset_id or gcs_uri"):
            TrainModelRequest(model_name="m")


class TestDeploymentDtos:
    def test_vertex_request(self):
        req = DeployToVertexRequest(model_id="m", endpoint_name="ep")
        assert req.model_id == "m"

    def test_gke_request(self):
        req = DeployToGkeRequest(model_id="m", cluster_name="c")
        assert req.cluster_name == "c"

    def test_deployment_response(self):
        resp = DeploymentResponse(resource_name="rn", status="DEPLOYED", target="vertex")
        assert resp.target == "vertex"


class TestMonitoringDtos:
    def test_default_thresholds(self):
        req = MonitoringRequest(endpoint_id="ep")
        assert req.drift_threshold == 0.05
        assert req.skew_threshold == 0.1

    def test_threshold_bounds(self):
        with pytest.raises(ValidationError):
            MonitoringRequest(endpoint_id="ep", drift_threshold=-0.1)
        with pytest.raises(ValidationError):
            MonitoringRequest(endpoint_id="ep", skew_threshold=1.5)

    def test_monitoring_response(self):
        resp = MonitoringResponse(endpoint_id="ep", monitoring_enabled=True, status="ACTIVE")
        assert resp.monitoring_enabled
