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

    def test_empty_bq_dataset_raises(self):
        """Empty string bq_dataset should violate min_length=1."""
        with pytest.raises(ValidationError):
            CreateDatasetRequest(bq_dataset="", bq_table="tbl", name="test")

    def test_empty_bq_table_raises(self):
        """Empty string bq_table should violate min_length=1."""
        with pytest.raises(ValidationError):
            CreateDatasetRequest(bq_dataset="ds", bq_table="", name="test")

    def test_empty_name_raises(self):
        """Empty string name should violate min_length=1."""
        with pytest.raises(ValidationError):
            CreateDatasetRequest(bq_dataset="ds", bq_table="tbl", name="")


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

    def test_empty_model_name_raises(self):
        """Empty string model_name should violate min_length=1."""
        with pytest.raises(ValidationError):
            TrainModelRequest(model_name="", dataset_id="ds-1")

    def test_with_both_dataset_id_and_gcs_uri_succeeds(self):
        """Providing both dataset_id and gcs_uri should succeed."""
        req = TrainModelRequest(model_name="m", dataset_id="ds-1", gcs_uri="gs://b/d")
        assert req.dataset_id == "ds-1"
        assert req.gcs_uri == "gs://b/d"


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

    def test_empty_model_id_vertex_raises(self):
        """Empty model_id should violate min_length=1 on DeployToVertexRequest."""
        with pytest.raises(ValidationError):
            DeployToVertexRequest(model_id="", endpoint_name="ep")

    def test_empty_model_id_gke_raises(self):
        """Empty model_id should violate min_length=1 on DeployToGkeRequest."""
        with pytest.raises(ValidationError):
            DeployToGkeRequest(model_id="", cluster_name="c")

    def test_empty_endpoint_name_raises(self):
        """Empty endpoint_name should violate min_length=1."""
        with pytest.raises(ValidationError):
            DeployToVertexRequest(model_id="m", endpoint_name="")

    def test_empty_cluster_name_raises(self):
        """Empty cluster_name should violate min_length=1."""
        with pytest.raises(ValidationError):
            DeployToGkeRequest(model_id="m", cluster_name="")

    def test_deployment_response_target_literal(self):
        """DeploymentResponse target must be 'vertex' or 'gke'."""
        resp_v = DeploymentResponse(resource_name="r", status="OK", target="vertex")
        resp_g = DeploymentResponse(resource_name="r", status="OK", target="gke")
        assert resp_v.target == "vertex"
        assert resp_g.target == "gke"

    def test_deployment_response_invalid_target(self):
        """Invalid target literal raises ValidationError."""
        with pytest.raises(ValidationError):
            DeploymentResponse(resource_name="r", status="OK", target="sagemaker")  # type: ignore[arg-type]

    def test_missing_required_vertex_fields(self):
        """DeployToVertexRequest with missing required fields raises."""
        with pytest.raises(ValidationError):
            DeployToVertexRequest(model_id="m")  # type: ignore[call-arg]

    def test_missing_required_gke_fields(self):
        """DeployToGkeRequest with missing required fields raises."""
        with pytest.raises(ValidationError):
            DeployToGkeRequest(model_id="m")  # type: ignore[call-arg]


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

    def test_empty_endpoint_id_raises(self):
        """Empty endpoint_id should violate min_length=1."""
        with pytest.raises(ValidationError):
            MonitoringRequest(endpoint_id="")

    def test_drift_threshold_above_1_raises(self):
        """drift_threshold > 1.0 should violate le=1.0."""
        with pytest.raises(ValidationError):
            MonitoringRequest(endpoint_id="ep", drift_threshold=1.1)
