"""Tests for domain value objects."""
from __future__ import annotations

import pytest
from datetime import datetime, UTC

from mlops_orchestrator.domain.value_objects.resource_name import ResourceName
from mlops_orchestrator.domain.value_objects.bq_source import BigQuerySource
from mlops_orchestrator.domain.value_objects.gcs_uri import GcsUri
from mlops_orchestrator.domain.value_objects.model_artifact import ModelArtifact
from mlops_orchestrator.domain.value_objects.machine_spec import MachineSpec
from mlops_orchestrator.domain.value_objects.drift_result import (
    DriftResult, DriftType, DriftSeverity, _compute_severity,
)
from mlops_orchestrator.domain.value_objects.cost_metrics import (
    CostMetrics, CostBreakdown, CostRecommendation,
)
from mlops_orchestrator.domain.value_objects.compliance import (
    RiskTier, RiskClassification, ModelCard,
)


# -- ResourceName ----------------------------------------------------------

class TestResourceName:
    def test_full_name(self):
        rn = ResourceName("my-proj", "us-central1", "datasets", "ds-123")
        assert rn.full_name == "projects/my-proj/locations/us-central1/datasets/ds-123"

    def test_str_returns_full_name(self):
        rn = ResourceName("p", "l", "t", "id")
        assert str(rn) == rn.full_name

    def test_from_string_valid(self):
        rn = ResourceName.from_string(
            "projects/my-proj/locations/us-central1/datasets/ds-123"
        )
        assert rn.project == "my-proj"
        assert rn.location == "us-central1"
        assert rn.resource_type == "datasets"
        assert rn.resource_id == "ds-123"

    def test_from_string_with_nested_id(self):
        rn = ResourceName.from_string(
            "projects/p/locations/l/models/m/versions/v1"
        )
        assert rn.resource_id == "m/versions/v1"

    def test_from_string_invalid_format(self):
        with pytest.raises(ValueError, match="Invalid resource name"):
            ResourceName.from_string("not/a/valid/name")

    def test_from_string_missing_projects_prefix(self):
        with pytest.raises(ValueError):
            ResourceName.from_string("repos/p/locations/l/datasets/d")

    def test_from_string_missing_locations(self):
        with pytest.raises(ValueError):
            ResourceName.from_string("projects/p/regions/l/datasets/d")

    def test_frozen(self):
        rn = ResourceName("p", "l", "t", "id")
        with pytest.raises(AttributeError):
            rn.project = "other"  # type: ignore[misc]

    def test_roundtrip(self):
        original = "projects/p/locations/l/datasets/d"
        assert str(ResourceName.from_string(original)) == original

    def test_from_string_missing_resource_id_trailing_slash(self):
        """Trailing slash means empty resource_id after join -- should raise."""
        with pytest.raises(ValueError, match="missing resource ID"):
            ResourceName.from_string("projects/p/locations/l/datasets/")

    def test_from_string_exactly_five_parts(self):
        """Exactly 5 parts (no resource_id) should fail the len < 6 check."""
        with pytest.raises(ValueError, match="Invalid resource name"):
            ResourceName.from_string("projects/p/locations/l/datasets")


# -- BigQuerySource --------------------------------------------------------

class TestBigQuerySource:
    def test_creation(self):
        bq = BigQuerySource(dataset="my_ds", table="my_tbl")
        assert bq.dataset == "my_ds"
        assert bq.table == "my_tbl"
        assert bq.project == ""

    def test_to_uri_without_project(self):
        bq = BigQuerySource(dataset="ds", table="tbl")
        assert bq.to_uri() == "bq://ds.tbl"

    def test_to_uri_with_project(self):
        bq = BigQuerySource(dataset="ds", table="tbl", project="proj")
        assert bq.to_uri() == "bq://proj.ds.tbl"

    def test_empty_dataset_raises(self):
        with pytest.raises(ValueError, match="dataset name cannot be empty"):
            BigQuerySource(dataset="", table="tbl")

    def test_empty_table_raises(self):
        with pytest.raises(ValueError, match="table name cannot be empty"):
            BigQuerySource(dataset="ds", table="")


# -- GcsUri ----------------------------------------------------------------

class TestGcsUri:
    def test_valid_uri(self):
        uri = GcsUri("gs://my-bucket/path/to/file")
        assert uri.bucket == "my-bucket"
        assert uri.path == "path/to/file"

    def test_bucket_only(self):
        uri = GcsUri("gs://my-bucket")
        assert uri.bucket == "my-bucket"
        assert uri.path == ""

    def test_invalid_prefix_raises(self):
        with pytest.raises(ValueError, match="must start with 'gs://'"):
            GcsUri("s3://bucket/key")

    def test_bucket_with_trailing_slash(self):
        uri = GcsUri("gs://bucket/")
        assert uri.bucket == "bucket"
        assert uri.path == ""

    def test_empty_bucket_gs_prefix_only(self):
        """gs:// with no bucket name should raise ValueError."""
        with pytest.raises(ValueError, match="must include a bucket name"):
            GcsUri("gs://")

    def test_empty_string_raises(self):
        """Empty string does not start with gs:// -- should raise."""
        with pytest.raises(ValueError, match="must start with 'gs://'"):
            GcsUri("")


# -- ModelArtifact ---------------------------------------------------------

class TestModelArtifact:
    def test_create_factory(self):
        before = datetime.now(UTC)
        artifact = ModelArtifact.create("rn", "tensorflow", "gs://b/model")
        after = datetime.now(UTC)
        assert artifact.resource_name == "rn"
        assert artifact.framework == "tensorflow"
        assert artifact.artifact_uri == "gs://b/model"
        assert before <= artifact.created_at <= after

    def test_frozen(self):
        artifact = ModelArtifact.create("rn", "pytorch", "gs://b/m")
        with pytest.raises(AttributeError):
            artifact.framework = "jax"  # type: ignore[misc]

    def test_empty_resource_name_raises(self):
        with pytest.raises(ValueError, match="resource_name cannot be empty"):
            ModelArtifact(
                resource_name="",
                framework="tf",
                artifact_uri="gs://b/m",
                created_at=datetime.now(UTC),
            )

    def test_empty_framework_raises(self):
        with pytest.raises(ValueError, match="framework cannot be empty"):
            ModelArtifact(
                resource_name="rn",
                framework="",
                artifact_uri="gs://b/m",
                created_at=datetime.now(UTC),
            )

    def test_empty_artifact_uri_raises(self):
        with pytest.raises(ValueError, match="artifact_uri cannot be empty"):
            ModelArtifact(
                resource_name="rn",
                framework="tf",
                artifact_uri="",
                created_at=datetime.now(UTC),
            )


# -- MachineSpec -----------------------------------------------------------

class TestMachineSpec:
    def test_defaults(self):
        spec = MachineSpec()
        assert spec.machine_type == "n1-standard-4"
        assert spec.accelerator_count == 0
        assert spec.replica_count == 1
        assert not spec.has_gpu

    def test_has_gpu_true(self):
        spec = MachineSpec(accelerator_type="NVIDIA_TESLA_T4", accelerator_count=1)
        assert spec.has_gpu

    def test_has_gpu_false_when_count_zero(self):
        spec = MachineSpec(accelerator_type="NVIDIA_TESLA_T4", accelerator_count=0)
        assert not spec.has_gpu

    def test_has_gpu_false_when_type_empty(self):
        spec = MachineSpec(accelerator_type="", accelerator_count=2)
        assert not spec.has_gpu

    def test_negative_accelerator_count_raises(self):
        with pytest.raises(ValueError, match="accelerator_count must be >= 0"):
            MachineSpec(accelerator_count=-1)

    def test_zero_replica_count_raises(self):
        with pytest.raises(ValueError, match="replica_count must be >= 1"):
            MachineSpec(replica_count=0)


# -- DriftResult -----------------------------------------------------------

class TestDriftResult:
    def test_is_drifted_below_threshold(self):
        dr = DriftResult.from_test("feat", "ks", DriftType.DATA, 0.5, 0.01)
        assert dr.is_drifted

    def test_not_drifted_above_threshold(self):
        dr = DriftResult.from_test("feat", "ks", DriftType.DATA, 0.01, 0.5)
        assert not dr.is_drifted

    def test_custom_threshold(self):
        dr = DriftResult.from_test("f", "t", DriftType.FEATURE, 0.1, 0.08, threshold=0.1)
        assert dr.is_drifted

    def test_from_test_computes_severity(self):
        dr = DriftResult.from_test("f", "ks", DriftType.DATA, 0.35, 0.01)
        assert dr.severity == DriftSeverity.CRITICAL

    def test_exact_threshold_boundary_not_drifted(self):
        """p_value == threshold means NOT drifted (strict less-than)."""
        dr = DriftResult.from_test("f", "ks_test", DriftType.DATA, 0.1, 0.05, threshold=0.05)
        assert not dr.is_drifted

    # -- _compute_severity with default ks_test thresholds --
    @pytest.mark.parametrize("stat, expected", [
        (0.01, DriftSeverity.NONE),
        (0.07, DriftSeverity.LOW),
        (0.15, DriftSeverity.MEDIUM),
        (0.25, DriftSeverity.HIGH),
        (0.5, DriftSeverity.CRITICAL),
    ])
    def test_compute_severity(self, stat, expected):
        assert _compute_severity(stat) == expected

    # -- _compute_severity boundary values (exact thresholds) --
    # ks_test thresholds: (0.05, 0.1, 0.2, 0.3)
    def test_compute_severity_ks_exact_boundary_none_to_low(self):
        """statistic == 0.05 is NOT < 0.05, so it's LOW."""
        assert _compute_severity(0.05, "ks_test") == DriftSeverity.LOW

    def test_compute_severity_ks_exact_boundary_low_to_medium(self):
        """statistic == 0.1 is NOT < 0.1, so it's MEDIUM."""
        assert _compute_severity(0.1, "ks_test") == DriftSeverity.MEDIUM

    def test_compute_severity_ks_exact_boundary_medium_to_high(self):
        """statistic == 0.2 is NOT < 0.2, so it's HIGH."""
        assert _compute_severity(0.2, "ks_test") == DriftSeverity.HIGH

    def test_compute_severity_ks_exact_boundary_high_to_critical(self):
        """statistic == 0.3 is NOT < 0.3, so it's CRITICAL."""
        assert _compute_severity(0.3, "ks_test") == DriftSeverity.CRITICAL

    # -- _compute_severity with chi_square thresholds: (3.84, 7.81, 15.51, 30.0)
    @pytest.mark.parametrize("stat, expected", [
        (1.0, DriftSeverity.NONE),
        (3.84, DriftSeverity.LOW),
        (5.0, DriftSeverity.LOW),
        (7.81, DriftSeverity.MEDIUM),
        (10.0, DriftSeverity.MEDIUM),
        (15.51, DriftSeverity.HIGH),
        (30.0, DriftSeverity.CRITICAL),
        (50.0, DriftSeverity.CRITICAL),
    ])
    def test_compute_severity_chi_square(self, stat, expected):
        assert _compute_severity(stat, "chi_square") == expected

    # -- _compute_severity with psi thresholds: (0.1, 0.2, 0.3, 0.5)
    @pytest.mark.parametrize("stat, expected", [
        (0.05, DriftSeverity.NONE),
        (0.1, DriftSeverity.LOW),
        (0.15, DriftSeverity.LOW),
        (0.2, DriftSeverity.MEDIUM),
        (0.3, DriftSeverity.HIGH),
        (0.5, DriftSeverity.CRITICAL),
    ])
    def test_compute_severity_psi(self, stat, expected):
        assert _compute_severity(stat, "psi") == expected

    # -- _compute_severity with kl_divergence thresholds: (0.05, 0.1, 0.3, 0.5)
    @pytest.mark.parametrize("stat, expected", [
        (0.01, DriftSeverity.NONE),
        (0.05, DriftSeverity.LOW),
        (0.1, DriftSeverity.MEDIUM),
        (0.3, DriftSeverity.HIGH),
        (0.5, DriftSeverity.CRITICAL),
    ])
    def test_compute_severity_kl_divergence(self, stat, expected):
        assert _compute_severity(stat, "kl_divergence") == expected


# -- CostMetrics -----------------------------------------------------------

class TestCostMetrics:
    def test_has_waste_true(self):
        cm = CostMetrics(gpu_idle_pct=31.0)
        assert cm.has_waste

    def test_has_waste_false(self):
        cm = CostMetrics(gpu_idle_pct=30.0)
        assert not cm.has_waste

    def test_defaults(self):
        cm = CostMetrics()
        assert cm.cost_per_tb_scanned == 0.0
        assert not cm.has_waste


class TestCostBreakdown:
    def test_total(self):
        cb = CostBreakdown(compute_cost=10.0, storage_cost=5.0, network_cost=2.0)
        assert cb.total == 17.0

    def test_total_defaults(self):
        cb = CostBreakdown()
        assert cb.total == 0.0


class TestCostRecommendation:
    def test_fields(self):
        cr = CostRecommendation("resize", "Downsize VM", 100.0, "high")
        assert cr.recommendation_type == "resize"
        assert cr.estimated_savings == 100.0


# -- Compliance ------------------------------------------------------------

class TestRiskClassification:
    def test_creation(self):
        rc = RiskClassification(
            tier=RiskTier.HIGH,
            domain="healthcare",
            justification="Medical diagnosis",
        )
        assert rc.tier == RiskTier.HIGH
        assert isinstance(rc.assessed_at, datetime)


class TestModelCard:
    def test_is_complete_true(self):
        mc = ModelCard(
            model_name="m",
            version="1",
            purpose="Classify images",
            limitations="Not for medical use",
            data_sources=("imagenet",),
            accuracy_metrics=(("f1", 0.95),),
        )
        assert mc.is_complete

    def test_is_complete_false_missing_purpose(self):
        mc = ModelCard(model_name="m", version="1", purpose="", limitations="x",
                       data_sources=("d",), accuracy_metrics=(("f1", 0.9),))
        assert not mc.is_complete

    def test_is_complete_false_no_data_sources(self):
        mc = ModelCard(model_name="m", version="1", purpose="p", limitations="l",
                       data_sources=(), accuracy_metrics=(("f1", 0.9),))
        assert not mc.is_complete

    def test_is_complete_false_no_metrics(self):
        mc = ModelCard(model_name="m", version="1", purpose="p", limitations="l",
                       data_sources=("d",), accuracy_metrics=())
        assert not mc.is_complete

    def test_is_complete_false_when_limitations_empty(self):
        mc = ModelCard(model_name="m", version="1", purpose="p", limitations="",
                       data_sources=("d",), accuracy_metrics=(("f1", 0.9),))
        assert not mc.is_complete
