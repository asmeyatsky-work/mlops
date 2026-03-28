"""Tests for domain services."""
from __future__ import annotations

import pytest

from mlops_orchestrator.domain.value_objects.bq_source import BigQuerySource
from mlops_orchestrator.domain.value_objects.drift_result import (
    DriftResult, DriftType, DriftSeverity,
)
from mlops_orchestrator.domain.services.dataset_service import DatasetDomainService
from mlops_orchestrator.domain.services.drift_detection_service import DriftDetectionService
from mlops_orchestrator.domain.services.compliance_service import ComplianceService
from mlops_orchestrator.domain.value_objects.compliance import RiskTier
from mlops_orchestrator.domain.services.remediation_service import (
    RemediationService, RemediationStrategy,
)


# -- DatasetDomainService --------------------------------------------------

class TestDatasetDomainService:
    svc = DatasetDomainService()

    def test_create_managed_dataset(self):
        ds = self.svc.create_managed_dataset("my_ds", "my_tbl", "display")
        assert ds.display_name == "display"
        assert ds.bq_source.dataset == "my_ds"
        assert ds.status == "PENDING"

    def test_validate_bq_source_valid(self):
        bq = BigQuerySource(dataset="ds", table="tbl")
        assert self.svc.validate_bq_source(bq) == []

    def test_validate_bq_source_dots_in_dataset(self):
        bq = BigQuerySource(dataset="my.ds", table="tbl")
        errors = self.svc.validate_bq_source(bq)
        assert "Dataset name must not contain dots" in errors

    def test_validate_bq_source_dots_in_table(self):
        bq = BigQuerySource(dataset="ds", table="my.tbl")
        errors = self.svc.validate_bq_source(bq)
        assert "Table name must not contain dots" in errors

    def test_validate_bq_source_dots_in_both(self):
        """Both dataset and table contain dots -- should report both errors."""
        bq = BigQuerySource(dataset="my.ds", table="my.tbl")
        errors = self.svc.validate_bq_source(bq)
        assert "Dataset name must not contain dots" in errors
        assert "Table name must not contain dots" in errors
        assert len(errors) == 2


# -- DriftDetectionService -------------------------------------------------

class TestDriftDetectionService:
    svc = DriftDetectionService()

    def test_ks_test_identical_distributions(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = self.svc.ks_test(data, data)
        assert result.statistic == 0.0
        assert not result.is_drifted

    def test_ks_test_different_distributions(self):
        baseline = [1.0, 2.0, 3.0, 4.0, 5.0] * 20
        current = [10.0, 20.0, 30.0, 40.0, 50.0] * 20
        result = self.svc.ks_test(baseline, current)
        assert result.statistic > 0.5
        assert result.is_drifted

    def test_ks_test_empty_input(self):
        result = self.svc.ks_test([], [1.0])
        assert result.statistic == 0.0
        assert result.p_value == 1.0

    def test_ks_test_both_empty(self):
        """Both baseline and current empty should return no-drift result."""
        result = self.svc.ks_test([], [])
        assert result.statistic == 0.0
        assert result.p_value == 1.0
        assert not result.is_drifted

    def test_chi_square_identical(self):
        counts = {"a": 100, "b": 100, "c": 100}
        result = self.svc.chi_square_test(counts, counts)
        assert result.statistic == pytest.approx(0.0)
        assert not result.is_drifted

    def test_chi_square_different(self):
        baseline = {"a": 100, "b": 100}
        current = {"a": 10, "b": 190}
        result = self.svc.chi_square_test(baseline, current)
        assert result.statistic > 0
        assert result.is_drifted

    def test_chi_square_empty(self):
        result = self.svc.chi_square_test({}, {})
        assert result.statistic == 0.0

    def test_kl_divergence_identical(self):
        dist = [0.25, 0.25, 0.25, 0.25]
        result = self.svc.kl_divergence(dist, dist)
        assert result.statistic == pytest.approx(0.0, abs=1e-6)

    def test_kl_divergence_different(self):
        p = [0.9, 0.1]
        q = [0.1, 0.9]
        result = self.svc.kl_divergence(p, q)
        assert result.statistic > 0.5

    def test_kl_divergence_mismatched_length(self):
        with pytest.raises(ValueError, match="same length"):
            self.svc.kl_divergence([0.5, 0.5], [0.3, 0.3, 0.4])

    def test_kl_divergence_all_zeros(self):
        """All-zero distributions: P is all zeros so KL = 0 (all bins skipped)."""
        result = self.svc.kl_divergence([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        assert result.statistic == pytest.approx(0.0, abs=1e-6)
        assert not result.is_drifted

    def test_kl_divergence_empty_lists(self):
        """Empty distributions (length 0) should produce KL = 0."""
        result = self.svc.kl_divergence([], [])
        assert result.statistic == pytest.approx(0.0, abs=1e-6)

    def test_psi_identical(self):
        data = [float(i) for i in range(100)]
        result = self.svc.psi(data, data)
        assert result.statistic == pytest.approx(0.0, abs=0.01)

    def test_psi_shifted(self):
        baseline = [float(i) for i in range(100)]
        current = [float(i + 50) for i in range(100)]
        result = self.svc.psi(baseline, current)
        assert result.statistic > 0

    def test_psi_empty(self):
        result = self.svc.psi([], [1.0])
        assert result.statistic == 0.0

    def test_psi_constant_values(self):
        result = self.svc.psi([1.0, 1.0], [1.0, 1.0])
        assert result.statistic == 0.0

    def test_psi_single_element(self):
        """Single-element lists where min == max should return 0 (no bins)."""
        result = self.svc.psi([5.0], [5.0])
        assert result.statistic == 0.0
        assert result.p_value == 1.0
        assert not result.is_drifted

    def test_evaluate_features(self):
        baseline = {"f1": [1.0, 2.0, 3.0] * 10, "f2": [4.0, 5.0, 6.0] * 10}
        current = {"f1": [1.0, 2.0, 3.0] * 10, "f2": [40.0, 50.0, 60.0] * 10}
        results = self.svc.evaluate_features(baseline, current)
        assert len(results) == 2
        names = [r.feature_name for r in results]
        assert "f1" in names
        assert "f2" in names

    def test_evaluate_features_skips_missing(self):
        baseline = {"f1": [1.0, 2.0]}
        current = {"f2": [3.0, 4.0]}
        results = self.svc.evaluate_features(baseline, current)
        assert len(results) == 0


# -- ComplianceService -----------------------------------------------------

class TestComplianceService:
    svc = ComplianceService()

    def test_classify_prohibited(self):
        rc = self.svc.classify_risk("gov", "social scoring system", False)
        assert rc.tier == RiskTier.PROHIBITED
        assert "immediate_withdrawal" in rc.required_controls

    def test_classify_high_risk_domain(self):
        rc = self.svc.classify_risk("healthcare", "diagnosis", False)
        assert rc.tier == RiskTier.HIGH
        assert "risk_management_system" in rc.required_controls

    def test_classify_high_risk_fundamental_rights(self):
        rc = self.svc.classify_risk("retail", "product recommendation", True)
        assert rc.tier == RiskTier.HIGH

    def test_classify_limited_chatbot(self):
        rc = self.svc.classify_risk("retail", "customer chatbot", False)
        assert rc.tier == RiskTier.LIMITED
        assert "transparency_notice" in rc.required_controls

    def test_classify_minimal(self):
        rc = self.svc.classify_risk("retail", "inventory optimization", False)
        assert rc.tier == RiskTier.MINIMAL
        assert rc.required_controls == ()

    def test_generate_model_card(self):
        mc = self.svc.generate_model_card(
            "m", "1", "classify", "not medical", ("imagenet",), (("f1", 0.9),)
        )
        assert mc.is_complete
        assert mc.model_name == "m"

    def test_validate_article_10_all_good(self):
        gaps = self.svc.validate_article_10(("source1",), True, True)
        assert gaps == []

    def test_validate_article_10_all_bad(self):
        gaps = self.svc.validate_article_10((), False, False)
        assert len(gaps) == 3

    def test_validate_article_15_all_good(self):
        gaps = self.svc.validate_article_15((("f1", 0.9),), True, True)
        assert gaps == []

    def test_validate_article_15_all_bad(self):
        gaps = self.svc.validate_article_15((), False, False)
        assert len(gaps) == 3

    # -- All prohibited keywords (parametrized) --
    @pytest.mark.parametrize("keyword", [
        "social_scoring",
        "mass_surveillance",
        "subliminal_manipulation",
        "exploitation_of_vulnerability",
        "real_time_biometric_identification",
    ])
    def test_all_prohibited_keywords(self, keyword):
        """Each prohibited keyword should trigger PROHIBITED tier."""
        purpose = keyword.replace("_", " ")
        rc = self.svc.classify_risk("any_domain", purpose, False)
        assert rc.tier == RiskTier.PROHIBITED

    # -- All transparency keywords (parametrized) --
    @pytest.mark.parametrize("keyword", [
        "chatbot",
        "deepfake",
        "emotion_recognition",
        "biometric",
    ])
    def test_all_transparency_keywords(self, keyword):
        """Each transparency keyword should trigger LIMITED tier (non-high-risk domain)."""
        purpose = keyword.replace("_", " ")
        rc = self.svc.classify_risk("retail", purpose, False)
        assert rc.tier == RiskTier.LIMITED

    def test_prohibited_overrides_high_risk_domain(self):
        """Even in a high-risk domain, prohibited purpose should yield PROHIBITED."""
        rc = self.svc.classify_risk("healthcare", "social scoring system", True)
        assert rc.tier == RiskTier.PROHIBITED


# -- RemediationService ----------------------------------------------------

class TestRemediationService:
    svc = RemediationService()

    def _drift(self, severity: DriftSeverity, dtype: DriftType = DriftType.DATA) -> DriftResult:
        stat_map = {
            DriftSeverity.NONE: 0.01,
            DriftSeverity.LOW: 0.07,
            DriftSeverity.MEDIUM: 0.15,
            DriftSeverity.HIGH: 0.25,
            DriftSeverity.CRITICAL: 0.5,
        }
        return DriftResult.from_test(
            "f", "ks_test", dtype, stat_map[severity], 0.001,
        )

    def test_no_results_no_action(self):
        assert self.svc.select_strategy([]) == RemediationStrategy.NO_ACTION

    def test_critical_rolls_back(self):
        assert self.svc.select_strategy([self._drift(DriftSeverity.CRITICAL)]) == RemediationStrategy.ROLLBACK

    def test_high_data_drift_incremental(self):
        assert self.svc.select_strategy(
            [self._drift(DriftSeverity.HIGH, DriftType.DATA)]
        ) == RemediationStrategy.INCREMENTAL_TRAINING

    def test_high_non_data_ensemble(self):
        assert self.svc.select_strategy(
            [self._drift(DriftSeverity.HIGH, DriftType.PREDICTION)]
        ) == RemediationStrategy.ENSEMBLE_SWITCHING

    def test_medium_ensemble(self):
        assert self.svc.select_strategy(
            [self._drift(DriftSeverity.MEDIUM)]
        ) == RemediationStrategy.ENSEMBLE_SWITCHING

    def test_low_concept_active_learning(self):
        assert self.svc.select_strategy(
            [self._drift(DriftSeverity.LOW, DriftType.CONCEPT)]
        ) == RemediationStrategy.ACTIVE_LEARNING

    def test_low_data_no_action(self):
        assert self.svc.select_strategy(
            [self._drift(DriftSeverity.LOW, DriftType.DATA)]
        ) == RemediationStrategy.NO_ACTION

    def test_none_severity_no_action(self):
        result = DriftResult.from_test("f", "ks_test", DriftType.DATA, 0.01, 0.9)
        assert self.svc.select_strategy([result]) == RemediationStrategy.NO_ACTION

    def test_create_plan_rollback(self):
        plan = self.svc.create_remediation_plan(
            RemediationStrategy.ROLLBACK, "ep-1", [self._drift(DriftSeverity.CRITICAL)]
        )
        assert plan.strategy == RemediationStrategy.ROLLBACK
        assert plan.target_endpoint_id == "ep-1"
        assert len(plan.steps) > 0
        assert plan.estimated_recovery_seconds == 1

    def test_create_plan_incremental(self):
        plan = self.svc.create_remediation_plan(
            RemediationStrategy.INCREMENTAL_TRAINING, "ep-1", []
        )
        assert plan.estimated_recovery_seconds == 3600

    def test_create_plan_active_learning(self):
        plan = self.svc.create_remediation_plan(
            RemediationStrategy.ACTIVE_LEARNING, "ep-1", []
        )
        assert plan.estimated_recovery_seconds == 86400

    def test_create_plan_ensemble(self):
        plan = self.svc.create_remediation_plan(
            RemediationStrategy.ENSEMBLE_SWITCHING, "ep-1", []
        )
        assert plan.estimated_recovery_seconds == 60

    def test_create_plan_no_action(self):
        plan = self.svc.create_remediation_plan(
            RemediationStrategy.NO_ACTION, "ep-1", []
        )
        assert plan.steps == ()

    def test_mixed_severities(self):
        """When results have mixed severities, the highest severity wins."""
        results = [
            self._drift(DriftSeverity.LOW, DriftType.DATA),
            self._drift(DriftSeverity.CRITICAL, DriftType.DATA),
            self._drift(DriftSeverity.MEDIUM, DriftType.FEATURE),
        ]
        assert self.svc.select_strategy(results) == RemediationStrategy.ROLLBACK

    def test_all_non_drifted_results(self):
        """All results are non-drifted (high p_value) -- should be NO_ACTION."""
        r1 = DriftResult.from_test("f1", "ks_test", DriftType.DATA, 0.01, 0.9)
        r2 = DriftResult.from_test("f2", "ks_test", DriftType.FEATURE, 0.02, 0.8)
        assert not r1.is_drifted
        assert not r2.is_drifted
        assert self.svc.select_strategy([r1, r2]) == RemediationStrategy.NO_ACTION
