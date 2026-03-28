from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum

from mlops_orchestrator.domain.value_objects.drift_result import (
    DriftResult,
    DriftSeverity,
    DriftType,
)


class RemediationStrategy(Enum):
    ROLLBACK = "automated_rollback"
    INCREMENTAL_TRAINING = "incremental_training"
    ACTIVE_LEARNING = "active_learning"
    ENSEMBLE_SWITCHING = "ensemble_switching"
    NO_ACTION = "no_action"


@dataclass(frozen=True)
class RemediationPlan:
    """Plan for automated remediation of drift."""
    strategy: RemediationStrategy
    target_endpoint_id: str
    details: str
    steps: tuple[str, ...] = ()
    estimated_recovery_seconds: int = 0


class RemediationService:
    """
    Selects remediation strategies based on drift analysis.
    Closed-loop: Observe -> Analyze -> Decide -> Act.
    Pure domain logic.
    """

    def select_strategy(self, drift_results: list[DriftResult]) -> RemediationStrategy:
        """Select the best remediation strategy based on drift severity and type."""
        if not drift_results:
            return RemediationStrategy.NO_ACTION

        max_severity = max(
            (r.severity for r in drift_results if r.is_drifted),
            default=DriftSeverity.NONE,
            key=lambda s: list(DriftSeverity).index(s),
        )

        if max_severity == DriftSeverity.CRITICAL:
            return RemediationStrategy.ROLLBACK
        if max_severity == DriftSeverity.HIGH:
            has_data_drift = any(
                r.drift_type == DriftType.DATA and r.is_drifted for r in drift_results
            )
            return (
                RemediationStrategy.INCREMENTAL_TRAINING
                if has_data_drift
                else RemediationStrategy.ENSEMBLE_SWITCHING
            )
        if max_severity == DriftSeverity.MEDIUM:
            return RemediationStrategy.ENSEMBLE_SWITCHING
        if max_severity == DriftSeverity.LOW:
            has_concept_drift = any(
                r.drift_type == DriftType.CONCEPT and r.is_drifted for r in drift_results
            )
            return (
                RemediationStrategy.ACTIVE_LEARNING
                if has_concept_drift
                else RemediationStrategy.NO_ACTION
            )
        return RemediationStrategy.NO_ACTION

    def create_remediation_plan(
        self,
        strategy: RemediationStrategy,
        endpoint_id: str,
        drift_results: list[DriftResult],
    ) -> RemediationPlan:
        """Create a detailed remediation plan for the selected strategy."""
        builders = {
            RemediationStrategy.ROLLBACK: self._plan_rollback,
            RemediationStrategy.INCREMENTAL_TRAINING: self._plan_incremental,
            RemediationStrategy.ACTIVE_LEARNING: self._plan_active_learning,
            RemediationStrategy.ENSEMBLE_SWITCHING: self._plan_ensemble,
            RemediationStrategy.NO_ACTION: self._plan_no_action,
        }
        return builders[strategy](endpoint_id, drift_results)

    def _plan_rollback(self, endpoint_id: str, results: list[DriftResult]) -> RemediationPlan:
        drifted = [r.feature_name for r in results if r.is_drifted]
        return RemediationPlan(
            strategy=RemediationStrategy.ROLLBACK,
            target_endpoint_id=endpoint_id,
            details=f"Critical drift in features: {', '.join(drifted)}. Rolling back to last stable model.",
            steps=(
                "identify_last_stable_model",
                "swap_model_on_endpoint",
                "verify_predictions",
                "notify_stakeholders",
            ),
            estimated_recovery_seconds=1,
        )

    def _plan_incremental(self, endpoint_id: str, results: list[DriftResult]) -> RemediationPlan:
        return RemediationPlan(
            strategy=RemediationStrategy.INCREMENTAL_TRAINING,
            target_endpoint_id=endpoint_id,
            details="Data drift detected. Triggering incremental training with recent data.",
            steps=(
                "collect_recent_labeled_data",
                "update_training_dataset",
                "run_incremental_training_job",
                "validate_new_model",
                "deploy_if_improved",
            ),
            estimated_recovery_seconds=3600,
        )

    def _plan_active_learning(self, endpoint_id: str, results: list[DriftResult]) -> RemediationPlan:
        return RemediationPlan(
            strategy=RemediationStrategy.ACTIVE_LEARNING,
            target_endpoint_id=endpoint_id,
            details="Concept drift detected. Selecting uncertain samples for expert labeling.",
            steps=(
                "identify_uncertain_predictions",
                "queue_samples_for_labeling",
                "wait_for_expert_labels",
                "retrain_with_new_labels",
                "deploy_updated_model",
            ),
            estimated_recovery_seconds=86400,
        )

    def _plan_ensemble(self, endpoint_id: str, results: list[DriftResult]) -> RemediationPlan:
        return RemediationPlan(
            strategy=RemediationStrategy.ENSEMBLE_SWITCHING,
            target_endpoint_id=endpoint_id,
            details="Moderate drift. Switching to challenger model in ensemble.",
            steps=(
                "evaluate_challenger_models",
                "select_best_performer",
                "update_traffic_routing",
                "monitor_new_performance",
            ),
            estimated_recovery_seconds=60,
        )

    def _plan_no_action(self, endpoint_id: str, results: list[DriftResult]) -> RemediationPlan:
        return RemediationPlan(
            strategy=RemediationStrategy.NO_ACTION,
            target_endpoint_id=endpoint_id,
            details="Drift within acceptable bounds. No action required.",
        )
