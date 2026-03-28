from __future__ import annotations
import logging
from typing import Any

from mlops_orchestrator.application.orchestration.dag_orchestrator import (
    DAGOrchestrator,
    WorkflowStep,
)
from mlops_orchestrator.domain.ports.deployment_port import VertexDeploymentPort
from mlops_orchestrator.domain.ports.monitoring_port import MonitoringPort
from mlops_orchestrator.domain.ports.training_port import TrainingPort
from mlops_orchestrator.domain.services.drift_detection_service import (
    DriftDetectionService,
)
from mlops_orchestrator.domain.services.remediation_service import (
    RemediationPlan,
    RemediationService,
    RemediationStrategy,
)
from mlops_orchestrator.domain.value_objects.drift_result import DriftResult, DriftType

logger = logging.getLogger(__name__)


class SelfHealingWorkflow:
    """
    Self-healing closed-loop: Observe -> Analyze -> Decide -> Act.

    Monitors deployed models for drift and triggers automated remediation.
    """

    def __init__(
        self,
        monitoring_port: MonitoringPort,
        training_port: TrainingPort,
        deployment_port: VertexDeploymentPort | None = None,
    ) -> None:
        self._monitoring_port = monitoring_port
        self._training_port = training_port
        self._deployment_port = deployment_port
        self._drift_service = DriftDetectionService()
        self._remediation_service = RemediationService()

    async def execute(self, endpoint_id: str) -> dict[str, Any]:
        orchestrator = DAGOrchestrator([
            WorkflowStep("observe", self._observe),
            WorkflowStep("analyze", self._analyze, depends_on=("observe",)),
            WorkflowStep("decide", self._decide, depends_on=("analyze",)),
            WorkflowStep("act", self._act, depends_on=("decide",)),
        ])
        context = {"endpoint_id": endpoint_id}
        return await orchestrator.execute(context)

    async def _observe(
        self, context: dict[str, Any], completed: dict[str, Any]
    ) -> list[dict[str, float]]:
        alerts = await self._monitoring_port.get_drift_alerts(context["endpoint_id"])
        logger.info("Observed %d drift alerts for endpoint %s", len(alerts), context["endpoint_id"])
        return alerts

    async def _analyze(
        self, context: dict[str, Any], completed: dict[str, Any]
    ) -> list[DriftResult]:
        alerts = completed["observe"]
        results: list[DriftResult] = []
        for alert in alerts:
            result = DriftResult.from_test(
                feature_name=str(alert.get("feature", "unknown")),
                test_name="monitoring_alert",
                drift_type=DriftType.DATA,
                statistic=alert.get("statistic", 0.0),
                p_value=alert.get("p_value", 1.0),
                threshold=alert.get("threshold", 0.05),
            )
            results.append(result)
        drifted_count = sum(1 for r in results if r.is_drifted)
        logger.info("Analyzed %d alerts: %d drifted", len(results), drifted_count)
        return results

    async def _decide(
        self, context: dict[str, Any], completed: dict[str, Any]
    ) -> RemediationPlan:
        drift_results = completed["analyze"]
        strategy = self._remediation_service.select_strategy(drift_results)
        plan = self._remediation_service.create_remediation_plan(
            strategy=strategy,
            endpoint_id=context["endpoint_id"],
            drift_results=drift_results,
        )
        logger.info("Decided strategy: %s for endpoint %s", strategy.value, context["endpoint_id"])
        return plan

    async def _act(
        self, context: dict[str, Any], completed: dict[str, Any]
    ) -> dict[str, str]:
        plan: RemediationPlan = completed["decide"]
        endpoint_id = context["endpoint_id"]

        if plan.strategy == RemediationStrategy.NO_ACTION:
            logger.info("No action required for endpoint %s", endpoint_id)
            return {"action": "none", "details": plan.details}

        if plan.strategy == RemediationStrategy.ROLLBACK:
            logger.warning("Executing ROLLBACK for endpoint %s", endpoint_id)
            if self._deployment_port:
                await self._deployment_port.undeploy(endpoint_id)
            return {"action": "rollback", "details": plan.details, "executed": "true"}

        if plan.strategy == RemediationStrategy.INCREMENTAL_TRAINING:
            logger.info("Triggering incremental training for endpoint %s", endpoint_id)
            job_rn = await self._training_port.start_training(
                model_name=f"retrain-{endpoint_id.split('/')[-1]}",
                dataset_id="",
                gcs_uri="",
                train_image="us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-12:latest",
            )
            return {"action": "incremental_training", "details": plan.details, "job_resource_name": job_rn}

        if plan.strategy == RemediationStrategy.ACTIVE_LEARNING:
            logger.info("Active learning triggered for endpoint %s", endpoint_id)
            return {"action": "active_learning", "details": plan.details}

        if plan.strategy == RemediationStrategy.ENSEMBLE_SWITCHING:
            logger.info("Ensemble switching for endpoint %s", endpoint_id)
            return {"action": "ensemble_switching", "details": plan.details}

        return {"action": "unknown", "details": plan.details}
