from __future__ import annotations
import asyncio
from typing import Any

from mlops_orchestrator.application.orchestration.dag_orchestrator import (
    DAGOrchestrator,
    OrchestrationError,
    WorkflowStep,
)
from mlops_orchestrator.application.session.session_state import SessionState
from mlops_orchestrator.domain.entities.training_job import TRAIN_IMAGE
from mlops_orchestrator.domain.ports.dataset_port import DatasetPort
from mlops_orchestrator.domain.ports.deployment_port import VertexDeploymentPort
from mlops_orchestrator.domain.ports.monitoring_port import MonitoringPort
from mlops_orchestrator.domain.ports.training_port import TrainingPort
from mlops_orchestrator.domain.value_objects.bq_source import BigQuerySource
from mlops_orchestrator.domain.value_objects.machine_spec import MachineSpec


class MLPipelineWorkflow:
    """
    End-to-end ML pipeline: data_ingest -> train -> deploy -> monitor.

    Parallelization Notes:
    - data_ingest has no dependencies
    - train depends on data_ingest (needs dataset_id)
    - deploy depends on train (needs model_resource_name)
    - monitor depends on deploy (needs endpoint_id)
    """

    def __init__(
        self,
        dataset_port: DatasetPort,
        training_port: TrainingPort,
        deployment_port: VertexDeploymentPort,
        monitoring_port: MonitoringPort,
    ) -> None:
        self._dataset_port = dataset_port
        self._training_port = training_port
        self._deployment_port = deployment_port
        self._monitoring_port = monitoring_port

    async def execute(
        self,
        bq_dataset: str,
        bq_table: str,
        model_name: str,
        endpoint_name: str,
        session: SessionState,
    ) -> dict[str, Any]:
        orchestrator = DAGOrchestrator([
            WorkflowStep("data_ingest", self._ingest_data),
            WorkflowStep("train", self._train_model, depends_on=("data_ingest",)),
            WorkflowStep("deploy", self._deploy_model, depends_on=("train",)),
            WorkflowStep("monitor", self._configure_monitoring, depends_on=("deploy",)),
        ])

        context: dict[str, Any] = {
            "bq_dataset": bq_dataset,
            "bq_table": bq_table,
            "model_name": model_name,
            "endpoint_name": endpoint_name,
            "session": session,
        }
        return await orchestrator.execute(context)

    async def _ingest_data(
        self, context: dict[str, Any], completed: dict[str, Any]
    ) -> str:
        bq_source = BigQuerySource(
            dataset=context["bq_dataset"], table=context["bq_table"]
        )
        resource_name = await self._dataset_port.create_dataset(
            bq_source=bq_source, display_name=context["model_name"] + "_dataset"
        )
        context["session"] = context["session"].add_dataset(resource_name)
        return resource_name

    async def _train_model(
        self, context: dict[str, Any], completed: dict[str, Any]
    ) -> str:
        dataset_id = completed["data_ingest"]
        job_resource = await self._training_port.start_training(
            model_name=context["model_name"],
            dataset_id=dataset_id,
            gcs_uri="",
            train_image=TRAIN_IMAGE,
        )
        # Poll until training completes
        while True:
            status = await self._training_port.get_job_status(job_resource)
            if status in {"SUCCEEDED", "FAILED", "CANCELLED"}:
                break
            await asyncio.sleep(5)
        if status != "SUCCEEDED":
            raise OrchestrationError(f"Training failed with status: {status}")
        context["session"] = context["session"].add_job_handle(job_resource)
        return job_resource

    async def _deploy_model(
        self, context: dict[str, Any], completed: dict[str, Any]
    ) -> str:
        job_resource = completed["train"]
        model_resource = await self._training_port.get_model_resource_name(job_resource)
        endpoint_resource = await self._deployment_port.create_endpoint_and_deploy(
            model_id=model_resource,
            endpoint_name=context["endpoint_name"],
            machine_spec=MachineSpec(machine_type="n1-standard-4"),
        )
        context["session"] = context["session"].add_endpoint(endpoint_resource)
        return endpoint_resource

    async def _configure_monitoring(
        self, context: dict[str, Any], completed: dict[str, Any]
    ) -> bool:
        endpoint_resource = completed["deploy"]
        return await self._monitoring_port.configure_monitoring(
            endpoint_id=endpoint_resource,
            drift_threshold=0.05,
            skew_threshold=0.1,
        )
