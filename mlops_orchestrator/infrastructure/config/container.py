from __future__ import annotations

from mlops_orchestrator.application.commands.batch_prediction_command import (
    BatchPredictionCommand,
)
from mlops_orchestrator.application.commands.configure_monitoring_command import (
    ConfigureMonitoringCommand,
)
from mlops_orchestrator.application.commands.create_dataset_command import (
    CreateDatasetCommand,
)
from mlops_orchestrator.application.commands.deploy_gke_command import (
    DeployToGkeCommand,
)
from mlops_orchestrator.application.commands.deploy_vertex_command import (
    DeployToVertexCommand,
)
from mlops_orchestrator.application.commands.model_registry_command import (
    ModelRegistryCommand,
)
from mlops_orchestrator.application.commands.train_model_command import (
    TrainModelCommand,
)
from mlops_orchestrator.application.queries.cost_query import CostQuery
from mlops_orchestrator.application.queries.job_status_query import JobStatusQuery
from mlops_orchestrator.infrastructure.config.settings import Settings


class DependencyContainer:
    """
    Composition root — the ONLY place where concrete adapter classes are imported.

    Wires all domain ports to infrastructure adapters based on settings.use_stubs.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        if self._settings.use_stubs:
            self._build_stub_adapters()
        else:
            self._build_gcp_adapters()
        self._build_alerting()

    def _build_stub_adapters(self) -> None:
        from mlops_orchestrator.infrastructure.adapters.stub_dataset_adapter import (
            StubDatasetAdapter,
        )
        from mlops_orchestrator.infrastructure.adapters.stub_training_adapter import (
            StubTrainingAdapter,
        )
        from mlops_orchestrator.infrastructure.adapters.stub_deployment_adapter import (
            StubGkeDeploymentAdapter,
            StubVertexDeploymentAdapter,
        )
        from mlops_orchestrator.infrastructure.adapters.stub_monitoring_adapter import (
            StubMonitoringAdapter,
        )
        from mlops_orchestrator.infrastructure.adapters.stub_infrastructure_adapters import (
            InMemoryEventBus,
            StubAuditLogAdapter,
            StubCostAdapter,
            StubSecurityAdapter,
        )
        from mlops_orchestrator.infrastructure.adapters.stub_batch_prediction_adapter import (
            StubBatchPredictionAdapter,
        )
        from mlops_orchestrator.infrastructure.adapters.stub_model_registry_adapter import (
            StubModelRegistryAdapter,
        )

        self._dataset_port = StubDatasetAdapter()
        self._training_port = StubTrainingAdapter()
        self._vertex_deployment_port = StubVertexDeploymentAdapter()
        self._gke_deployment_port = StubGkeDeploymentAdapter()
        self._monitoring_port = StubMonitoringAdapter()
        self._event_bus = InMemoryEventBus()
        self._audit_log = StubAuditLogAdapter()
        self._cost_port = StubCostAdapter()
        self._security_port = StubSecurityAdapter()
        self._batch_prediction_port = StubBatchPredictionAdapter()
        self._model_registry_port = StubModelRegistryAdapter()
        from mlops_orchestrator.infrastructure.adapters.in_memory_governance_adapter import (
            InMemoryGovernanceAdapter,
        )
        self._governance_port = InMemoryGovernanceAdapter()

    def _build_gcp_adapters(self) -> None:
        from mlops_orchestrator.infrastructure.adapters.vertex_dataset_adapter import (
            VertexDatasetAdapter,
        )
        from mlops_orchestrator.infrastructure.adapters.vertex_training_adapter import (
            VertexTrainingAdapter,
        )
        from mlops_orchestrator.infrastructure.adapters.vertex_deployment_adapter import (
            VertexEndpointAdapter,
        )
        from mlops_orchestrator.infrastructure.adapters.gke_deployment_adapter import (
            GkeDeploymentAdapter,
        )
        from mlops_orchestrator.infrastructure.adapters.vertex_monitoring_adapter import (
            VertexMonitoringAdapter,
        )
        from mlops_orchestrator.infrastructure.adapters.cloud_logging_adapter import (
            CloudLoggingAuditAdapter,
        )
        from mlops_orchestrator.infrastructure.adapters.security_adapter import (
            GcpSecurityAdapter,
        )
        from mlops_orchestrator.infrastructure.adapters.stub_infrastructure_adapters import (
            InMemoryEventBus,
            StubCostAdapter,
        )
        from mlops_orchestrator.infrastructure.adapters.vertex_batch_prediction_adapter import (
            VertexBatchPredictionAdapter,
        )
        from mlops_orchestrator.infrastructure.adapters.vertex_model_registry_adapter import (
            VertexModelRegistryAdapter,
        )

        project = self._settings.gcp_project
        location = self._settings.gcp_location

        self._dataset_port = VertexDatasetAdapter(project=project, location=location)
        self._training_port = VertexTrainingAdapter(project=project, location=location)
        self._vertex_deployment_port = VertexEndpointAdapter(project=project, location=location)
        self._gke_deployment_port = GkeDeploymentAdapter()
        self._monitoring_port = VertexMonitoringAdapter(project=project, location=location)
        self._event_bus = InMemoryEventBus()
        self._audit_log = CloudLoggingAuditAdapter(project=project)
        self._security_port = GcpSecurityAdapter(project=project)
        self._batch_prediction_port = VertexBatchPredictionAdapter(project=project, location=location)
        self._model_registry_port = VertexModelRegistryAdapter(project=project, location=location)
        from mlops_orchestrator.infrastructure.adapters.in_memory_governance_adapter import (
            InMemoryGovernanceAdapter,
        )
        # TODO: swap for a persistent governance backend (BigQuery / dedicated DB)
        # once the production registry is provisioned. The in-memory adapter
        # still enforces the gate when records are populated at startup.
        self._governance_port = InMemoryGovernanceAdapter()

        # Use real cost adapter if billing table is configured
        if self._settings.billing_table:
            from mlops_orchestrator.infrastructure.adapters.billing_cost_adapter import (
                BigQueryCostAdapter,
            )
            self._cost_port = BigQueryCostAdapter(
                project=project, billing_table=self._settings.billing_table
            )
        else:
            self._cost_port = StubCostAdapter()

    def _build_alerting(self) -> None:
        """Build alerting adapters based on configured channels."""
        from mlops_orchestrator.infrastructure.adapters.alerting_adapters import (
            CompositeAlertAdapter,
            StubAlertAdapter,
        )

        adapters: list = []

        if self._settings.slack_webhook_url:
            from mlops_orchestrator.infrastructure.adapters.alerting_adapters import (
                SlackAlertAdapter,
            )
            adapters.append(SlackAlertAdapter(self._settings.slack_webhook_url))

        if self._settings.pagerduty_routing_key:
            from mlops_orchestrator.infrastructure.adapters.alerting_adapters import (
                PagerDutyAlertAdapter,
            )
            adapters.append(PagerDutyAlertAdapter(self._settings.pagerduty_routing_key))

        if self._settings.alert_email_smtp_host and self._settings.alert_email_recipients:
            from mlops_orchestrator.infrastructure.adapters.alerting_adapters import (
                EmailAlertAdapter,
            )
            adapters.append(
                EmailAlertAdapter(
                    smtp_host=self._settings.alert_email_smtp_host,
                    smtp_port=self._settings.alert_email_smtp_port,
                    sender=self._settings.alert_email_sender,
                    recipients=self._settings.alert_email_recipients.split(","),
                    username=self._settings.alert_email_username,
                    password=self._settings.alert_email_password,
                )
            )

        if adapters:
            self._alerting_port = CompositeAlertAdapter(adapters)
        elif self._settings.use_stubs:
            self._alerting_port = StubAlertAdapter()
        else:
            self._alerting_port = None  # type: ignore[assignment]

    # ─── Command factories ───

    def create_dataset_command(self) -> CreateDatasetCommand:
        return CreateDatasetCommand(
            dataset_port=self._dataset_port,
            event_bus=self._event_bus,
            audit_log=self._audit_log,
        )

    def train_model_command(self) -> TrainModelCommand:
        return TrainModelCommand(
            training_port=self._training_port,
            event_bus=self._event_bus,
            audit_log=self._audit_log,
        )

    def deploy_vertex_command(self) -> DeployToVertexCommand:
        return DeployToVertexCommand(
            deployment_port=self._vertex_deployment_port,
            event_bus=self._event_bus,
            audit_log=self._audit_log,
            governance_port=self._governance_port if self._settings.compliance_strict else None,
        )

    def deploy_gke_command(self) -> DeployToGkeCommand:
        return DeployToGkeCommand(
            deployment_port=self._gke_deployment_port,
            event_bus=self._event_bus,
            audit_log=self._audit_log,
            governance_port=self._governance_port if self._settings.compliance_strict else None,
        )

    def configure_monitoring_command(self) -> ConfigureMonitoringCommand:
        return ConfigureMonitoringCommand(
            monitoring_port=self._monitoring_port,
            event_bus=self._event_bus,
            audit_log=self._audit_log,
        )

    def batch_prediction_command(self) -> BatchPredictionCommand:
        return BatchPredictionCommand(
            batch_port=self._batch_prediction_port,
            event_bus=self._event_bus,
            audit_log=self._audit_log,
        )

    def model_registry_command(self) -> ModelRegistryCommand:
        return ModelRegistryCommand(
            registry_port=self._model_registry_port,
            audit_log=self._audit_log,
        )

    # ─── Query factories ───

    def job_status_query(self) -> JobStatusQuery:
        return JobStatusQuery(training_port=self._training_port)

    def cost_query(self) -> CostQuery:
        return CostQuery(cost_port=self._cost_port)

    # ─── Direct port access (for workflows/testing) ───

    @property
    def dataset_port(self):
        return self._dataset_port

    @property
    def training_port(self):
        return self._training_port

    @property
    def vertex_deployment_port(self):
        return self._vertex_deployment_port

    @property
    def gke_deployment_port(self):
        return self._gke_deployment_port

    @property
    def monitoring_port(self):
        return self._monitoring_port

    @property
    def event_bus(self):
        return self._event_bus

    @property
    def audit_log(self):
        return self._audit_log

    @property
    def alerting_port(self):
        return self._alerting_port

    @property
    def batch_prediction_port(self):
        return self._batch_prediction_port

    @property
    def model_registry_port(self):
        return self._model_registry_port

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def governance_port(self):
        return self._governance_port
