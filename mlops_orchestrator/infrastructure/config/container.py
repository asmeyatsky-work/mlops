from __future__ import annotations

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

        self._dataset_port = StubDatasetAdapter()
        self._training_port = StubTrainingAdapter()
        self._vertex_deployment_port = StubVertexDeploymentAdapter()
        self._gke_deployment_port = StubGkeDeploymentAdapter()
        self._monitoring_port = StubMonitoringAdapter()
        self._event_bus = InMemoryEventBus()
        self._audit_log = StubAuditLogAdapter()
        self._cost_port = StubCostAdapter()
        self._security_port = StubSecurityAdapter()

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

        project = self._settings.gcp_project
        location = self._settings.gcp_location

        self._dataset_port = VertexDatasetAdapter(project=project, location=location)
        self._training_port = VertexTrainingAdapter(project=project, location=location)
        self._vertex_deployment_port = VertexEndpointAdapter(project=project, location=location)
        self._gke_deployment_port = GkeDeploymentAdapter()
        self._monitoring_port = VertexMonitoringAdapter(project=project, location=location)
        self._event_bus = InMemoryEventBus()
        self._audit_log = CloudLoggingAuditAdapter(project=project)
        self._cost_port = StubCostAdapter()  # Real cost adapter can be added later
        self._security_port = GcpSecurityAdapter()

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
        )

    def deploy_gke_command(self) -> DeployToGkeCommand:
        return DeployToGkeCommand(
            deployment_port=self._gke_deployment_port,
            event_bus=self._event_bus,
            audit_log=self._audit_log,
        )

    def configure_monitoring_command(self) -> ConfigureMonitoringCommand:
        return ConfigureMonitoringCommand(
            monitoring_port=self._monitoring_port,
            event_bus=self._event_bus,
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
    def settings(self) -> Settings:
        return self._settings
