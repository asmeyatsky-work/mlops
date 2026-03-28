from __future__ import annotations
from dataclasses import dataclass, field, replace
from datetime import datetime, UTC
from uuid import uuid4

from mlops_orchestrator.domain.events.event_base import DomainEvent
from mlops_orchestrator.domain.events.monitoring_events import (
    DriftDetectedEvent,
    MonitoringConfiguredEvent,
    RemediationTriggeredEvent,
)
from mlops_orchestrator.domain.value_objects.drift_result import DriftResult


@dataclass(frozen=True)
class MonitoringConfig:
    """
    Monitoring configuration aggregate root.

    Architectural Intent:
    - Configures Vertex Model Monitoring for drift/skew detection
    - Records drift events and triggers remediation
    """
    id: str
    endpoint_id: str
    drift_threshold: float = 0.05
    skew_threshold: float = 0.1
    enabled: bool = False
    drift_history: tuple[DriftResult, ...] = ()
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    domain_events: tuple[DomainEvent, ...] = ()

    @classmethod
    def create(cls, endpoint_id: str, drift_threshold: float = 0.05, skew_threshold: float = 0.1) -> MonitoringConfig:
        return cls(
            id=str(uuid4()),
            endpoint_id=endpoint_id,
            drift_threshold=drift_threshold,
            skew_threshold=skew_threshold,
        )

    def enable(self) -> MonitoringConfig:
        return replace(
            self,
            enabled=True,
            domain_events=self.domain_events + (
                MonitoringConfiguredEvent(
                    aggregate_id=self.id,
                    endpoint_id=self.endpoint_id,
                ),
            ),
        )

    def record_drift(self, drift_result: DriftResult) -> MonitoringConfig:
        new_config = replace(
            self,
            drift_history=self.drift_history + (drift_result,),
        )
        if drift_result.is_drifted:
            new_config = replace(
                new_config,
                domain_events=new_config.domain_events + (
                    DriftDetectedEvent(
                        aggregate_id=self.id,
                        drift_type=drift_result.drift_type.value,
                        severity=drift_result.severity.value,
                        metric_value=drift_result.statistic,
                    ),
                ),
            )
        return new_config

    def trigger_remediation(self, remediation_type: str, details: str) -> MonitoringConfig:
        return replace(
            self,
            domain_events=self.domain_events + (
                RemediationTriggeredEvent(
                    aggregate_id=self.id,
                    remediation_type=remediation_type,
                    details=details,
                ),
            ),
        )
