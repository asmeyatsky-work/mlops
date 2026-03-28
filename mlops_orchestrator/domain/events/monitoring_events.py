from __future__ import annotations
from dataclasses import dataclass
from mlops_orchestrator.domain.events.event_base import DomainEvent

@dataclass(frozen=True)
class DriftDetectedEvent(DomainEvent):
    """Emitted when statistical drift is detected."""
    drift_type: str = ""
    severity: str = ""
    metric_value: float = 0.0

@dataclass(frozen=True)
class MonitoringConfiguredEvent(DomainEvent):
    """Emitted when monitoring is set up for an endpoint."""
    endpoint_id: str = ""

@dataclass(frozen=True)
class RemediationTriggeredEvent(DomainEvent):
    """Emitted when an automated remediation action is taken."""
    remediation_type: str = ""
    details: str = ""
