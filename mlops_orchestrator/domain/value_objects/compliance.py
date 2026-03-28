from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum

class RiskTier(Enum):
    PROHIBITED = "prohibited"
    HIGH = "high_risk"
    LIMITED = "limited_risk"
    MINIMAL = "minimal_risk"

@dataclass(frozen=True)
class RiskClassification:
    """EU AI Act risk classification for a model."""
    tier: RiskTier
    domain: str
    justification: str
    required_controls: tuple[str, ...] = ()
    assessed_at: datetime = field(default_factory=lambda: datetime.now(UTC))

@dataclass(frozen=True)
class ModelCard:
    """Model Card per EU AI Act Article 11."""
    model_name: str
    version: str
    purpose: str
    limitations: str
    data_sources: tuple[str, ...] = ()
    accuracy_metrics: tuple[tuple[str, float], ...] = ()
    fairness_assessment: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_complete(self) -> bool:
        return bool(
            self.purpose
            and self.limitations
            and self.data_sources
            and self.accuracy_metrics
        )
