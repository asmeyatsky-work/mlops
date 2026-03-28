from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

class DriftType(Enum):
    DATA = "data_drift"
    PREDICTION = "prediction_skew"
    CONCEPT = "concept_drift"
    FEATURE = "feature_drift"

class DriftSeverity(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass(frozen=True)
class DriftResult:
    """Result of a statistical drift test on a single feature."""
    feature_name: str
    test_name: str
    drift_type: DriftType
    statistic: float
    p_value: float
    threshold: float
    severity: DriftSeverity

    @property
    def is_drifted(self) -> bool:
        return self.p_value < self.threshold

    @classmethod
    def from_test(
        cls,
        feature_name: str,
        test_name: str,
        drift_type: DriftType,
        statistic: float,
        p_value: float,
        threshold: float = 0.05,
    ) -> DriftResult:
        severity = _compute_severity(statistic)
        return cls(
            feature_name=feature_name,
            test_name=test_name,
            drift_type=drift_type,
            statistic=statistic,
            p_value=p_value,
            threshold=threshold,
            severity=severity,
        )

def _compute_severity(statistic: float) -> DriftSeverity:
    if statistic < 0.05:
        return DriftSeverity.NONE
    if statistic < 0.1:
        return DriftSeverity.LOW
    if statistic < 0.2:
        return DriftSeverity.MEDIUM
    if statistic < 0.3:
        return DriftSeverity.HIGH
    return DriftSeverity.CRITICAL
