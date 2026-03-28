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
        severity = _compute_severity(statistic, test_name)
        return cls(
            feature_name=feature_name,
            test_name=test_name,
            drift_type=drift_type,
            statistic=statistic,
            p_value=p_value,
            threshold=threshold,
            severity=severity,
        )

# Test-specific severity thresholds: each maps statistic ranges to severity.
_KS_THRESHOLDS = (0.05, 0.1, 0.2, 0.3)
_CHI2_THRESHOLDS = (3.84, 7.81, 15.51, 30.0)
_PSI_THRESHOLDS = (0.1, 0.2, 0.3, 0.5)
_KL_THRESHOLDS = (0.05, 0.1, 0.3, 0.5)

_THRESHOLD_MAP: dict[str, tuple[float, ...]] = {
    "ks_test": _KS_THRESHOLDS,
    "chi_square": _CHI2_THRESHOLDS,
    "psi": _PSI_THRESHOLDS,
    "kl_divergence": _KL_THRESHOLDS,
}

def _compute_severity(statistic: float, test_name: str = "ks_test") -> DriftSeverity:
    thresholds = _THRESHOLD_MAP.get(test_name, _KS_THRESHOLDS)
    if statistic < thresholds[0]:
        return DriftSeverity.NONE
    if statistic < thresholds[1]:
        return DriftSeverity.LOW
    if statistic < thresholds[2]:
        return DriftSeverity.MEDIUM
    if statistic < thresholds[3]:
        return DriftSeverity.HIGH
    return DriftSeverity.CRITICAL
