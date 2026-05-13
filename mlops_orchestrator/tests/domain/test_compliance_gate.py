"""Tests for the EU AI Act compliance gate."""
from __future__ import annotations

import pytest

from mlops_orchestrator.domain.services.compliance_service import (
    ComplianceGateError,
    ComplianceGateService,
)
from mlops_orchestrator.domain.value_objects.compliance import (
    ModelCard,
    RiskClassification,
    RiskTier,
)


@pytest.fixture
def gate() -> ComplianceGateService:
    return ComplianceGateService()


@pytest.fixture
def complete_card() -> ModelCard:
    return ModelCard(
        model_name="m",
        version="1",
        purpose="recommend products",
        limitations="english only",
        data_sources=("bq://prod/transactions",),
        accuracy_metrics=(("auc", 0.91),),
    )


class TestComplianceGate:
    def test_missing_risk_blocks(self, gate, complete_card):
        with pytest.raises(ComplianceGateError, match="risk classification"):
            gate.enforce(None, complete_card)

    def test_prohibited_always_blocks(self, gate, complete_card):
        risk = RiskClassification(
            tier=RiskTier.PROHIBITED, domain="x", justification="social scoring",
        )
        with pytest.raises(ComplianceGateError, match="prohibited"):
            gate.enforce(risk, complete_card)

    def test_high_risk_requires_complete_card(self, gate):
        risk = RiskClassification(
            tier=RiskTier.HIGH,
            domain="healthcare",
            justification="patient outcomes",
            required_controls=("human_oversight",),
        )
        with pytest.raises(ComplianceGateError, match="model card"):
            gate.enforce(risk, None)

    def test_high_risk_requires_required_controls(self, gate, complete_card):
        risk = RiskClassification(
            tier=RiskTier.HIGH, domain="finance", justification="credit",
        )
        with pytest.raises(ComplianceGateError, match="required controls"):
            gate.enforce(risk, complete_card)

    def test_high_risk_passes_when_complete(self, gate, complete_card):
        risk = RiskClassification(
            tier=RiskTier.HIGH,
            domain="healthcare",
            justification="x",
            required_controls=("human_oversight", "data_governance"),
        )
        gate.enforce(risk, complete_card)

    def test_minimal_risk_passes_without_card(self, gate):
        risk = RiskClassification(
            tier=RiskTier.MINIMAL, domain="other", justification="x",
        )
        gate.enforce(risk, None)
