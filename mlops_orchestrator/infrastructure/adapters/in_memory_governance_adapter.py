"""In-memory ModelGovernancePort adapter.

Stub backend for the compliance gate: production environments would replace
this with a registry that persists governance records (e.g. backed by the
Vertex Model Registry's metadata, BigQuery, or a dedicated governance DB).
"""
from __future__ import annotations

from mlops_orchestrator.domain.value_objects.compliance import ModelCard, RiskClassification


class InMemoryGovernanceAdapter:
    """Thread-safe-ish (async single-loop) governance store for tests and dev."""

    def __init__(self) -> None:
        self._risks: dict[str, RiskClassification] = {}
        self._cards: dict[str, ModelCard] = {}

    async def get_risk_classification(self, model_id: str) -> RiskClassification | None:
        return self._risks.get(model_id)

    async def get_model_card(self, model_id: str) -> ModelCard | None:
        return self._cards.get(model_id)

    # Test / admin helpers — not part of the port.
    def record_risk(self, model_id: str, risk: RiskClassification) -> None:
        self._risks[model_id] = risk

    def record_card(self, model_id: str, card: ModelCard) -> None:
        self._cards[model_id] = card
