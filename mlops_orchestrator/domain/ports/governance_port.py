"""Port for retrieving model governance records (risk classification, model card)."""
from __future__ import annotations

from typing import Protocol

from mlops_orchestrator.domain.value_objects.compliance import ModelCard, RiskClassification


class ModelGovernancePort(Protocol):
    """Lookup interface for governance records keyed by model id.

    Returning ``None`` indicates the artifact was never registered — the
    compliance gate treats that as a blocking reason for HIGH-risk inputs.
    """

    async def get_risk_classification(
        self, model_id: str
    ) -> RiskClassification | None: ...

    async def get_model_card(self, model_id: str) -> ModelCard | None: ...
