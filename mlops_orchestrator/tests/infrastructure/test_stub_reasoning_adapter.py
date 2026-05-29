"""Tests for the deterministic StubReasoningAdapter."""
from __future__ import annotations

import pytest

from mlops_orchestrator.domain.entities.agent import AgentRole
from mlops_orchestrator.domain.value_objects.plan import (
    Plan,
    Recommendation,
    RecommendationKind,
)
from mlops_orchestrator.infrastructure.adapters.stub_reasoning_adapter import (
    StubReasoningAdapter,
)


class TestStubReasoningAdapter:
    async def test_decompose_one_task_per_capability(self):
        adapter = StubReasoningAdapter()
        plan = await adapter.decompose("ship a model", ("etl", "training", "deploy"))
        assert isinstance(plan, Plan)
        assert len(plan.tasks) == 3
        assert {t.capability for t in plan.tasks} == {"etl", "training", "deploy"}

    async def test_decompose_defaults_when_no_capabilities(self):
        adapter = StubReasoningAdapter()
        plan = await adapter.decompose("goal", ())
        assert len(plan.tasks) == 1
        assert plan.tasks[0].capability == "coordination"

    @pytest.mark.parametrize(
        "role,kind",
        [
            (AgentRole.ARCHITECT, RecommendationKind.INFRASTRUCTURE),
            (AgentRole.FINOPS, RecommendationKind.COST),
            (AgentRole.VALIDATION, RecommendationKind.VALIDATION),
            (AgentRole.DATA_ENGINEER, RecommendationKind.DATA),
            (AgentRole.ORCHESTRATOR, RecommendationKind.GENERIC),
            (AgentRole.SECURITY, RecommendationKind.GENERIC),
        ],
    )
    async def test_advise_maps_role_to_kind(self, role, kind):
        adapter = StubReasoningAdapter()
        rec = await adapter.advise(role, "review this")
        assert isinstance(rec, Recommendation)
        assert rec.kind is kind
        assert "review this" in rec.summary
        assert 0.0 <= rec.confidence <= 1.0
