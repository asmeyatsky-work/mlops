"""Tests for reasoning-agent value objects (Plan / Recommendation)."""
from __future__ import annotations

import pytest

from mlops_orchestrator.domain.value_objects.plan import (
    Plan,
    PlannedTask,
    Recommendation,
    RecommendationKind,
)


class TestPlannedTask:
    def test_valid(self):
        t = PlannedTask(description="ingest data", capability="etl")
        assert t.capability == "etl"
        assert t.id  # auto-assigned

    def test_unique_ids(self):
        a = PlannedTask(description="x", capability="etl")
        b = PlannedTask(description="x", capability="etl")
        assert a.id != b.id

    @pytest.mark.parametrize("desc,cap", [("", "etl"), ("  ", "etl"), ("x", ""), ("x", " ")])
    def test_rejects_blank_fields(self, desc, cap):
        with pytest.raises(ValueError):
            PlannedTask(description=desc, capability=cap)


class TestPlan:
    def test_valid(self):
        plan = Plan(goal="train a model", tasks=(PlannedTask(description="x", capability="etl"),))
        assert plan.goal == "train a model"

    def test_rejects_empty_goal(self):
        with pytest.raises(ValueError):
            Plan(goal="  ", tasks=(PlannedTask(description="x", capability="etl"),))

    def test_rejects_no_tasks(self):
        with pytest.raises(ValueError):
            Plan(goal="train", tasks=())


class TestRecommendation:
    def test_valid(self):
        r = Recommendation(kind=RecommendationKind.COST, summary="use spot", confidence=0.8)
        assert r.kind is RecommendationKind.COST

    def test_rejects_blank_summary(self):
        with pytest.raises(ValueError):
            Recommendation(kind=RecommendationKind.GENERIC, summary="")

    @pytest.mark.parametrize("conf", [-0.1, 1.1, 2.0])
    def test_rejects_out_of_range_confidence(self, conf):
        with pytest.raises(ValueError):
            Recommendation(kind=RecommendationKind.GENERIC, summary="ok", confidence=conf)

    def test_immutable(self):
        r = Recommendation(kind=RecommendationKind.GENERIC, summary="ok")
        with pytest.raises(Exception):
            r.summary = "changed"  # type: ignore[misc]
