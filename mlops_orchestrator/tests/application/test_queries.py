"""Tests for application queries."""
from __future__ import annotations

import pytest

from mlops_orchestrator.application.queries.job_status_query import JobStatusQuery
from mlops_orchestrator.application.queries.cost_query import CostQuery
from mlops_orchestrator.infrastructure.adapters.stub_training_adapter import StubTrainingAdapter
from mlops_orchestrator.infrastructure.adapters.stub_infrastructure_adapters import (
    StubCostAdapter,
)
from mlops_orchestrator.domain.value_objects.cost_metrics import CostMetrics, CostRecommendation


class TestJobStatusQuery:
    async def test_execute(self):
        adapter = StubTrainingAdapter(auto_succeed=True)
        job_rn = await adapter.start_training("m", "ds", "", "img")
        query = JobStatusQuery(adapter)
        result = await query.execute(job_rn)
        assert result["status"] == "SUCCEEDED"
        assert result["job_resource_name"] == job_rn

    async def test_poll_succeeds_immediately(self):
        adapter = StubTrainingAdapter(auto_succeed=True)
        job_rn = await adapter.start_training("m", "ds", "", "img")
        query = JobStatusQuery(adapter)
        result = await query.poll_until_complete(job_rn, interval_seconds=0, timeout_seconds=1)
        assert result["status"] == "SUCCEEDED"
        assert "model_resource_name" in result

    async def test_execute_unknown_job(self):
        adapter = StubTrainingAdapter()
        query = JobStatusQuery(adapter)
        result = await query.execute("nonexistent")
        assert result["status"] == "UNKNOWN"


class TestCostQuery:
    async def test_get_project_metrics(self):
        metrics = CostMetrics(cost_per_tb_scanned=5.0, gpu_idle_pct=40.0)
        adapter = StubCostAdapter(metrics=metrics)
        query = CostQuery(adapter)
        result = await query.get_project_metrics("proj")
        assert result["cost_per_tb_scanned"] == 5.0
        assert result["gpu_idle_pct"] == 40.0

    async def test_get_recommendations(self):
        recs = [CostRecommendation("resize", "Downsize VM", 50.0, "high")]
        adapter = StubCostAdapter(recommendations=recs)
        query = CostQuery(adapter)
        result = await query.get_recommendations("proj")
        assert len(result) == 1
        assert result[0]["type"] == "resize"
        assert result[0]["estimated_savings"] == 50.0
