"""Tests for batch prediction adapters and entities."""
from __future__ import annotations

import pytest

from mlops_orchestrator.domain.entities.batch_prediction_job import BatchPredictionJob
from mlops_orchestrator.infrastructure.adapters.stub_batch_prediction_adapter import (
    StubBatchPredictionAdapter,
)


class TestBatchPredictionJob:
    def test_create(self):
        job = BatchPredictionJob.create(
            model_resource_name="projects/p/locations/l/models/m",
            input_uri="gs://bucket/input",
            output_uri="gs://bucket/output",
        )
        assert job.status == "PENDING"
        assert job.model_resource_name == "projects/p/locations/l/models/m"

    def test_start(self):
        job = BatchPredictionJob.create("m", "gs://in", "gs://out")
        started = job.start("projects/p/locations/l/batchPredictionJobs/j1")
        assert started.status == "RUNNING"
        assert started.job_resource_name.endswith("j1")

    def test_complete(self):
        job = BatchPredictionJob.create("m", "gs://in", "gs://out")
        started = job.start("j1")
        completed = started.complete("gs://out/results")
        assert completed.status == "SUCCEEDED"
        assert completed.output_location == "gs://out/results"

    def test_fail(self):
        job = BatchPredictionJob.create("m", "gs://in", "gs://out")
        started = job.start("j1")
        failed = started.fail()
        assert failed.status == "FAILED"

    def test_invalid_transition(self):
        job = BatchPredictionJob.create("m", "gs://in", "gs://out")
        started = job.start("j1")
        completed = started.complete("gs://out")
        with pytest.raises(ValueError, match="Invalid state transition"):
            completed.start("j2")

    def test_is_terminal(self):
        job = BatchPredictionJob.create("m", "gs://in", "gs://out")
        assert not job.is_terminal
        started = job.start("j1")
        assert not started.is_terminal
        assert started.complete("out").is_terminal
        assert started.fail().is_terminal

    def test_pending_to_failed(self):
        job = BatchPredictionJob.create("m", "gs://in", "gs://out")
        failed = job.fail()
        assert failed.status == "FAILED"


class TestStubBatchPredictionAdapter:
    async def test_auto_succeed(self):
        adapter = StubBatchPredictionAdapter(auto_succeed=True)
        rn = await adapter.start_batch_prediction("m", "gs://in", "gs://out", "jsonl")
        assert "batchPredictionJobs" in rn
        assert await adapter.get_job_status(rn) == "SUCCEEDED"

    async def test_running_mode(self):
        adapter = StubBatchPredictionAdapter(auto_succeed=False)
        rn = await adapter.start_batch_prediction("m", "gs://in", "gs://out", "jsonl")
        assert await adapter.get_job_status(rn) == "RUNNING"

    async def test_cancel(self):
        adapter = StubBatchPredictionAdapter(auto_succeed=False)
        rn = await adapter.start_batch_prediction("m", "gs://in", "gs://out", "jsonl")
        assert await adapter.cancel_job(rn) is True
        assert await adapter.get_job_status(rn) == "CANCELLED"

    async def test_cancel_nonexistent(self):
        adapter = StubBatchPredictionAdapter()
        assert await adapter.cancel_job("nope") is False

    async def test_get_output_uri(self):
        adapter = StubBatchPredictionAdapter()
        rn = await adapter.start_batch_prediction("m", "gs://in", "gs://out", "jsonl")
        assert await adapter.get_job_output_uri(rn) == "gs://out"

    async def test_unknown_job_status(self):
        adapter = StubBatchPredictionAdapter()
        assert await adapter.get_job_status("nonexistent") == "UNKNOWN"
