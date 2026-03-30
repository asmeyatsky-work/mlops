"""Tests for the batch prediction command."""
from __future__ import annotations

import pytest

from mlops_orchestrator.application.commands.batch_prediction_command import (
    BatchPredictionCommand,
)
from mlops_orchestrator.application.session.session_state import SessionState
from mlops_orchestrator.infrastructure.adapters.stub_batch_prediction_adapter import (
    StubBatchPredictionAdapter,
)
from mlops_orchestrator.infrastructure.adapters.stub_infrastructure_adapters import (
    InMemoryEventBus,
    StubAuditLogAdapter,
)


@pytest.fixture
def event_bus():
    return InMemoryEventBus()


@pytest.fixture
def audit_log():
    return StubAuditLogAdapter()


@pytest.fixture
def session():
    return SessionState()


class TestBatchPredictionCommand:
    async def test_execute(self, event_bus, audit_log, session):
        adapter = StubBatchPredictionAdapter()
        cmd = BatchPredictionCommand(adapter, event_bus, audit_log)
        job_rn = await cmd.execute(
            model_resource_name="projects/p/locations/l/models/m",
            input_uri="gs://bucket/input",
            output_uri="gs://bucket/output",
            instance_type="jsonl",
            session=session,
        )
        assert "batchPredictionJobs" in job_rn
        assert len(audit_log.all_entries) == 1
        assert audit_log.all_entries[0]["action"] == "batch_prediction_submitted"

    async def test_execute_returns_resource_name(self, event_bus, audit_log, session):
        adapter = StubBatchPredictionAdapter()
        cmd = BatchPredictionCommand(adapter, event_bus, audit_log)
        job_rn = await cmd.execute(
            model_resource_name="m",
            input_uri="gs://in",
            output_uri="gs://out",
            instance_type="csv",
            session=session,
        )
        assert job_rn != ""

    async def test_audit_log_contains_details(self, event_bus, audit_log, session):
        adapter = StubBatchPredictionAdapter()
        cmd = BatchPredictionCommand(adapter, event_bus, audit_log)
        await cmd.execute(
            model_resource_name="projects/p/models/m",
            input_uri="gs://bucket/input.jsonl",
            output_uri="gs://bucket/output/",
            instance_type="jsonl",
            session=session,
        )
        entry = audit_log.all_entries[0]
        assert entry["model"] == "projects/p/models/m"
        assert entry["input_uri"] == "gs://bucket/input.jsonl"
