"""Tests for the browser-facing demo runner and Starlette app."""
from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

from mlops_orchestrator.infrastructure.config.container import DependencyContainer
from mlops_orchestrator.infrastructure.config.settings import Settings
from mlops_orchestrator.presentation.web.app import build_demo_app
from mlops_orchestrator.presentation.web.demo_runner import (
    DEMO_AGENTS,
    DemoRunner,
)


@pytest.fixture
def container():
    return DependencyContainer(Settings(use_stubs=True))


class TestDemoRunner:
    async def test_emits_expected_step_sequence(self, container):
        runner = DemoRunner(container)
        events = [e async for e in runner.run(model_name="t")]

        kinds = [e.kind for e in events]
        assert kinds[0] == "run.started"
        assert kinds[-1] == "run.completed"

        steps = [
            e.payload["step"]
            for e in events
            if e.kind == "step.completed"
        ]
        assert steps == [
            "create_dataset",
            "train_model",
            "job_status",
            "compliance_gate",
            "deploy_to_vertex",
            "configure_monitoring",
            "cost_summary",
        ]

    async def test_run_completed_carries_session_state(self, container):
        runner = DemoRunner(container)
        events = [e async for e in runner.run(model_name="t")]
        last = events[-1]
        assert last.kind == "run.completed"
        session = last.payload["session"]
        assert session["dataset_ids"], "dataset id should be stitched in"
        assert session["job_handles"], "training job handle should be stitched in"
        assert session["endpoint_names"], "endpoint should be stitched in"


class TestDemoApp:
    def test_index_returns_html(self, container):
        client = TestClient(build_demo_app(container))
        r = client.get("/")
        assert r.status_code == 200
        assert "MLOps Orchestrator" in r.text
        assert "<!DOCTYPE html>" in r.text

    def test_agents_endpoint(self, container):
        client = TestClient(build_demo_app(container))
        r = client.get("/api/agents")
        assert r.status_code == 200
        body = r.json()
        assert len(body) == len(DEMO_AGENTS)
        roles = {a["role"] for a in body}
        assert "data_engineer" in roles and "deployment" in roles

    def test_run_streams_sse_events(self, container):
        client = TestClient(build_demo_app(container))
        with client.stream("GET", "/api/run?model=ci") as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            payloads = []
            for line in r.iter_lines():
                if line.startswith("data: "):
                    payloads.append(json.loads(line[6:]))
                if len(payloads) >= 3:
                    break
        kinds = [p.get("kind") for p in payloads]
        assert "run.started" in kinds

    def test_healthz(self, container):
        client = TestClient(build_demo_app(container))
        assert client.get("/healthz").json() == {"status": "ok"}
