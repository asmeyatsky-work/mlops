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
    SCENARIOS,
    DemoRunner,
)


@pytest.fixture
def container():
    return DependencyContainer(Settings(use_stubs=True))


@pytest.fixture
def fast_runner(container):
    # pace_factor=0 makes every asyncio.sleep return immediately
    return DemoRunner(container, pace_factor=0.0)


class TestScenarioCatalog:
    def test_all_scenarios_have_metadata(self):
        required = {"title", "summary", "icon"}
        for sid, meta in SCENARIOS.items():
            assert required.issubset(meta.keys()), f"{sid} missing metadata"

    def test_four_scenarios_registered(self):
        assert set(SCENARIOS) == {
            "standard",
            "compliance_block",
            "drift_self_heal",
            "finops_optimize",
        }


class TestStandardScenario:
    async def test_full_step_sequence(self, fast_runner):
        events = [e async for e in fast_runner.run(scenario="standard", model_name="t")]
        steps = [e.payload["step"] for e in events if e.kind == "step.completed"]
        assert steps == [
            "create_dataset", "train_model", "job_status",
            "compliance_gate", "deploy_to_vertex",
            "configure_monitoring", "cost_summary",
        ]

    async def test_narration_events_emitted(self, fast_runner):
        events = [e async for e in fast_runner.run(scenario="standard", model_name="t")]
        narration = [e for e in events if e.kind == "narration"]
        assert len(narration) >= 7  # at least one per major step
        assert any("EU AI Act" in e.payload["text"] for e in narration)

    async def test_run_started_includes_metadata(self, fast_runner):
        events = [e async for e in fast_runner.run(scenario="standard", model_name="t")]
        run_started = events[0]
        assert run_started.kind == "run.started"
        assert run_started.payload["scenario"] == "standard"
        assert run_started.payload["title"]


class TestComplianceBlockScenario:
    async def test_gate_blocks_and_audit_recorded(self, fast_runner):
        events = [e async for e in fast_runner.run(scenario="compliance_block", model_name="t")]
        completed = [e for e in events if e.kind == "step.completed"]
        gate = next(e for e in completed if e.payload["step"] == "compliance_gate")
        assert gate.payload["result"]["decision"] == "block"
        audit = next(e for e in completed if e.payload["step"] == "audit_block")
        assert audit.payload["result"]["audit_entry"] == "recorded"

    async def test_no_deploy_step_executed(self, fast_runner):
        events = [e async for e in fast_runner.run(scenario="compliance_block", model_name="t")]
        steps = {e.payload["step"] for e in events if e.kind == "step.completed"}
        assert "deploy_to_vertex" not in steps


class TestDriftSelfHealScenario:
    async def test_drift_then_retrain(self, fast_runner):
        events = [e async for e in fast_runner.run(scenario="drift_self_heal", model_name="t")]
        steps = [e.payload["step"] for e in events if e.kind == "step.completed"]
        assert "drift_analysis" in steps
        assert "fire_alert" in steps
        assert "retrain" in steps

    async def test_drift_verdict_is_confirmed(self, fast_runner):
        events = [e async for e in fast_runner.run(scenario="drift_self_heal", model_name="t")]
        analysis = next(
            e for e in events
            if e.kind == "step.completed" and e.payload["step"] == "drift_analysis"
        )
        assert analysis.payload["result"]["verdict"] == "DRIFT_CONFIRMED"


class TestFinopsOptimizeScenario:
    async def test_recommendations_and_apply(self, fast_runner):
        events = [e async for e in fast_runner.run(scenario="finops_optimize", model_name="t")]
        steps = [e.payload["step"] for e in events if e.kind == "step.completed"]
        assert "billing_scan" in steps
        assert "generate_recommendations" in steps
        assert "apply_optimization" in steps

    async def test_savings_surface(self, fast_runner):
        events = [e async for e in fast_runner.run(scenario="finops_optimize", model_name="t")]
        recs = next(
            e for e in events
            if e.kind == "step.completed" and e.payload["step"] == "generate_recommendations"
        )
        assert recs.payload["result"]["total_projected_savings_monthly_usd"] > 0


class TestUnknownScenario:
    async def test_unknown_scenario_raises(self, fast_runner):
        with pytest.raises(ValueError):
            async for _ in fast_runner.run(scenario="nope"):
                pass


class TestDemoApp:
    def test_index_returns_html(self, container):
        client = TestClient(build_demo_app(container, pace_factor=0.0))
        r = client.get("/")
        assert r.status_code == 200
        assert "<!DOCTYPE html>" in r.text
        assert "MLOps Orchestrator" in r.text

    def test_agents_endpoint(self, container):
        client = TestClient(build_demo_app(container, pace_factor=0.0))
        body = client.get("/api/agents").json()
        assert len(body) == len(DEMO_AGENTS)

    def test_scenarios_endpoint(self, container):
        client = TestClient(build_demo_app(container, pace_factor=0.0))
        body = client.get("/api/scenarios").json()
        ids = {s["id"] for s in body}
        assert ids == set(SCENARIOS.keys())
        for s in body:
            assert s["title"] and s["summary"]

    def test_run_streams_for_each_scenario(self, container):
        client = TestClient(build_demo_app(container, pace_factor=0.0))
        for scenario in SCENARIOS:
            with client.stream("GET", f"/api/run?scenario={scenario}") as r:
                assert r.status_code == 200
                assert r.headers["content-type"].startswith("text/event-stream")
                kinds = []
                for line in r.iter_lines():
                    if line.startswith("data: "):
                        kinds.append(json.loads(line[6:]).get("kind"))
                    if "run.started" in kinds:
                        break
                assert "run.started" in kinds

    def test_healthz(self, container):
        client = TestClient(build_demo_app(container, pace_factor=0.0))
        assert client.get("/healthz").json() == {"status": "ok"}
