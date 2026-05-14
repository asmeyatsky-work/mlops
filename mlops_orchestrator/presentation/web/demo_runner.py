"""Demo workflow runner.

Drives an end-to-end ML lifecycle using the existing stub-backed commands
and emits a stream of structured events the web UI consumes. No new
domain logic: every step calls the same use-case the MCP tool would.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator

from mlops_orchestrator.application.dtos.dataset_dto import CreateDatasetRequest
from mlops_orchestrator.application.dtos.deployment_dto import (
    DeployToVertexRequest,
    MonitoringRequest,
)
from mlops_orchestrator.application.dtos.training_dto import TrainModelRequest
from mlops_orchestrator.application.session.session_state import SessionState
from mlops_orchestrator.domain.entities.agent import Agent, AgentRole


# Predefined demo swarm — matches the README narrative.
DEMO_AGENTS = (
    Agent.create(
        role=AgentRole.ORCHESTRATOR,
        capabilities=("plan", "coordinate"),
        permitted_tools=(),
    ),
    Agent.create(
        role=AgentRole.DATA_ENGINEER,
        capabilities=("dataset", "bigquery"),
        permitted_tools=("create_dataset",),
    ),
    Agent.create(
        role=AgentRole.ARCHITECT,
        capabilities=("train", "model"),
        permitted_tools=("train_model",),
    ),
    Agent.create(
        role=AgentRole.VALIDATION,
        capabilities=("drift", "compliance", "monitoring"),
        permitted_tools=("configure_monitoring",),
    ),
    Agent.create(
        role=AgentRole.DEPLOYMENT,
        capabilities=("deploy", "endpoint"),
        permitted_tools=("deploy_to_vertex", "deploy_to_gke"),
    ),
    Agent.create(
        role=AgentRole.FINOPS,
        capabilities=("cost", "billing"),
        permitted_tools=(),
    ),
    Agent.create(
        role=AgentRole.SECURITY,
        capabilities=("iam", "compliance"),
        permitted_tools=(),
    ),
)


@dataclass
class DemoEvent:
    kind: str
    timestamp: float = field(default_factory=time.time)
    payload: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "ts": self.timestamp, **self.payload}


class DemoRunner:
    """Streams a scripted end-to-end pipeline. Idempotent per run_id."""

    def __init__(self, container) -> None:
        self._container = container

    def _agent(self, role: AgentRole) -> Agent:
        return next(a for a in DEMO_AGENTS if a.role == role)

    async def run(self, model_name: str = "demo-model") -> AsyncIterator[DemoEvent]:
        run_id = uuid.uuid4().hex[:8]
        c = self._container
        session = SessionState()

        yield DemoEvent("run.started", payload={"run_id": run_id, "agents": [a.role.value for a in DEMO_AGENTS]})

        # ── 1. Dataset ────────────────────────────────────────────────
        agent = self._agent(AgentRole.DATA_ENGINEER)
        yield DemoEvent(
            "step.started",
            payload={"step": "create_dataset", "agent": agent.role.value,
                     "args": {"bq_dataset": "demo_prod", "bq_table": "transactions"}},
        )
        t0 = time.perf_counter()
        ds_response, session = await c.create_dataset_command().execute(
            CreateDatasetRequest(bq_dataset="demo_prod", bq_table="transactions", name=f"{model_name}-ds"),
            session,
        )
        yield DemoEvent(
            "step.completed",
            payload={"step": "create_dataset", "agent": agent.role.value,
                     "result": ds_response.model_dump(),
                     "duration_ms": int((time.perf_counter() - t0) * 1000)},
        )
        await asyncio.sleep(0.4)

        # ── 2. Training ───────────────────────────────────────────────
        agent = self._agent(AgentRole.ARCHITECT)
        yield DemoEvent(
            "step.started",
            payload={"step": "train_model", "agent": agent.role.value,
                     "args": {"model_name": model_name, "dataset_id": session.latest_dataset}},
        )
        t0 = time.perf_counter()
        tr_response, session = await c.train_model_command().execute(
            TrainModelRequest(model_name=model_name, dataset_id=session.latest_dataset, gcs_uri=""),
            session,
        )
        yield DemoEvent(
            "step.completed",
            payload={"step": "train_model", "agent": agent.role.value,
                     "result": tr_response.model_dump(),
                     "duration_ms": int((time.perf_counter() - t0) * 1000)},
        )
        await asyncio.sleep(0.4)

        # ── 3. Validation: poll job ──────────────────────────────────
        agent = self._agent(AgentRole.VALIDATION)
        yield DemoEvent(
            "step.started",
            payload={"step": "job_status", "agent": agent.role.value,
                     "args": {"job_resource_name": tr_response.job_resource_name}},
        )
        t0 = time.perf_counter()
        status = await c.job_status_query().execute(tr_response.job_resource_name)
        yield DemoEvent(
            "step.completed",
            payload={"step": "job_status", "agent": agent.role.value, "result": status,
                     "duration_ms": int((time.perf_counter() - t0) * 1000)},
        )
        await asyncio.sleep(0.3)

        # ── 4. Compliance gate (informational; off by default) ──────
        agent = self._agent(AgentRole.SECURITY)
        yield DemoEvent(
            "step.started",
            payload={"step": "compliance_gate", "agent": agent.role.value,
                     "args": {"model_id": model_name}},
        )
        await asyncio.sleep(0.2)
        from mlops_orchestrator.domain.services.compliance_service import ComplianceService
        cs = ComplianceService()
        risk = cs.classify_risk(domain="finance", intended_purpose="credit_scoring", impacts_fundamental_rights=False)
        yield DemoEvent(
            "step.completed",
            payload={"step": "compliance_gate", "agent": agent.role.value,
                     "result": {
                         "risk_tier": risk.tier.value,
                         "required_controls": list(risk.required_controls),
                         "decision": "allow" if risk.tier.value != "prohibited" else "block",
                     },
                     "duration_ms": 200},
        )
        await asyncio.sleep(0.2)

        # ── 5. Deploy ────────────────────────────────────────────────
        agent = self._agent(AgentRole.DEPLOYMENT)
        yield DemoEvent(
            "step.started",
            payload={"step": "deploy_to_vertex", "agent": agent.role.value,
                     "args": {"model_id": model_name, "endpoint_name": f"{model_name}-ep"}},
        )
        t0 = time.perf_counter()
        dep_response, session = await c.deploy_vertex_command().execute(
            DeployToVertexRequest(model_id=model_name, endpoint_name=f"{model_name}-ep"),
            session,
        )
        yield DemoEvent(
            "step.completed",
            payload={"step": "deploy_to_vertex", "agent": agent.role.value,
                     "result": dep_response.model_dump(),
                     "duration_ms": int((time.perf_counter() - t0) * 1000)},
        )
        await asyncio.sleep(0.3)

        # ── 6. Monitoring ────────────────────────────────────────────
        agent = self._agent(AgentRole.VALIDATION)
        yield DemoEvent(
            "step.started",
            payload={"step": "configure_monitoring", "agent": agent.role.value,
                     "args": {"endpoint_id": session.latest_endpoint}},
        )
        t0 = time.perf_counter()
        mon_response, session = await c.configure_monitoring_command().execute(
            MonitoringRequest(endpoint_id=session.latest_endpoint),
            session,
        )
        yield DemoEvent(
            "step.completed",
            payload={"step": "configure_monitoring", "agent": agent.role.value,
                     "result": mon_response.model_dump(),
                     "duration_ms": int((time.perf_counter() - t0) * 1000)},
        )
        await asyncio.sleep(0.3)

        # ── 7. FinOps summary ────────────────────────────────────────
        # In stub mode the cost port returns zeros; the demo overlays
        # representative numbers so the FinOps story is visible. With
        # MLOPS_BILLING_TABLE configured, the BigQueryCostAdapter returns
        # real numbers and this overlay is bypassed.
        agent = self._agent(AgentRole.FINOPS)
        yield DemoEvent(
            "step.started",
            payload={"step": "cost_summary", "agent": agent.role.value, "args": {"project_id": "mlops-491617"}},
        )
        t0 = time.perf_counter()
        metrics = await c.cost_query().get_project_metrics("mlops-491617")
        recommendations = await c.cost_query().get_recommendations("mlops-491617")
        if c.settings.use_stubs and not recommendations:
            metrics = {
                "cost_per_tb_scanned": 5.0,
                "cost_per_1000_queries": 0.12,
                "cost_per_user": 320.0,
                "gpu_idle_pct": 0.47,
                "_source": "demo-overlay",
            }
            recommendations = [
                {"type": "right_size_gpu", "description": "Vertex training job uses A100; T4 sufficient for 92% of runs", "estimated_savings": 4200.0, "priority": "high"},
                {"type": "schedule_idle_endpoints", "description": "Endpoint demo-model-ep idle 47% of the week — scale to zero off-hours", "estimated_savings": 1800.0, "priority": "high"},
                {"type": "bq_slot_commitment", "description": "On-demand BigQuery > 80% of project; 200-slot commitment breaks even", "estimated_savings": 950.0, "priority": "medium"},
            ]
        yield DemoEvent(
            "step.completed",
            payload={"step": "cost_summary", "agent": agent.role.value,
                     "result": {"metrics": metrics, "recommendations": recommendations},
                     "duration_ms": int((time.perf_counter() - t0) * 1000)},
        )

        yield DemoEvent("run.completed", payload={"run_id": run_id, "session": session.to_dict()})
