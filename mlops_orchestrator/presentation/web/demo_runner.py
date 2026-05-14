"""Scenario-driven demo runner.

Each scenario tells one story by walking the orchestrator through a
scripted sequence of agent actions, emitting `narration` events between
steps so the audience understands what is happening and why. Scenarios
reuse the real commands and ports (stub-backed) so the engine, agents,
and session-state stitching are genuine — only the cost overlay and
drift / alerting are scripted to keep the demo deterministic.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable

from mlops_orchestrator.application.dtos.dataset_dto import CreateDatasetRequest
from mlops_orchestrator.application.dtos.deployment_dto import (
    DeployToVertexRequest,
    MonitoringRequest,
)
from mlops_orchestrator.application.dtos.training_dto import TrainModelRequest
from mlops_orchestrator.application.session.session_state import SessionState
from mlops_orchestrator.domain.entities.agent import Agent, AgentRole
from mlops_orchestrator.domain.services.compliance_service import (
    ComplianceGateError,
    ComplianceGateService,
    ComplianceService,
)
from mlops_orchestrator.domain.value_objects.compliance import (
    ModelCard,
    RiskClassification,
    RiskTier,
)


# Predefined swarm — same agents the README narrates.
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


# Pacing — keep the demo readable. Override with `pace_factor` for tests.
DEFAULT_PACE = 1.0


@dataclass
class DemoEvent:
    kind: str
    timestamp: float = field(default_factory=time.time)
    payload: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "ts": self.timestamp, **self.payload}


def _agent(role: AgentRole) -> Agent:
    return next(a for a in DEMO_AGENTS if a.role == role)


# ─── Scenario metadata ───────────────────────────────────────────────

SCENARIOS = {
    "standard": {
        "title": "Standard ML lifecycle",
        "summary": "End-to-end pipeline: data → train → deploy → monitor. Shows the swarm coordinating through the full ML lifecycle.",
        "icon": "🚀",
    },
    "compliance_block": {
        "title": "Compliance gate blocks a prohibited model",
        "summary": "An EU AI Act PROHIBITED use case (social scoring) is stopped at the gate before any Vertex resources are created.",
        "icon": "🛡",
    },
    "drift_self_heal": {
        "title": "Drift detection triggers self-healing",
        "summary": "Production drift is detected by Vertex Model Monitoring. The self-healing loop fires alerts and queues a retraining run.",
        "icon": "🔁",
    },
    "finops_optimize": {
        "title": "FinOps recommends a cost cut",
        "summary": "BigQuery billing export reveals GPU idle waste. The FinOps agent proposes right-sizing and the Deployment agent applies it.",
        "icon": "💸",
    },
}


# ─── Scenario implementations ────────────────────────────────────────


class DemoRunner:
    """Streams scripted scenarios. Idempotent per run_id; deterministic."""

    def __init__(self, container, pace_factor: float = DEFAULT_PACE) -> None:
        self._container = container
        self._pace = pace_factor

    async def _pause(self, seconds: float) -> None:
        await asyncio.sleep(seconds * self._pace)

    def _event(self, kind: str, **payload) -> DemoEvent:
        return DemoEvent(kind, payload=payload)

    async def run(
        self, scenario: str = "standard", model_name: str = "demo-model"
    ) -> AsyncIterator[DemoEvent]:
        scenarios: dict[str, Callable] = {
            "standard": self._scenario_standard,
            "compliance_block": self._scenario_compliance_block,
            "drift_self_heal": self._scenario_drift_self_heal,
            "finops_optimize": self._scenario_finops_optimize,
        }
        if scenario not in scenarios:
            raise ValueError(f"unknown scenario: {scenario}")

        meta = SCENARIOS[scenario]
        run_id = uuid.uuid4().hex[:8]
        yield self._event(
            "run.started",
            run_id=run_id,
            scenario=scenario,
            title=meta["title"],
            summary=meta["summary"],
            agents=[a.role.value for a in DEMO_AGENTS],
        )
        await self._pause(0.6)

        async for event in scenarios[scenario](model_name):
            yield event

        yield self._event("run.completed", run_id=run_id, scenario=scenario)

    # ── Shared narration helpers ───────────────────────────────────

    async def _narrate(self, agent: AgentRole, text: str, pause: float = 1.4) -> AsyncIterator[DemoEvent]:
        yield self._event("narration", agent=agent.value, text=text)
        await self._pause(pause)

    async def _step(
        self,
        step: str,
        agent: AgentRole,
        args: dict,
        body: Callable,
        post_pause: float = 1.0,
    ) -> AsyncIterator[DemoEvent]:
        yield self._event("step.started", step=step, agent=agent.value, args=args)
        t0 = time.perf_counter()
        try:
            result = await body()
        except Exception as exc:
            yield self._event(
                "step.failed",
                step=step,
                agent=agent.value,
                error=str(exc),
                duration_ms=int((time.perf_counter() - t0) * 1000),
            )
            return
        yield self._event(
            "step.completed",
            step=step,
            agent=agent.value,
            result=result,
            duration_ms=int((time.perf_counter() - t0) * 1000),
        )
        await self._pause(post_pause)

    # ── Scenario 1: Standard ML lifecycle ──────────────────────────

    async def _scenario_standard(self, model_name: str) -> AsyncIterator[DemoEvent]:
        c = self._container
        session = SessionState()

        async for ev in self._narrate(
            AgentRole.ORCHESTRATOR,
            "Kicking off a fresh ML lifecycle. I'll fan tasks out to the specialists and stitch their outputs together so the user never plumbs resource IDs by hand.",
        ):
            yield ev

        # 1. Dataset
        async for ev in self._narrate(
            AgentRole.DATA_ENGINEER,
            "Creating a Vertex Managed Dataset from a BigQuery table. The dataset resource name becomes the input for training.",
            pause=1.0,
        ):
            yield ev

        async def _create():
            nonlocal session
            r, session = await c.create_dataset_command().execute(
                CreateDatasetRequest(bq_dataset="demo_prod", bq_table="transactions", name=f"{model_name}-ds"),
                session,
            )
            return r.model_dump()

        async for ev in self._step(
            "create_dataset", AgentRole.DATA_ENGINEER,
            {"bq_dataset": "demo_prod", "bq_table": "transactions"}, _create,
        ):
            yield ev

        # 2. Train
        async for ev in self._narrate(
            AgentRole.ARCHITECT,
            f"Submitting a CustomTrainingJob. Returns a job handle immediately; the Validation agent will poll for completion. Notice the dataset_id was filled in from session state — no copy-paste.",
        ):
            yield ev

        async def _train():
            nonlocal session
            r, session = await c.train_model_command().execute(
                TrainModelRequest(model_name=model_name, dataset_id=session.latest_dataset, gcs_uri=""),
                session,
            )
            return r.model_dump()

        async for ev in self._step(
            "train_model", AgentRole.ARCHITECT,
            {"model_name": model_name, "dataset_id": session.latest_dataset}, _train,
        ):
            yield ev

        # 3. Validation polling
        async for ev in self._narrate(
            AgentRole.VALIDATION,
            "Polling the job. In production this respects an asyncio.wait_for hard timeout so a hung Vertex backend can't block the orchestrator indefinitely.",
        ):
            yield ev

        async def _poll():
            return await c.job_status_query().execute(session.latest_job)

        async for ev in self._step(
            "job_status", AgentRole.VALIDATION,
            {"job_resource_name": session.latest_job}, _poll,
        ):
            yield ev

        # 4. Compliance
        async for ev in self._narrate(
            AgentRole.SECURITY,
            "Classifying the model under EU AI Act Article 6. For 'finance / credit_scoring' I'm returning HIGH-risk with the seven required controls listed in Article 9. The gate would block deploy if any are missing — for the demo I report 'ALLOW'.",
        ):
            yield ev

        cs = ComplianceService()
        risk = cs.classify_risk(domain="finance", intended_purpose="credit_scoring", impacts_fundamental_rights=False)

        async def _gate():
            return {
                "risk_tier": risk.tier.value,
                "required_controls": list(risk.required_controls),
                "decision": "allow",
            }

        async for ev in self._step(
            "compliance_gate", AgentRole.SECURITY, {"model_id": model_name}, _gate,
        ):
            yield ev

        # 5. Deploy
        async for ev in self._narrate(
            AgentRole.DEPLOYMENT,
            "Creating a Vertex Endpoint with an n1-standard-4 machine spec and deploying the trained model. Endpoint resource name is appended to session state for downstream steps.",
        ):
            yield ev

        async def _deploy():
            nonlocal session
            r, session = await c.deploy_vertex_command().execute(
                DeployToVertexRequest(model_id=model_name, endpoint_name=f"{model_name}-ep"),
                session,
            )
            return r.model_dump()

        async for ev in self._step(
            "deploy_to_vertex", AgentRole.DEPLOYMENT,
            {"model_id": model_name, "endpoint_name": f"{model_name}-ep"}, _deploy,
        ):
            yield ev

        # 6. Monitoring
        async for ev in self._narrate(
            AgentRole.VALIDATION,
            "Wiring up Vertex Model Monitoring on the new endpoint. Drift + skew thresholds default to 0.05 / 0.10. Alerts will route to Slack + PagerDuty via the CompositeAlertAdapter.",
        ):
            yield ev

        async def _mon():
            nonlocal session
            r, session = await c.configure_monitoring_command().execute(
                MonitoringRequest(endpoint_id=session.latest_endpoint),
                session,
            )
            return r.model_dump()

        async for ev in self._step(
            "configure_monitoring", AgentRole.VALIDATION,
            {"endpoint_id": session.latest_endpoint}, _mon,
        ):
            yield ev

        # 7. FinOps
        async for ev in self._narrate(
            AgentRole.FINOPS,
            "Reading BigQuery billing export. In stub mode I'm overlaying representative numbers; in production this is real per-project, per-resource spend with GPU idle detection.",
        ):
            yield ev

        async def _cost():
            return _stub_cost_overlay() if c.settings.use_stubs else {
                "metrics": await c.cost_query().get_project_metrics("mlops-491617"),
                "recommendations": await c.cost_query().get_recommendations("mlops-491617"),
            }

        async for ev in self._step(
            "cost_summary", AgentRole.FINOPS, {"project_id": "mlops-491617"}, _cost,
        ):
            yield ev

        async for ev in self._narrate(
            AgentRole.ORCHESTRATOR,
            "Lifecycle complete. The session now carries dataset / job / endpoint / model IDs — the next conversational turn from the user can reference any of them without re-pasting.",
            pause=0.6,
        ):
            yield ev

    # ── Scenario 2: Compliance gate blocks ─────────────────────────

    async def _scenario_compliance_block(self, model_name: str) -> AsyncIterator[DemoEvent]:
        c = self._container

        async for ev in self._narrate(
            AgentRole.ORCHESTRATOR,
            f"A request landed to deploy '{model_name}' for a social-scoring use case. Before I touch any GCP resources, the compliance gate must classify and approve.",
        ):
            yield ev

        async for ev in self._narrate(
            AgentRole.SECURITY,
            "Running EU AI Act Article 6 classification. The keyword 'social_scoring' matches the Annex III prohibited list. This is a hard stop — not even a model card can override it.",
        ):
            yield ev

        cs = ComplianceService()
        gate = ComplianceGateService()
        risk = cs.classify_risk(
            domain="public_sector", intended_purpose="social_scoring", impacts_fundamental_rights=True,
        )

        async def _classify():
            return {
                "risk_tier": risk.tier.value,
                "justification": risk.justification,
                "required_controls": list(risk.required_controls),
            }

        async for ev in self._step(
            "risk_classification", AgentRole.SECURITY,
            {"domain": "public_sector", "intended_purpose": "social_scoring"}, _classify,
        ):
            yield ev

        async for ev in self._narrate(
            AgentRole.SECURITY,
            "Calling the gate. The gate evaluates risk + model card together and either raises or returns. Watch the failure event below — no Vertex Endpoint is created.",
        ):
            yield ev

        async def _gate_call():
            try:
                gate.enforce(risk, ModelCard(model_name=model_name, version="1", purpose="social scoring", limitations=""))
            except ComplianceGateError as exc:
                return {"decision": "block", "reasons": [str(exc)]}
            return {"decision": "allow"}

        async for ev in self._step(
            "compliance_gate", AgentRole.SECURITY, {"model_id": model_name}, _gate_call,
        ):
            yield ev

        async for ev in self._narrate(
            AgentRole.DEPLOYMENT,
            "I would have called deploy_to_vertex next, but the gate returned block. Skipping. The audit log records the decision and the reason — that's the artifact the EU regulator asks for.",
        ):
            yield ev

        async def _audit():
            await c.audit_log.log_action(
                action="deploy_blocked",
                resource_id=model_name,
                details={"reason": "EU AI Act Article 6 — prohibited use case"},
            )
            return {"audit_entry": "recorded"}

        async for ev in self._step(
            "audit_block", AgentRole.SECURITY, {"action": "deploy_blocked"}, _audit,
        ):
            yield ev

        async for ev in self._narrate(
            AgentRole.ORCHESTRATOR,
            "Outcome: zero GCP spend, full audit trail, regulator-ready evidence. The same pattern protects you against HIGH-risk models that lack required documentation.",
            pause=0.6,
        ):
            yield ev

    # ── Scenario 3: Drift detection + self-healing ─────────────────

    async def _scenario_drift_self_heal(self, model_name: str) -> AsyncIterator[DemoEvent]:
        c = self._container
        session = SessionState()

        async for ev in self._narrate(
            AgentRole.ORCHESTRATOR,
            "Model has been live for two weeks. Validation agent just received a drift alert from Vertex Model Monitoring. Walking through the observe → analyze → decide → act loop.",
        ):
            yield ev

        async for ev in self._narrate(
            AgentRole.VALIDATION,
            "Observe: KS test on the 'transaction_amount' feature returned p=0.003 (threshold 0.05). PSI=0.42, decisively above the 0.25 alert threshold. This is real drift, not noise.",
        ):
            yield ev

        async def _observe():
            return {
                "endpoint_id": f"projects/x/locations/us-central1/endpoints/{model_name}-ep",
                "features_drifted": [
                    {"name": "transaction_amount", "ks_pvalue": 0.003, "psi": 0.42, "severity": "high"},
                    {"name": "merchant_category", "chi_square": 18.4, "severity": "medium"},
                ],
                "verdict": "DRIFT_CONFIRMED",
            }

        async for ev in self._step(
            "drift_analysis", AgentRole.VALIDATION,
            {"endpoint_id": f"{model_name}-ep", "window": "24h"}, _observe,
        ):
            yield ev

        async for ev in self._narrate(
            AgentRole.SECURITY,
            "Analyze: cross-checking against the model card's declared distribution. The merchant_category shift correlates with a known holiday surge — but transaction_amount drift is anomalous and warrants retraining.",
        ):
            yield ev

        async for ev in self._narrate(
            AgentRole.ORCHESTRATOR,
            "Decide: queue a remediation. I'll fan out to (a) Deployment to scale endpoint replicas defensively and (b) Architect to kick a retraining run with the latest two weeks of data.",
        ):
            yield ev

        async def _alert():
            return {
                "channels": ["slack", "pagerduty"],
                "severity": "high",
                "incident_id": f"INC-{uuid.uuid4().hex[:6]}",
            }

        async for ev in self._step(
            "fire_alert", AgentRole.VALIDATION,
            {"severity": "high", "channels": ["slack", "pagerduty"]}, _alert,
        ):
            yield ev

        async for ev in self._narrate(
            AgentRole.ARCHITECT,
            "Act: submitting a retrain on the most recent BigQuery slice. New model artifacts will be staged under a fresh version in the registry — production stays on the old version until validation gives the new one a green light.",
        ):
            yield ev

        async def _retrain():
            nonlocal session
            r, session = await c.train_model_command().execute(
                TrainModelRequest(model_name=f"{model_name}-v2", dataset_id="", gcs_uri="gs://demo/recent.csv"),
                session,
            )
            return r.model_dump()

        async for ev in self._step(
            "retrain", AgentRole.ARCHITECT, {"model_name": f"{model_name}-v2"}, _retrain,
        ):
            yield ev

        async for ev in self._narrate(
            AgentRole.DEPLOYMENT,
            "Once the new version passes validation, I promote it via the model registry's staging → production lifecycle. Old version stays archived for fast rollback.",
            pause=0.8,
        ):
            yield ev

        async for ev in self._narrate(
            AgentRole.ORCHESTRATOR,
            "Outcome: drift detected, audit logged, alert fired, retrain queued — all without paging a human. The platform team gets a notification, not an incident.",
            pause=0.6,
        ):
            yield ev

    # ── Scenario 4: FinOps right-sizing ────────────────────────────

    async def _scenario_finops_optimize(self, model_name: str) -> AsyncIterator[DemoEvent]:
        c = self._container

        async for ev in self._narrate(
            AgentRole.ORCHESTRATOR,
            "Weekly FinOps review. I'm running the cost agent over BigQuery billing export to find the biggest savings opportunity in this project.",
        ):
            yield ev

        async for ev in self._narrate(
            AgentRole.FINOPS,
            "Reading the last 30 days from the billing export table. Aggregating spend by service and looking for the GPU-idle signal — the single biggest line item in most ML budgets.",
        ):
            yield ev

        async def _scan():
            return {
                "project_id": "mlops-491617",
                "total_spend_30d_usd": 18420.0,
                "top_services": [
                    {"service": "Vertex AI Training", "spend": 9320.0, "share": 0.51},
                    {"service": "BigQuery", "spend": 4880.0, "share": 0.26},
                    {"service": "Cloud Storage", "spend": 1700.0, "share": 0.09},
                ],
                "gpu_idle_pct": 0.47,
            }

        async for ev in self._step(
            "billing_scan", AgentRole.FINOPS,
            {"window": "30d", "table": "project_billing_export.gcp_billing"}, _scan,
        ):
            yield ev

        async for ev in self._narrate(
            AgentRole.FINOPS,
            "GPU idle is 47% — that's $4,200/month lit on fire. The A100 used in training is overkill for 92% of jobs; recommendation engine flags right-sizing to T4.",
        ):
            yield ev

        async def _recommend():
            return {
                "recommendations": [
                    {"type": "right_size_gpu", "description": "Swap A100 → T4 for non-vision workloads", "estimated_savings": 4200.0, "priority": "high"},
                    {"type": "schedule_idle_endpoints", "description": f"{model_name}-ep idle 47% of the week — scale to zero off-hours", "estimated_savings": 1800.0, "priority": "high"},
                    {"type": "bq_slot_commitment", "description": "On-demand BigQuery > 80% — 200-slot commitment breaks even at week 3", "estimated_savings": 950.0, "priority": "medium"},
                ],
                "total_projected_savings_monthly_usd": 6950.0,
            }

        async for ev in self._step(
            "generate_recommendations", AgentRole.FINOPS, {"limit": 5}, _recommend,
        ):
            yield ev

        async for ev in self._narrate(
            AgentRole.SECURITY,
            "Before any change touches production, the compliance gate checks that down-spec'd resources still meet Article 15 robustness requirements. T4 throughput is documented as sufficient — green-lit.",
        ):
            yield ev

        async for ev in self._narrate(
            AgentRole.DEPLOYMENT,
            "Applying the highest-priority change first: right-sizing the endpoint to a T4 machine spec. Rolling update; no downtime.",
        ):
            yield ev

        async def _apply():
            return {
                "action": "right_size_endpoint",
                "endpoint": f"{model_name}-ep",
                "before": "n1-standard-4 + nvidia-tesla-a100",
                "after": "n1-standard-4 + nvidia-tesla-t4",
                "estimated_monthly_savings_usd": 4200.0,
                "rollout": "rolling, 2/2 replicas updated",
            }

        async for ev in self._step(
            "apply_optimization", AgentRole.DEPLOYMENT, {"change": "right_size_gpu"}, _apply,
        ):
            yield ev

        async for ev in self._narrate(
            AgentRole.ORCHESTRATOR,
            "Outcome: ~$50K/year saved, change recorded in the audit log, no human in the loop. The same loop runs weekly.",
            pause=0.6,
        ):
            yield ev


def _stub_cost_overlay() -> dict:
    """Representative cost numbers for the standard-scenario FinOps step."""
    return {
        "metrics": {
            "cost_per_tb_scanned": 5.0,
            "cost_per_1000_queries": 0.12,
            "cost_per_user": 320.0,
            "gpu_idle_pct": 0.47,
            "_source": "demo-overlay",
        },
        "recommendations": [
            {"type": "right_size_gpu", "description": "Vertex training A100 — T4 sufficient for 92% of runs", "estimated_savings": 4200.0, "priority": "high"},
            {"type": "schedule_idle_endpoints", "description": "Endpoint idle 47% of the week — scale to zero off-hours", "estimated_savings": 1800.0, "priority": "high"},
            {"type": "bq_slot_commitment", "description": "On-demand BQ > 80% of project; 200-slot commitment breaks even", "estimated_savings": 950.0, "priority": "medium"},
        ],
    }
