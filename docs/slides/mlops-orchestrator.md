<!--
Render with: npx @marp-team/marp-cli mlops-orchestrator.md -o mlops-orchestrator.pdf
                                                            -o mlops-orchestrator.pptx
                                                            -o mlops-orchestrator.html
-->
---
marp: true
theme: default
paginate: true
size: 16:9
header: "MLOps Orchestrator"
footer: "Agentic MLOps on GCP — confidential"
style: |
  section { font-size: 22px; }
  h1 { color: #1a73e8; }
  h2 { color: #1a73e8; border-bottom: 1px solid #e8eaed; padding-bottom: 4px; }
  code { background: #f1f3f4; padding: 1px 4px; border-radius: 3px; }
  table { font-size: 18px; }
---

# Agentic MLOps Orchestrator
### Multi-agent ML lifecycle management on GCP

MCP-native · Vertex AI · GKE · EU AI Act-aware

---

## The problem

Teams running ML on GCP repeatedly rebuild the same plumbing:

- **Manual handoffs** between data eng, ML, SRE — weeks of pipeline work
- **GPU idle waste** invisible until the monthly bill arrives
- **Silent model drift** caught only after a business KPI moves
- **EU AI Act** adds risk classification, model cards, provenance, audit
- **Agent-first stacks** (MCP, autonomous LLMs) demand a new control plane

A control plane that's *agentic, governed, and cost-aware* is the gap.

---

## Solution at a glance

A single MCP server exposing the full ML lifecycle as **tools** + **resources**,
orchestrated by a swarm of specialist agents.

- **7 specialists** (Orchestrator, Architect, Data Eng, Validation, Deployment, FinOps, Security)
- **5 coordination patterns** (Orchestrator-Worker, Swarm, Hierarchical, Mesh, Pipeline)
- **DAG-aware** workflow runner with automatic parallelism
- **Self-healing** observe → analyze → decide → act loop with alerting
- **EU AI Act compliance gate** that *enforces*, not just documents
- ~10K LOC Python · 487 tests · 90% coverage

---

## Architecture — clean layers, ports & adapters

```
mlops_orchestrator/
  domain/           value objects · entities · ports (Protocols) · services · events
  application/      commands · queries · DTOs · session state · orchestration
  infrastructure/   GCP adapters · auth · observability · config · stubs
  presentation/     MCP server · CLI · K8s health API
```

- Domain layer has **zero GCP imports** — pure logic
- Every external system behind a port (`TrainingPort`, `DatasetPort`, `ModelGovernancePort`, …)
- Stub adapters in-memory → unit tests run with zero cloud cost
- Production composition root wires Vertex AI / GKE / BigQuery adapters

---

## Multi-agent swarm + DAG orchestration

```
Orchestrator ──┬── Data Engineer  ── BQ source · dataset
               ├── Architect      ── model design
               ├── Validation     ── drift · adversarial
               ├── Deployment     ── Vertex / GKE
               ├── FinOps         ── BigQuery billing export
               └── Security       ── IAM · metadata sanitization
```

- Agents are **frozen dataclasses** with `permitted_tools` allowlist
- DAG orchestrator parallelizes independent steps via `asyncio.gather`
- Whole-word capability matching defeats prompt-injection on routing

---

## MCP interface — tools = writes, resources = reads

| Type | Name | Purpose |
|------|------|---------|
| Tool | `create_dataset` | Vertex Managed Dataset from BigQuery |
| Tool | `train_model` | CustomTrainingJob (async handle) |
| Tool | `deploy_to_vertex` / `deploy_to_gke` | Endpoint / GKE Deployment |
| Tool | `batch_predict` | Vertex BatchPredictionJob |
| Tool | `register_model` / `promote_model` | Lifecycle promotion |
| Tool | `configure_monitoring` | Drift & skew detection |
| Resource | `mlops://session` | Stitched session state |
| Resource | `mlops://jobs/{id}` · `mlops://costs/{p}` · `mlops://models/{m}` | Observable state |

**Session-state stitching** = the agent never plumbs resource IDs by hand.

---

## Resilience under degraded conditions

Every external call is wrapped:

- **Retry with exponential backoff + full jitter** on 408/429/5xx
- **Hard per-attempt timeout** via `asyncio.wait_for`
- **Circuit breaker** (CLOSED → OPEN → HALF_OPEN) trips after N failures
- **Lock-free I/O** — MCP server snapshots session state under lock, runs the GCP call lock-free, applies result under lock. Concurrent tool calls don't serialize.

`JobStatusQuery.poll_until_complete` honours its declared timeout even if Vertex hangs.

---

## Observability by default

- **Correlation ID** generated per MCP tool call, threaded via `contextvars`, surfaced in every JSON log line
- **OpenTelemetry spans** (optional `[otel]` extra) wrap DAG steps, swarm coordination, every adapter call
- **Structured JSON logs** with PII redaction (bearer tokens, API keys, emails, JWT-shaped strings, password/secret K=V)
- **Cloud Logging audit trail** on every command (success and failure)
- **Health probes** — liveness · readiness · startup endpoints for K8s

---

## Security & AI safety

- **SSE auth middleware** (ASGI) — `X-API-Key` or `Authorization: Bearer <JWT HS256>`; bad creds → 401 before the MCP handler
- **Per-agent RBAC** — `enforce_tool_authz` at every tool dispatch checks the active `Principal.permitted_tools`
- **DNS-rebinding guard** — `MLOPS_ALLOWED_HOSTS` whitelists the platform hostname
- **Production guard** — `MLOPS_ENVIRONMENT=production` refuses to boot without auth, compliance gate, and real adapters
- **Whole-word specialist routing** defeats embedded-substring attacks in task descriptions
- **Supply chain** — pinned upper bounds, pip-audit, CycloneDX SBOM, Trivy scan, Dependabot, multi-stage non-root Dockerfile

---

## EU AI Act — enforced, not documented

`ComplianceGateService` runs **before** every deploy:

- **PROHIBITED** tier → blocked, always
- **HIGH-risk** tier → must have complete `ModelCard` (Article 11) + non-empty required controls
- **LIMITED** tier → must record a transparency disclosure
- **MINIMAL** tier → allowed

Plus pure-domain helpers for Article 6 (risk classification), Article 10 (data governance gaps), Article 15 (accuracy + adversarial + robustness).

> A high-risk model with no card never reaches a Vertex Endpoint.

---

## FinOps — real costs, not stubs

- **BigQuery billing export adapter** — actual project spend, per-resource breakdown
- **GPU idle detection** — surfaces the biggest line item in most ML budgets
- **Optimization recommendations** — concrete actions, not generic advice
- Exposed via `mlops://costs/{project_id}` MCP resource → agents can reason about cost

A common 10-person ML platform team saves multiples of its salary cost in idle-accelerator and right-sizing alone.

---

## Demo deployment — Cloud Run, ~$0

```bash
gcloud builds submit . --tag=$IMG --project=$P
gcloud run deploy mlops-orchestrator-demo \
  --image=$IMG --region=us-central1 \
  --no-allow-unauthenticated \
  --port=8000 --cpu=1 --memory=512Mi \
  --min-instances=0 --max-instances=2 \
  --set-env-vars="MLOPS_USE_STUBS=true,MLOPS_TRANSPORT=sse,\
MLOPS_AUTH_ENABLED=true,MLOPS_AUTH_API_KEYS=$KEY,\
MLOPS_ALLOWED_HOSTS=$HOST"
```

- Scale-to-zero · Cloud Run free tier covers 2M req/mo
- Stub adapters → no Vertex / GKE / BigQuery spend
- SSE endpoint, API-key gated, IAM-restricted to your domain
- Already running: `mlops-orchestrator-demo.us-central1.run.app`

---

## Business value

| Lever | Mechanism | Outcome |
|-------|-----------|---------|
| **Time-to-prod** | Multi-agent + DAG · session stitching | Weeks → days for new pipelines |
| **Cost** | BQ billing · GPU idle · recommendations | Direct reduction on biggest line item |
| **Risk** | Drift + self-healing + alerting | Catch model decay before KPIs move |
| **Regulation** | Enforced EU AI Act gate | Sell into EU-regulated markets |
| **Enterprise fit** | Auth · RBAC · audit · K8s probes | Adoptable by platform teams |

---

## What's next

- Persistent governance backend (BigQuery / dedicated DB) behind `ModelGovernancePort`
- OpenTelemetry exporter wired to Cloud Trace by default
- Cloud Run job + Vertex Pipelines templates for non-MCP-aware clients
- Cross-region failover for the MCP control plane
- SLSA Level 3 release pipeline (keyless signing via OIDC + cosign)

---

# Thank you

**Repo:** `github.com/asmeyatsky-work/mlops`
**Demo:** `mlops-orchestrator-demo-946022635729.us-central1.run.app`
**Stack:** Python 3.11+ · MCP · Vertex AI · GKE · BigQuery · Cloud Run

*Questions?*
