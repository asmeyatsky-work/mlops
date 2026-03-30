# MLOps Orchestrator

Agentic MLOps lifecycle management with multi-agent swarm intelligence, built on the Model Context Protocol (MCP).

## Architecture

Domain-Driven Design with clean architecture layers:

```
mlops_orchestrator/
  domain/           # Value objects, entities, services, ports, events
  application/      # Commands, queries, DTOs, session state, orchestration
  infrastructure/   # GCP adapters (Vertex AI, GKE, Cloud Logging, Billing), auth, stubs, config
  presentation/     # MCP server, CLI, health API (K8s probes)
```

### Key Components

- **Multi-Agent Swarm** — 7 specialist agents (Orchestrator, Architect, Data Engineer, Validation, Deployment, FinOps, Security) coordinated via 5 patterns: Orchestrator-Worker, Swarm, Hierarchical, Mesh, Pipeline
- **DAG Orchestrator** — Dependency-aware workflow execution with automatic parallelization via `asyncio.gather`
- **ML Pipeline Workflow** — End-to-end: data ingestion → training (with polling) → deployment → monitoring
- **Self-Healing Loop** — Observe → Analyze → Decide → Act closed-loop with automated drift remediation and alerting (Slack, PagerDuty, email)
- **EU AI Act Compliance** — Risk classification (Article 6), model cards (Article 11), data governance (Article 10), accuracy/robustness (Article 15)
- **Drift Detection** — KS test, chi-square (with Laplace smoothing), KL divergence, PSI with test-aware severity classification
- **Batch Prediction** — Submit and track Vertex AI BatchPredictionJobs via MCP tool
- **Model Registry** — Model versioning, lifecycle promotion (development → staging → production → archived), and version comparison
- **Session State Stitching** — Immutable state threading across MCP tool calls so the agent handles GCP resource plumbing
- **Authentication** — API key and JWT (HS256) authentication with per-key tool-level authorization
- **Retry & Resilience** — Exponential backoff with jitter on all GCP adapter calls for transient failure recovery
- **FinOps** — Real cost tracking via BigQuery billing export with GPU idle detection and optimization recommendations

### MCP Interface

| Type | Name | Description |
|------|------|-------------|
| Tool | `create_dataset` | Create a Vertex Managed Dataset from BigQuery |
| Tool | `train_model` | Submit a CustomTrainingJob to Vertex AI |
| Tool | `deploy_to_vertex` | Deploy model to a Vertex AI Endpoint |
| Tool | `deploy_to_gke` | Deploy model to a GKE cluster |
| Tool | `configure_monitoring` | Set up drift/skew monitoring |
| Tool | `batch_predict` | Submit a Vertex AI BatchPredictionJob |
| Tool | `register_model` | Register a model version in the model registry |
| Tool | `promote_model` | Promote a model version to a lifecycle stage |
| Resource | `mlops://session` | Current session state |
| Resource | `mlops://jobs/{id}` | Training job status |
| Resource | `mlops://costs/{id}` | FinOps cost metrics |
| Resource | `mlops://models/{id}` | Model versions in registry |

## Quick Start

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run with stubs (no GCP credentials needed)
MLOPS_USE_STUBS=true mlops-orchestrator

# Run with real GCP
MLOPS_GCP_PROJECT=your-project MLOPS_USE_STUBS=false mlops-orchestrator
```

### Docker

```bash
docker-compose up                          # stubs by default
MLOPS_USE_STUBS=false docker-compose up    # with GCP
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MLOPS_GCP_PROJECT` | (empty) | GCP project ID |
| `MLOPS_GCP_LOCATION` | `us-central1` | GCP region |
| `MLOPS_USE_STUBS` | `false` | Use in-memory stubs instead of GCP |
| `MLOPS_TRANSPORT` | `stdio` | MCP transport: `stdio` or `sse` |
| `MLOPS_BILLING_TABLE` | (empty) | Fully-qualified BigQuery billing export table for real cost tracking |
| `MLOPS_SLACK_WEBHOOK_URL` | (empty) | Slack incoming webhook URL for alerts |
| `MLOPS_PAGERDUTY_ROUTING_KEY` | (empty) | PagerDuty Events API v2 routing key |
| `MLOPS_ALERT_EMAIL_SMTP_HOST` | (empty) | SMTP host for email alerts |
| `MLOPS_ALERT_EMAIL_SMTP_PORT` | `587` | SMTP port |
| `MLOPS_ALERT_EMAIL_SENDER` | (empty) | Email sender address |
| `MLOPS_ALERT_EMAIL_RECIPIENTS` | (empty) | Comma-separated recipient emails |
| `MLOPS_ALERT_EMAIL_USERNAME` | (empty) | SMTP username |
| `MLOPS_ALERT_EMAIL_PASSWORD` | (empty) | SMTP password |
| `MLOPS_AUTH_ENABLED` | `false` | Enable API key / JWT authentication |
| `MLOPS_AUTH_API_KEYS` | (empty) | Comma-separated API keys |
| `MLOPS_AUTH_JWT_SECRET` | (empty) | JWT signing secret (HS256) |

## Health Probes (Kubernetes)

The health API supports K8s liveness, readiness, and startup probes:

| Probe | Endpoint | Checks |
|-------|----------|--------|
| Liveness | `/healthz` | Process alive, event loop responsive, uptime |
| Readiness | `/readyz` | All adapter ports instantiated, event bus functional |
| Startup | `/startupz` | Same as readiness (for slow-starting containers) |

## Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=mlops_orchestrator --cov-report=term-missing
```

459 tests, 91% coverage. Test structure mirrors source:

```
tests/
  domain/         # Value objects, entities, services (state machines, drift math, compliance)
  application/    # Commands, queries, DTOs, session state, orchestration (DAG, swarm), batch prediction
  infrastructure/ # Stub adapters, config, auth, alerting, retry, model registry, health probes
  integration/    # End-to-end pipeline, self-healing with alerting, swarm, command chain
```

## Design Decisions

- **Frozen dataclasses** for all domain entities — mutations return new instances, events accumulate in tuples
- **State transition guards** on all entities with explicit valid-transition maps
- **Port/Adapter pattern** — domain ports are `typing.Protocol`, infrastructure provides stub and GCP implementations
- **`asyncio.to_thread`** wraps all synchronous GCP SDK calls to avoid blocking the event loop
- **Immutable session state** with `MappingProxyType` metadata to prevent accidental mutation
- **Test-aware drift severity** — separate threshold scales for KS, chi-square, PSI, and KL divergence statistics
- **Retry with jitter** — all GCP adapters use `@with_retry` decorator with exponential backoff and full jitter for transient errors (408, 429, 500, 502, 503, 504)
- **Composite alerting** — multiple alert channels (Slack, PagerDuty, email) fan out in parallel via `CompositeAlertAdapter`
- **Constant-time auth** — API key validation uses HMAC comparison to prevent timing attacks
- **GCP adapter omission from coverage** — real GCP adapters are excluded from coverage metrics since they require live credentials
