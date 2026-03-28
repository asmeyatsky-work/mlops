# MLOps Orchestrator

Agentic MLOps lifecycle management with multi-agent swarm intelligence, built on the Model Context Protocol (MCP).

## Architecture

Domain-Driven Design with clean architecture layers:

```
mlops_orchestrator/
  domain/           # Value objects, entities, services, ports, events
  application/      # Commands, queries, DTOs, session state, orchestration
  infrastructure/   # GCP adapters (Vertex AI, GKE, Cloud Logging), stubs, config
  presentation/     # MCP server, CLI, health API
```

### Key Components

- **Multi-Agent Swarm** — 7 specialist agents (Orchestrator, Architect, Data Engineer, Validation, Deployment, FinOps, Security) coordinated via 5 patterns: Orchestrator-Worker, Swarm, Hierarchical, Mesh, Pipeline
- **DAG Orchestrator** — Dependency-aware workflow execution with automatic parallelization via `asyncio.gather`
- **ML Pipeline Workflow** — End-to-end: data ingestion → training (with polling) → deployment → monitoring
- **Self-Healing Loop** — Observe → Analyze → Decide → Act closed-loop for automated drift remediation
- **EU AI Act Compliance** — Risk classification (Article 6), model cards (Article 11), data governance (Article 10), accuracy/robustness (Article 15)
- **Drift Detection** — KS test, chi-square (with Laplace smoothing), KL divergence, PSI with test-aware severity classification
- **Session State Stitching** — Immutable state threading across MCP tool calls so the agent handles GCP resource plumbing

### MCP Interface

| Type | Name | Description |
|------|------|-------------|
| Tool | `create_dataset` | Create a Vertex Managed Dataset from BigQuery |
| Tool | `train_model` | Submit a CustomTrainingJob to Vertex AI |
| Tool | `deploy_to_vertex` | Deploy model to a Vertex AI Endpoint |
| Tool | `deploy_to_gke` | Deploy model to a GKE cluster |
| Tool | `configure_monitoring` | Set up drift/skew monitoring |
| Resource | `mlops://session` | Current session state |
| Resource | `mlops://jobs/{id}` | Training job status |
| Resource | `mlops://costs/{id}` | FinOps cost metrics |

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

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MLOPS_GCP_PROJECT` | (empty) | GCP project ID |
| `MLOPS_GCP_LOCATION` | `us-central1` | GCP region |
| `MLOPS_USE_STUBS` | `false` | Use in-memory stubs instead of GCP |
| `MLOPS_TRANSPORT` | `stdio` | MCP transport: `stdio` or `sse` |

## Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=mlops_orchestrator --cov-report=term-missing
```

339 tests, 81% coverage. Test structure mirrors source:

```
tests/
  domain/         # Value objects, entities, services (state machines, drift math, compliance)
  application/    # Commands, queries, DTOs, session state, orchestration (DAG, swarm)
  infrastructure/ # Stub adapters, config, dependency container
  integration/    # End-to-end pipeline, self-healing, swarm, command chain
```

## Design Decisions

- **Frozen dataclasses** for all domain entities — mutations return new instances, events accumulate in tuples
- **State transition guards** on all entities with explicit valid-transition maps
- **Port/Adapter pattern** — domain ports are `typing.Protocol`, infrastructure provides stub and GCP implementations
- **`asyncio.to_thread`** wraps all synchronous GCP SDK calls to avoid blocking the event loop
- **Immutable session state** with `MappingProxyType` metadata to prevent accidental mutation
- **Test-aware drift severity** — separate threshold scales for KS, chi-square, PSI, and KL divergence statistics
