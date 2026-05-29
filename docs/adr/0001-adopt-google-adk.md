# ADR 0001 — Adopt Google Agent Development Kit (ADK) for reasoning agents

- Status: Accepted
- Date: 2026-05-29
- Deciders: MLOps Orchestrator maintainers

## Context

The swarm's seven roles existed only as immutable DTOs (`domain/entities/agent.py`)
with a `SwarmCoordinator` whose `task_executor` callback had no production
implementation — agency was scaffolding, never wired to a reasoning engine. We
are introducing genuine LLM reasoning for the roles where ambiguity and
natural-language decomposition add value (Orchestrator, Architect, Data
Engineer, Validation, FinOps), while keeping side-effecting/policy roles
(Deployment, Security) deterministic.

Architectural Rules §1 lists Python as the default for agentic orchestration but
does not name an agent framework; adopting a specific SDK (`google-adk`) is a
stack deviation and therefore requires this ADR.

## Decision

Adopt `google-adk` (Gemini-backed) as the reasoning engine, behind a domain port
(`ReasoningAgentPort`). The SDK is confined to a single infrastructure adapter
(`ADKReasoningAdapter`) and is an optional `[adk]` extra; the domain and
application layers import nothing from it (§2, §3.3). The default and unit-tested
path is the deterministic `StubReasoningAdapter`.

## Rationale

- Already on GCP (Vertex AI, BigQuery, Cloud Logging); ADK + Gemini keeps the
  cloud surface, IAM model, and Workload Identity story consistent (§1).
- ADK's `output_schema` gives structured JSON, which we validate with Pydantic
  before mapping to immutable domain value objects — satisfying §4 ("AI output
  that mutates state must be validated against an explicit schema first").
- Reasoning stays advisory: agents return `Plan`/`Recommendation` value objects;
  the deterministic command layer and EU AI Act compliance gate retain control
  of all actuation.

## Consequences

- New optional dependency `google-adk>=1.0.0,<2.0` (pinned upper bound per the
  supply-chain rule). Not required for stub/dev/test runs.
- `ADKReasoningAdapter` requires a live Gemini endpoint, so it is excluded from
  unit-test coverage (like the `vertex_*`/`gke_*` adapters) and exercised via
  integration runs. The deterministic stub holds the testing floor (§5).
- Per-AI-call observability (model id, prompt hash, tokens, latency) is logged on
  the existing correlation id (§6).
- Reasoning is disabled by default (`MLOPS_REASONING_ENABLED=false`); GCP mode
  boots on the stub unless explicitly enabled.
