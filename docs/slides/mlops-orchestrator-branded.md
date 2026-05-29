---
marp: true
theme: default
paginate: true
size: 16:9
header: "MLOps Orchestrator"
footer: "Agentic MLOps on GCP · Confidential"
style: |
  @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  :root {
    --brand:      #0064FF;
    --brand-2:    #4285f4;
    --navy:       #002659;
    --navy-deep:  #05162e;
    --steel:      #456085;
    --ink:        #212121;
    --ink-soft:   #595959;
    --muted:      #7d7d7d;
    --line:       #d2e3fc;
    --tint:       #e7f0fe;
    --tint-2:     #e2eeff;
    --tint-3:     #e8f0fe;
    --sky:        #79adff;
    --sky-soft:   #a1c6ff;
  }

  section {
    font-family: 'Poppins', sans-serif;
    background: #ffffff;
    color: var(--ink);
    font-size: 22px;
    font-weight: 300;
    padding: 56px 64px;
  }
  section * { font-family: 'Poppins', sans-serif; }

  h1 { color: var(--brand); font-weight: 600; font-size: 46px; letter-spacing: -0.5px; }
  h2 {
    color: var(--navy); font-weight: 600; font-size: 32px;
    border-bottom: 2px solid var(--line); padding-bottom: 8px; letter-spacing: -0.3px;
  }
  h3 { color: var(--steel); font-weight: 500; }
  strong { color: var(--navy); font-weight: 600; }
  a { color: var(--brand); text-decoration: none; }
  ul { line-height: 1.5; }
  li::marker { color: var(--brand); }

  header { color: var(--muted); font-weight: 500; font-size: 13px; }
  footer { color: var(--muted); font-weight: 400; font-size: 12px; }
  section::after { color: var(--muted); font-weight: 500; }

  code {
    font-family: 'JetBrains Mono', monospace;
    background: var(--tint); color: var(--navy);
    padding: 1px 6px; border-radius: 4px; font-size: 0.82em;
  }
  pre {
    background: var(--navy-deep); border-radius: 10px; padding: 18px 20px;
  }
  pre code { background: transparent; color: #e2eeff; font-size: 0.78em; }

  table { font-size: 18px; border-collapse: collapse; width: 100%; }
  th { background: var(--navy); color: #ffffff; font-weight: 500; text-align: left; padding: 9px 12px; }
  td { padding: 8px 12px; border-bottom: 1px solid var(--line); color: var(--ink-soft); }
  tr:nth-child(even) td { background: var(--tint-3); }

  /* utility components */
  .chip {
    display: inline-block; background: var(--tint-2); color: var(--navy);
    border-radius: 999px; padding: 4px 14px; font-size: 14px; font-weight: 500;
    margin: 2px 4px;
  }
  .lead { color: var(--ink-soft); font-weight: 300; font-size: 24px; }
  .cards { display: flex; gap: 16px; margin-top: 10px; }
  .card {
    flex: 1; background: var(--tint-3); border: 1px solid var(--line);
    border-radius: 12px; padding: 18px 20px;
  }
  .card h4 { color: var(--brand); margin: 6px 0 8px; font-weight: 600; font-size: 19px; }
  .card p { color: var(--ink-soft); font-size: 16px; margin: 0; line-height: 1.45; }
  .kpi { color: var(--brand); font-weight: 700; font-size: 30px; }
  .muted { color: var(--muted); }

  section.title {
    background: #ffffff;
    display: flex; flex-direction: column; justify-content: center;
    border-left: 10px solid var(--brand); padding-left: 80px;
  }
  section.section-divider {
    display: flex; flex-direction: column; justify-content: center; align-items: flex-start;
  }
  section.section-divider h1 { font-size: 52px; }
---

<!-- _class: title -->
<!-- _paginate: false -->
<!-- _footer: "" -->

<svg width="64" height="64" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
  <rect x="6" y="6" width="22" height="22" rx="5" fill="#0064FF"/>
  <rect x="36" y="6" width="22" height="22" rx="5" fill="#79adff"/>
  <rect x="6" y="36" width="22" height="22" rx="5" fill="#456085"/>
  <rect x="36" y="36" width="22" height="22" rx="5" fill="#002659"/>
</svg>

# Agentic MLOps Orchestrator

### Multi-agent ML lifecycle management on GCP

<span class="chip">MCP-native</span> <span class="chip">Vertex AI · GKE</span> <span class="chip">Google ADK reasoning</span> <span class="chip">EU AI Act-aware</span>

---

## The problem

<p class="lead">Teams running ML on GCP rebuild the same plumbing — and the same risk.</p>

- **Manual handoffs** between data eng, ML, and SRE — weeks of pipeline work
- **GPU idle waste** invisible until the monthly bill arrives
- **Silent model drift** caught only after a business KPI moves
- **EU AI Act** adds risk classification, model cards, provenance, audit
- **Agent-first stacks** (MCP, autonomous LLMs) demand a new control plane

A control plane that is **agentic, governed, and cost-aware** is the gap.

---

## Solution at a glance

A single **MCP server** exposing the full ML lifecycle as *tools* + *resources*, orchestrated by a swarm of specialist agents — now with **LLM reasoning behind a port**.

<div class="cards">
  <div class="card"><h4>7 specialists</h4><p>Orchestrator, Architect, Data Eng, Validation, Deployment, FinOps, Security</p></div>
  <div class="card"><h4>5 patterns</h4><p>Orchestrator-Worker, Swarm, Hierarchical, Mesh, Pipeline</p></div>
  <div class="card"><h4>Reasoning seam</h4><p>Google ADK agents advise; deterministic commands actuate</p></div>
</div>

<br>

<span class="chip">DAG-aware parallelism</span> <span class="chip">Self-healing loop</span> <span class="chip">Enforced compliance gate</span> <span class="chip">538 tests · 90% coverage</span>

---

## Architecture — clean layers, ports & adapters

<svg width="980" height="220" viewBox="0 0 980 220" xmlns="http://www.w3.org/2000/svg">
  <g font-family="Poppins" font-size="17" font-weight="500">
    <rect x="0"   y="40" width="220" height="120" rx="12" fill="#e2eeff" stroke="#d2e3fc"/>
    <rect x="250" y="40" width="220" height="120" rx="12" fill="#c9deff" stroke="#a1c6ff"/>
    <rect x="500" y="40" width="220" height="120" rx="12" fill="#a1c6ff" stroke="#79adff"/>
    <rect x="750" y="40" width="220" height="120" rx="12" fill="#0064FF"/>
    <text x="110" y="34" fill="#002659" text-anchor="middle" font-weight="600">domain</text>
    <text x="360" y="34" fill="#002659" text-anchor="middle" font-weight="600">application</text>
    <text x="610" y="34" fill="#002659" text-anchor="middle" font-weight="600">infrastructure</text>
    <text x="860" y="34" fill="#002659" text-anchor="middle" font-weight="600">presentation</text>
    <text x="110" y="92"  fill="#05162e" text-anchor="middle" font-size="14">value objects</text>
    <text x="110" y="114" fill="#05162e" text-anchor="middle" font-size="14">entities · ports</text>
    <text x="110" y="136" fill="#05162e" text-anchor="middle" font-size="14">services · events</text>
    <text x="360" y="92"  fill="#05162e" text-anchor="middle" font-size="14">commands · queries</text>
    <text x="360" y="114" fill="#05162e" text-anchor="middle" font-size="14">orchestration</text>
    <text x="360" y="136" fill="#05162e" text-anchor="middle" font-size="14">session state</text>
    <text x="610" y="92"  fill="#05162e" text-anchor="middle" font-size="14">GCP + ADK adapters</text>
    <text x="610" y="114" fill="#05162e" text-anchor="middle" font-size="14">auth · observability</text>
    <text x="610" y="136" fill="#05162e" text-anchor="middle" font-size="14">config · stubs</text>
    <text x="860" y="98"  fill="#ffffff" text-anchor="middle" font-size="14">MCP server</text>
    <text x="860" y="120" fill="#ffffff" text-anchor="middle" font-size="14">CLI · health API</text>
    <path d="M250 100 L220 100 M500 100 L470 100 M750 100 L720 100" stroke="#456085" stroke-width="2" marker-end="url(#a)"/>
    <defs><marker id="a" markerWidth="9" markerHeight="9" refX="6" refY="4.5" orient="auto"><path d="M0 0 L7 4.5 L0 9 z" fill="#456085"/></marker></defs>
  </g>
</svg>

- Domain layer has **zero SDK imports** — pure logic (`domain ← application ← infrastructure ← presentation`)
- Every external system behind a **port**; stub adapters run unit tests at **zero cloud cost**
- Composition root wires Vertex AI / GKE / BigQuery — **and the new reasoning adapter**

---

## Multi-agent swarm + DAG orchestration

<svg width="940" height="250" viewBox="0 0 940 250" xmlns="http://www.w3.org/2000/svg">
  <g font-family="Poppins" font-size="15">
    <rect x="20" y="86" width="150" height="56" rx="10" fill="#0064FF"/>
    <text x="95" y="119" fill="#ffffff" text-anchor="middle" font-weight="600">Orchestrator</text>
    <g fill="#e8f0fe" stroke="#a1c6ff">
      <rect x="320" y="6"   width="220" height="34" rx="8"/>
      <rect x="320" y="46"  width="220" height="34" rx="8"/>
      <rect x="320" y="86"  width="220" height="34" rx="8"/>
      <rect x="320" y="126" width="220" height="34" rx="8"/>
      <rect x="320" y="166" width="220" height="34" rx="8"/>
      <rect x="320" y="206" width="220" height="34" rx="8" fill="#e8f0fe"/>
    </g>
    <g fill="#002659" font-size="14" font-weight="500">
      <text x="334" y="28">Data Engineer · BQ source</text>
      <text x="334" y="68">Architect · model design</text>
      <text x="334" y="108">Validation · drift checks</text>
      <text x="334" y="148">Deployment · Vertex / GKE</text>
      <text x="334" y="188">FinOps · billing export</text>
      <text x="334" y="228">Security · IAM · metadata</text>
    </g>
    <path d="M170 114 C 250 114, 250 23, 320 23
             M170 114 C 250 114, 250 63, 320 63
             M170 114 C 250 114, 250 103, 320 103
             M170 114 C 250 114, 250 143, 320 143
             M170 114 C 250 114, 250 183, 320 183
             M170 114 C 250 114, 250 223, 320 223"
          stroke="#79adff" stroke-width="2" fill="none"/>
  </g>
</svg>

- Agents are **frozen dataclasses** carrying a `permitted_tools` least-privilege allowlist
- DAG orchestrator parallelizes independent steps via `asyncio.gather`
- Whole-word capability matching **defeats prompt-injection** on routing

---

<!-- _class: section-divider -->
<!-- _footer: "" -->

<svg width="56" height="56" viewBox="0 0 56 56" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="28" cy="28" r="26" fill="#e2eeff"/>
  <path d="M18 30 l7 7 l14 -16" stroke="#0064FF" stroke-width="4" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
</svg>

# The ADK upgrade

### From static role DTOs to real reasoning — without losing determinism

<span class="muted">Google Agent Development Kit · Gemini · advisory-only, schema-validated</span>

---

## Why the upgrade — DTOs were scaffolding

The seven roles existed only as **immutable DTOs**. The `SwarmCoordinator` took a `task_executor` callback that had **no production implementation** — agency was wiring, never connected to a reasoning engine.

<div class="cards">
  <div class="card">
    <h4>Before</h4>
    <p>Agent = role + capabilities + tool allowlist. No reasoning. Coordinator executor existed only in tests.</p>
  </div>
  <div class="card">
    <h4>After</h4>
    <p>Reasoning roles think via Google ADK behind a port. Side-effecting roles stay deterministic. The real executor finally ships.</p>
  </div>
</div>

<br>

> The question was never *"convert all seven."* It was: **where does LLM agency add value — and where is determinism the correct, safer design?**

---

## The split — reason where it helps, stay deterministic where it matters

| Role | Verdict | Rationale |
|------|---------|-----------|
| **Orchestrator** | → ADK | NL goal → task decomposition, routing, aggregation |
| **Architect** | → ADK *(advisory)* | Region / infra / data-flow recommendations |
| **FinOps** | → ADK *(advisory)* | Interprets billing → right-sizing, spot, idle cuts |
| **Validation** | → ADK *(advisory)* | Test selection, bias interpretation |
| **Data Engineer** | → ADK *(hybrid)* | SQL/ETL planning reasons; dataset creation stays deterministic |
| **Deployment** | **Stay DTO** | Irreversible side effects → typed command only |
| **Security** | **Stay DTO** | Policy enforcement must be deterministic, never an LLM |

---

## The seam — Agent DTO → port → adapter

<svg width="960" height="250" viewBox="0 0 960 250" xmlns="http://www.w3.org/2000/svg">
  <g font-family="Poppins" font-size="14" font-weight="500">
    <!-- domain -->
    <rect x="10" y="95" width="150" height="60" rx="10" fill="#e2eeff" stroke="#d2e3fc"/>
    <text x="85" y="120" fill="#002659" text-anchor="middle" font-weight="600">Agent (DTO)</text>
    <text x="85" y="140" fill="#456085" text-anchor="middle" font-size="12">role · tools · RBAC</text>
    <!-- port -->
    <rect x="210" y="95" width="180" height="60" rx="10" fill="#0064FF"/>
    <text x="300" y="120" fill="#ffffff" text-anchor="middle" font-weight="600">ReasoningAgentPort</text>
    <text x="300" y="140" fill="#c9deff" text-anchor="middle" font-size="12">decompose · advise</text>
    <!-- adapters -->
    <rect x="450" y="20"  width="220" height="60" rx="10" fill="#e8f0fe" stroke="#a1c6ff"/>
    <text x="560" y="45"  fill="#002659" text-anchor="middle" font-weight="600">StubReasoningAdapter</text>
    <text x="560" y="65"  fill="#456085" text-anchor="middle" font-size="12">deterministic · tested · default</text>
    <rect x="450" y="170" width="220" height="60" rx="10" fill="#c9deff" stroke="#79adff"/>
    <text x="560" y="195" fill="#002659" text-anchor="middle" font-weight="600">ADKReasoningAdapter</text>
    <text x="560" y="215" fill="#456085" text-anchor="middle" font-size="12">Gemini · schema-validated</text>
    <!-- gemini -->
    <rect x="730" y="170" width="160" height="60" rx="10" fill="#4285f4"/>
    <text x="810" y="205" fill="#ffffff" text-anchor="middle" font-weight="600">Gemini</text>
    <!-- deterministic path -->
    <rect x="730" y="20"  width="160" height="60" rx="10" fill="#002659"/>
    <text x="810" y="45"  fill="#ffffff" text-anchor="middle" font-weight="600">Typed commands</text>
    <text x="810" y="65"  fill="#a1c6ff" text-anchor="middle" font-size="12">+ compliance gate</text>
    <!-- arrows -->
    <g stroke-width="2" fill="none">
      <path d="M160 125 H210" stroke="#456085" marker-end="url(#b)"/>
      <path d="M390 120 C 420 120, 420 50, 450 50"  stroke="#79adff" marker-end="url(#b)"/>
      <path d="M390 130 C 420 130, 420 200, 450 200" stroke="#79adff" marker-end="url(#b)"/>
      <path d="M670 200 H730" stroke="#456085" marker-end="url(#b)"/>
    </g>
    <text x="300" y="185" fill="#7d7d7d" text-anchor="middle" font-size="12">Deployment / Security bypass the LLM →</text>
    <path d="M300 188 C 560 250, 700 120, 730 60" stroke="#bdbdbd" stroke-width="2" fill="none" stroke-dasharray="4 4" marker-end="url(#b)"/>
    <defs><marker id="b" markerWidth="9" markerHeight="9" refX="6" refY="4.5" orient="auto"><path d="M0 0 L7 4.5 L0 9 z" fill="#456085"/></marker></defs>
  </g>
</svg>

- **`Agent` DTO is unchanged** — it stays the role / capability / least-privilege contract
- Domain & application import **no SDK**; `google-adk` is confined to one adapter (`[adk]` extra)

---

## Safety by construction — advisory, validated, observed

<div class="cards">
  <div class="card"><h4>Advisory only</h4><p>Reasoning returns <code>Plan</code> / <code>Recommendation</code> value objects. The LLM never actuates a deploy or a security decision.</p></div>
  <div class="card"><h4>Schema-validated</h4><p>Gemini JSON is parsed through Pydantic and mapped to immutable domain objects before it can influence state.</p></div>
  <div class="card"><h4>Bounded & logged</h4><p>Hard <code>asyncio.wait_for</code> timeout; per-call model id, prompt hash, tokens, latency on the correlation id.</p></div>
</div>

<br>

- Aligns with **Architectural Rules §4**: *AI output that mutates state must be validated against an explicit schema first*
- Disabled by default (`MLOPS_REASONING_ENABLED=false`) → stub path; flip on for live Gemini
- New code **100% covered**; ADR-0001 records the `google-adk` stack decision

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
| Resource | `mlops://session` · `mlops://jobs/{id}` · `mlops://costs/{p}` · `mlops://models/{m}` | Observable state |

**Session-state stitching** = the agent never plumbs resource IDs by hand.

---

## Resilience & observability

<div class="cards">
  <div class="card">
    <h4>Resilience</h4>
    <p>Retry with backoff + full jitter · hard per-attempt timeout · circuit breaker (CLOSED→OPEN→HALF_OPEN) · lock-free I/O so concurrent tool calls don't serialize.</p>
  </div>
  <div class="card">
    <h4>Observability</h4>
    <p>Correlation ID per call via <code>contextvars</code> · OpenTelemetry spans · structured JSON logs with PII redaction · Cloud Logging audit trail · K8s liveness / readiness / startup probes.</p>
  </div>
</div>

<br>

- Every external call — GCP **and** Gemini — is bounded; no unbounded waits
- `JobStatusQuery.poll_until_complete` honours its declared timeout even if Vertex hangs

---

## Security & AI safety

- **SSE auth middleware** (ASGI) — `X-API-Key` or `Authorization: Bearer <JWT HS256>`; bad creds → 401 before the MCP handler
- **Per-agent RBAC** — `enforce_tool_authz` at every dispatch checks the active `Principal.permitted_tools`
- **DNS-rebinding guard** — `MLOPS_ALLOWED_HOSTS` whitelists the platform hostname
- **Production guard** — `MLOPS_ENVIRONMENT=production` refuses to boot without auth, compliance gate, and real adapters
- **Reasoning is sandboxed by design** — advisory output, schema-validated, deterministic actuation only
- **Supply chain** — pinned upper bounds, pip-audit, CycloneDX SBOM, Trivy scan, Dependabot, multi-stage non-root image

---

## EU AI Act — enforced, not documented

`ComplianceGateService` runs **before** every deploy:

<div class="cards">
  <div class="card"><h4>PROHIBITED</h4><p>Blocked, always.</p></div>
  <div class="card"><h4>HIGH-risk</h4><p>Complete ModelCard (Art. 11) + non-empty required controls.</p></div>
  <div class="card"><h4>LIMITED</h4><p>Must record a transparency disclosure.</p></div>
  <div class="card"><h4>MINIMAL</h4><p>Allowed.</p></div>
</div>

<br>

Pure-domain helpers cover Article 6 (risk classification), Article 10 (data governance), Article 15 (accuracy + robustness).

> A high-risk model with no card never reaches a Vertex Endpoint — and no reasoning agent can override that.

---

## FinOps — real costs, not stubs

<div class="cards">
  <div class="card"><h4>BigQuery billing export</h4><p>Actual project spend with per-resource breakdown.</p></div>
  <div class="card"><h4>GPU idle detection</h4><p>Surfaces the biggest line item in most ML budgets.</p></div>
  <div class="card"><h4>Recommendations</h4><p>Concrete right-sizing actions — now reasoned over by the FinOps ADK agent.</p></div>
</div>

<br>

Exposed via the `mlops://costs/{project_id}` resource → agents reason about cost, then a deterministic command applies the change.

---

## Deployment — local build, direct to Cloud Run

```bash
# Build locally for Cloud Run (linux/amd64), push to Artifact Registry
IMG=us-central1-docker.pkg.dev/$PROJECT/demo/mlops-orchestrator:demo
docker buildx build --platform linux/amd64 -t $IMG --push .

# Deploy directly — no Cloud Build; CI stays on GitHub Actions
gcloud run deploy mlops-orchestrator-demo \
  --image=$IMG --region=us-central1 \
  --no-allow-unauthenticated --port=8000 --cpu=1 --memory=512Mi \
  --min-instances=0 --max-instances=2
```

<span class="chip">Scale-to-zero</span> <span class="chip">Stub adapters · ~$0</span> <span class="chip">API-key + IAM gated</span> <span class="chip">GitHub Actions CI</span>

Live now: `mlops-orchestrator-demo-946022635729.us-central1.run.app`

---

## Business value

| Lever | Mechanism | Outcome |
|-------|-----------|---------|
| **Time-to-prod** | Multi-agent + DAG · session stitching · ADK planning | Weeks → days for new pipelines |
| **Cost** | BQ billing · GPU idle · reasoned recommendations | Direct cut on the biggest line item |
| **Risk** | Drift + self-healing + alerting | Catch decay before KPIs move |
| **Regulation** | Enforced EU AI Act gate | Sell into EU-regulated markets |
| **Trust in AI** | Advisory, schema-validated reasoning | Autonomy without losing control |

---

## What's next

- Persistent governance backend (BigQuery / dedicated DB) behind `ModelGovernancePort`
- Live ADK rollout — Gemini-backed Orchestrator decomposition in production
- OpenTelemetry exporter wired to Cloud Trace by default
- Cloud Run job + Vertex Pipelines templates for non-MCP-aware clients
- SLSA Level 3 release pipeline (keyless OIDC signing + cosign)

---

<!-- _class: title -->
<!-- _paginate: false -->
<!-- _footer: "" -->

# Thank you

**Repo** · `github.com/asmeyatsky-work/mlops`
**Demo** · `mlops-orchestrator-demo-946022635729.us-central1.run.app`
**Stack** · Python 3.11+ · MCP · Google ADK · Vertex AI · GKE · BigQuery · Cloud Run

<span class="muted">Questions?</span>
