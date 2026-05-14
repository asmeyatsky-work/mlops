---
marp: true
theme: default
paginate: false
size: A4
style: |
  section {
    padding: 28px 34px;
    font-size: 13px;
    line-height: 1.42;
    color: #15233f;
    background: #ffffff;
    font-family: -apple-system, "Segoe UI", system-ui, Roboto, sans-serif;
  }
  h1 { color: #1a73e8; margin: 0; font-size: 26px; }
  h2 {
    color: #1a73e8;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 0 0 6px;
    border-bottom: 1px solid #d2dbef;
    padding-bottom: 4px;
  }
  .tagline { color: #5b6788; font-size: 13px; margin-top: 4px; margin-bottom: 14px; }
  .row { display: flex; gap: 14px; margin-top: 10px; }
  .col { flex: 1; }
  .pillar {
    background: #f4f7ff;
    border-left: 3px solid #1a73e8;
    padding: 8px 10px;
    margin-bottom: 6px;
    border-radius: 0 4px 4px 0;
  }
  .pillar b { color: #102852; }
  .grid3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .stat {
    background: #0f1e3e;
    color: #e6ecff;
    padding: 8px 10px;
    border-radius: 6px;
  }
  .stat .v { font-size: 18px; font-weight: 700; color: #6aa6ff; }
  .stat .l { font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; color: #9bb0d8; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  table th, table td { padding: 4px 6px; text-align: left; }
  table th { background: #eef3ff; color: #102852; font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; }
  table td { border-bottom: 1px solid #e4eaf6; }
  code { background: #eef3ff; padding: 1px 4px; border-radius: 3px; font-size: 11px; }
  .footer {
    margin-top: 14px;
    border-top: 1px solid #d2dbef;
    padding-top: 8px;
    font-size: 11px;
    color: #5b6788;
    display: flex;
    justify-content: space-between;
  }
  .badge {
    display: inline-block;
    background: #e6f0ff;
    color: #1a73e8;
    padding: 1px 8px;
    border-radius: 999px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }
---

# Agentic MLOps Orchestrator
<div class="tagline">MCP-native, multi-agent control plane for the full ML lifecycle on GCP. <span class="badge">EU AI Act-aware</span> <span class="badge">FinOps-built-in</span> <span class="badge">production-ready</span></div>

<div class="row">
<div class="col">

## The problem
Teams running ML on GCP repeatedly rebuild the same plumbing: manual handoffs between data eng, ML, and SRE; GPU idle waste invisible until the bill; silent model drift caught only after a KPI moves; mounting EU AI Act obligations; and an agent-first stack (MCP, autonomous LLMs) that needs a new control plane.

</div>
<div class="col">

## What it is
A single MCP server exposing the full ML lifecycle as tools and resources, orchestrated by a swarm of seven specialist agents across five coordination patterns. Domain-driven Python, ~10K LOC, 504 tests, 90% coverage. Open source, GCP-native, ships with stub adapters for cost-free demos.

</div>
</div>

## Capabilities
<div class="grid3">
<div class="pillar"><b>Multi-agent swarm.</b> 7 specialists · 5 patterns (orchestrator-worker, swarm, hierarchical, mesh, pipeline). Per-agent RBAC on every tool dispatch.</div>
<div class="pillar"><b>EU AI Act gate.</b> PROHIBITED blocked; HIGH-risk requires complete model card + controls. Article 6/10/11/15 helpers built in.</div>
<div class="pillar"><b>FinOps.</b> Real BigQuery billing export; GPU idle detection; right-sizing + scheduling recommendations actioned by agents.</div>
<div class="pillar"><b>Self-healing.</b> Observe → analyze → decide → act loop. KS · χ² · KL · PSI drift; Slack/PagerDuty/email alerts; auto-retrain.</div>
<div class="pillar"><b>Resilience.</b> Exponential backoff + jitter, hard <code>asyncio.wait_for</code> timeouts, circuit breaker, lock-free MCP I/O.</div>
<div class="pillar"><b>Observability.</b> Correlation IDs in every log; optional OTel spans; PII/secret redaction; Cloud Logging audit trail.</div>
<div class="pillar"><b>Security.</b> SSE auth middleware (API key + JWT HS256); per-agent RBAC; DNS-rebinding guard; production-env safety checks.</div>
<div class="pillar"><b>Supply chain.</b> Pinned upper-bound deps, pip-audit, CycloneDX SBOM, Trivy scan, weekly Dependabot, multi-stage non-root Docker.</div>
<div class="pillar"><b>Compatibility.</b> Vertex AI · GKE · BigQuery · Cloud Logging · IAM. Stub adapters for local dev. Cloud Run / K8s ready.</div>
</div>

<div class="row">
<div class="col">

## Business outcomes

| Lever | Mechanism | Outcome |
|-------|-----------|---------|
| Time-to-prod | Session stitching · DAG parallelism | Weeks → days |
| Cost | GPU idle detection · right-sizing | $50K+/yr typical |
| Risk | Drift loop · alerting · auto-retrain | Catch decay pre-KPI |
| Regulation | Enforced (not documented) compliance | EU markets unlocked |
| Adoption | Auth · RBAC · audit · K8s probes | Platform-team ready |

</div>
<div class="col">

## Architecture

```
domain        value objects · entities · ports
application   commands · queries · DTOs · DAG · swarm
infrastructure GCP adapters · auth · OTel · config
presentation  MCP server · web demo · CLI · health
```

Zero GCP imports in domain. Every external system behind a Protocol port; stubs ⇄ real adapters by env flag.

</div>
</div>

<div class="row" style="margin-top:14px">
<div class="col">

## Try it live
**Browser demo:** four scripted scenarios — standard lifecycle, compliance block, drift + self-heal, FinOps optimization. Live narration from each agent.

`https://mlops-orchestrator-demo-946022635729.us-central1.run.app`

**MCP endpoint:** `/sse` with `X-API-Key`. Plug into Claude Desktop / Claude Code / Cursor.

</div>
<div class="col">

<div class="grid2">
<div class="stat"><div class="v">504</div><div class="l">tests passing</div></div>
<div class="stat"><div class="v">90%</div><div class="l">coverage</div></div>
<div class="stat"><div class="v">~10K</div><div class="l">lines of python</div></div>
<div class="stat"><div class="v">$0</div><div class="l">demo cost</div></div>
</div>

</div>
</div>

<div class="footer">
<div><b>Repo:</b> github.com/asmeyatsky-work/mlops</div>
<div><b>Stack:</b> Python 3.11+ · MCP · FastMCP · Starlette · Vertex AI · GKE · BigQuery · Cloud Run</div>
</div>
