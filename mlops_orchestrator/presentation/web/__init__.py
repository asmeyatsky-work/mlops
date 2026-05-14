"""Browser-facing demo web UI for the MLOps Orchestrator.

Distinct from the MCP SSE transport: this is a normal Starlette app that
drives the existing commands via the same DependencyContainer and streams
events to the browser. The MCP server still lives at /sse; the demo lives
at /, /demo, /api/*.
"""
