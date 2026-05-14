"""Starlette app that serves the interactive demo UI + a workflow SSE feed.

Mounted alongside the MCP server in the same uvicorn process. Public —
no API key required (the MCP /sse endpoint stays gated).
"""
from __future__ import annotations

import json
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.responses import StreamingResponse
from starlette.routing import Route

from mlops_orchestrator.presentation.web.demo_runner import DEMO_AGENTS, DemoRunner


_INDEX_HTML = (Path(__file__).parent / "templates" / "index.html").read_text()


def build_demo_app(container) -> Starlette:
    """Construct the public-facing demo app."""
    runner = DemoRunner(container)

    async def index(request: Request) -> HTMLResponse:
        return HTMLResponse(_INDEX_HTML)

    async def agents(request: Request) -> JSONResponse:
        return JSONResponse(
            [
                {
                    "id": a.id[:8],
                    "role": a.role.value,
                    "capabilities": list(a.capabilities),
                    "permitted_tools": list(a.permitted_tools),
                }
                for a in DEMO_AGENTS
            ]
        )

    async def run_pipeline(request: Request) -> StreamingResponse:
        model_name = request.query_params.get("model", "demo-model")

        async def stream():
            try:
                async for event in runner.run(model_name=model_name):
                    yield f"event: {event.kind}\ndata: {json.dumps(event.to_dict())}\n\n"
            except Exception as exc:  # surface errors to the UI
                err = {"kind": "error", "error": str(exc)}
                yield f"event: error\ndata: {json.dumps(err)}\n\n"

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    async def healthz(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    return Starlette(
        debug=False,
        routes=[
            Route("/", index),
            Route("/demo", index),
            Route("/api/agents", agents),
            Route("/api/run", run_pipeline),
            Route("/healthz", healthz),
        ],
    )
