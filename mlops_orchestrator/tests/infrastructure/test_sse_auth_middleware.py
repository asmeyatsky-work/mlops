"""Tests for the SSE ASGI auth middleware."""
from __future__ import annotations

import pytest

from mlops_orchestrator.infrastructure.auth.auth_context import (
    current_principal,
)
from mlops_orchestrator.infrastructure.auth.auth_middleware import (
    AuthConfig,
    AuthMiddleware,
)
from mlops_orchestrator.infrastructure.auth.sse_auth_middleware import (
    SSEAuthMiddleware,
)


class _Recorder:
    """Captures responses sent through the ASGI ``send`` callable."""

    def __init__(self) -> None:
        self.status: int | None = None
        self.body = b""
        self.app_called = False

    async def receive(self) -> dict:
        return {"type": "http.disconnect"}

    async def send(self, msg: dict) -> None:
        if msg["type"] == "http.response.start":
            self.status = msg["status"]
        elif msg["type"] == "http.response.body":
            self.body += msg.get("body", b"")


def _make_scope(headers: list[tuple[bytes, bytes]]) -> dict:
    return {
        "type": "http",
        "method": "GET",
        "path": "/sse",
        "headers": headers,
        "query_string": b"",
    }


@pytest.fixture
def auth() -> AuthMiddleware:
    return AuthMiddleware(
        AuthConfig(
            enabled=True,
            api_keys=("good-key",),
            jwt_secret="",
            allowed_tools={"good-key": ("create_dataset",)},
        )
    )


class TestSSEAuthMiddleware:
    async def test_missing_credential_returns_401(self, auth):
        async def downstream(scope, receive, send):
            raise AssertionError("downstream should not run")

        mw = SSEAuthMiddleware(downstream, auth)
        rec = _Recorder()
        await mw(_make_scope([]), rec.receive, rec.send)
        assert rec.status == 401
        assert b"missing credential" in rec.body

    async def test_invalid_api_key_returns_401(self, auth):
        async def downstream(scope, receive, send):
            raise AssertionError("downstream should not run")

        mw = SSEAuthMiddleware(downstream, auth)
        rec = _Recorder()
        await mw(
            _make_scope([(b"x-api-key", b"nope")]), rec.receive, rec.send
        )
        assert rec.status == 401

    async def test_valid_key_sets_principal_and_calls_downstream(self, auth):
        captured: dict = {}

        async def downstream(scope, receive, send):
            captured["principal"] = current_principal.get()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        mw = SSEAuthMiddleware(downstream, auth)
        rec = _Recorder()
        await mw(
            _make_scope([(b"x-api-key", b"good-key")]), rec.receive, rec.send
        )
        assert rec.status == 200
        principal = captured["principal"]
        assert principal.kind == "user"
        assert "create_dataset" in principal.permitted_tools
        assert "good-key" not in principal.subject  # fingerprinted, not raw

    async def test_principal_cleared_after_request(self, auth):
        async def downstream(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        mw = SSEAuthMiddleware(downstream, auth)
        rec = _Recorder()
        await mw(
            _make_scope([(b"x-api-key", b"good-key")]), rec.receive, rec.send
        )
        # outside the middleware scope, principal reverts to anonymous default
        assert current_principal.get().kind == "anonymous"

    async def test_authorization_header_with_raw_key_also_works(self, auth):
        captured = {}

        async def downstream(scope, receive, send):
            captured["principal"] = current_principal.get()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        mw = SSEAuthMiddleware(downstream, auth)
        rec = _Recorder()
        await mw(
            _make_scope([(b"authorization", b"good-key")]), rec.receive, rec.send
        )
        assert rec.status == 200
        assert captured["principal"].kind == "user"

    async def test_lifespan_scope_passes_through(self, auth):
        called = {}

        async def downstream(scope, receive, send):
            called["yes"] = True

        mw = SSEAuthMiddleware(downstream, auth)
        rec = _Recorder()
        await mw({"type": "lifespan"}, rec.receive, rec.send)
        assert called.get("yes") is True
