"""Starlette middleware that gates the MCP SSE transport on API key / JWT.

FastMCP's own auth machinery is OAuth-shaped (token verifier + scopes).
Our deployment uses simple API keys and HS256 JWTs, so we slot in here as
a plain ASGI middleware before the SSE handler.

Header conventions:
- ``X-API-Key: <key>``                       — primary
- ``Authorization: Bearer <jwt>``            — JWT (HS256)
- ``Authorization: <key>``                   — API key (legacy clients)

On success the request's ``Principal`` is set in the ``current_principal``
contextvar so ``enforce_tool_authz`` can act on it inside tool handlers.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Awaitable, Callable

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from mlops_orchestrator.infrastructure.auth.auth_context import (
    Principal,
    current_principal,
)
from mlops_orchestrator.infrastructure.auth.auth_middleware import AuthMiddleware

logger = logging.getLogger(__name__)


def _extract_credentials(headers: dict[bytes, bytes]) -> tuple[str, str]:
    """Return (raw_credential, kind_hint). Kind hint is 'jwt' or 'apikey'."""
    api_key_hdr = headers.get(b"x-api-key")
    if api_key_hdr:
        return api_key_hdr.decode(errors="replace"), "apikey"
    auth_hdr = headers.get(b"authorization")
    if auth_hdr:
        value = auth_hdr.decode(errors="replace")
        if value.startswith("Bearer "):
            return value, "jwt"
        return value, "apikey"
    return "", ""


class SSEAuthMiddleware:
    """Validates credentials on every SSE / message POST request."""

    def __init__(self, app: ASGIApp, auth: AuthMiddleware) -> None:
        self._app = app
        self._auth = auth

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        credential, kind = _extract_credentials(headers)

        if not credential:
            await self._deny(scope, receive, send, "missing credential")
            return

        if kind == "jwt":
            jwt_token = credential[7:]
            claims = self._auth.validate_jwt(jwt_token)
            if claims is None:
                await self._deny(scope, receive, send, "invalid jwt")
                return
            subject = str(claims.get("sub", "jwt-user"))
            permitted = self._auth._config.allowed_tools.get(subject, ())  # type: ignore[attr-defined]
            principal = Principal(
                subject=subject,
                kind="user",
                permitted_tools=frozenset(permitted),
            )
        else:
            if not self._auth.validate_api_key(credential):
                await self._deny(scope, receive, send, "invalid api key")
                return
            permitted = self._auth._config.allowed_tools.get(credential, ())  # type: ignore[attr-defined]
            fingerprint = hashlib.sha256(credential.encode()).hexdigest()[:12]
            principal = Principal(
                subject=f"key:{fingerprint}",  # never log the raw key
                kind="user",
                permitted_tools=frozenset(permitted),
            )

        ctx_token = current_principal.set(principal)
        try:
            await self._app(scope, receive, send)
        finally:
            current_principal.reset(ctx_token)

    async def _deny(self, scope: Scope, receive: Receive, send: Send, reason: str) -> None:
        logger.warning("auth denied: %s", reason)
        response = JSONResponse(
            {"error": "unauthorized", "reason": reason}, status_code=401
        )
        await response(scope, receive, send)
