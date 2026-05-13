"""Per-request authentication and authorization context for MCP tool dispatch.

A ``Principal`` represents whoever is invoking a tool — either a human via
API key/JWT or one of our specialist agents. The MCP server installs the
principal into ``current_principal`` (a contextvar) for the duration of a
tool call, and tool dispatch consults it via ``enforce_tool_authz``.

Tools enforce two checks at dispatch time:

1. Is the credential itself valid (handled by AuthMiddleware)?
2. Is THIS principal — possibly an agent with its own narrower
   ``permitted_tools`` — allowed to invoke THIS tool?

Without (2) the per-key ``allowed_tools`` map in AuthConfig is the only
gate, which means any caller holding a valid key can invoke any tool the
key permits. Agent-level RBAC further restricts this when an agent acts
on a user's behalf.
"""
from __future__ import annotations

import contextvars
from dataclasses import dataclass


@dataclass(frozen=True)
class Principal:
    """Identity active for the current MCP tool call."""

    subject: str  # API key fingerprint, JWT sub claim, or agent id
    kind: str  # "user" | "agent" | "anonymous"
    permitted_tools: frozenset[str] = frozenset()  # empty == all (only for users)


_ANONYMOUS = Principal(subject="anonymous", kind="anonymous")

current_principal: contextvars.ContextVar[Principal] = contextvars.ContextVar(
    "current_principal", default=_ANONYMOUS
)


class ToolAuthzError(PermissionError):
    """Raised when the current principal is not allowed to invoke a tool."""


def enforce_tool_authz(tool_name: str) -> None:
    """Raise ``ToolAuthzError`` if the current principal cannot use ``tool_name``.

    Users with no explicit allowlist pass through (legacy behaviour). Agents
    always have a non-empty allowlist enforced strictly.
    """
    principal = current_principal.get()
    if principal.kind == "agent":
        if tool_name not in principal.permitted_tools:
            raise ToolAuthzError(
                f"agent '{principal.subject}' is not permitted to invoke '{tool_name}'"
            )
        return
    if principal.permitted_tools and tool_name not in principal.permitted_tools:
        raise ToolAuthzError(
            f"principal '{principal.subject}' is not permitted to invoke '{tool_name}'"
        )
