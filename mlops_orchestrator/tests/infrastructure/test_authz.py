"""Tests for per-agent RBAC at MCP dispatch."""
from __future__ import annotations

import pytest

from mlops_orchestrator.infrastructure.auth.auth_context import (
    Principal,
    ToolAuthzError,
    current_principal,
    enforce_tool_authz,
)


class TestEnforceToolAuthz:
    def test_anonymous_default_passes(self):
        enforce_tool_authz("train_model")

    def test_agent_with_permission_passes(self):
        agent = Principal(
            subject="a-1", kind="agent",
            permitted_tools=frozenset({"train_model"}),
        )
        token = current_principal.set(agent)
        try:
            enforce_tool_authz("train_model")
        finally:
            current_principal.reset(token)

    def test_agent_without_permission_blocks(self):
        agent = Principal(
            subject="a-2", kind="agent",
            permitted_tools=frozenset({"train_model"}),
        )
        token = current_principal.set(agent)
        try:
            with pytest.raises(ToolAuthzError):
                enforce_tool_authz("deploy_to_vertex")
        finally:
            current_principal.reset(token)

    def test_user_with_explicit_allowlist_enforced(self):
        user = Principal(
            subject="user-1", kind="user",
            permitted_tools=frozenset({"create_dataset"}),
        )
        token = current_principal.set(user)
        try:
            enforce_tool_authz("create_dataset")
            with pytest.raises(ToolAuthzError):
                enforce_tool_authz("train_model")
        finally:
            current_principal.reset(token)
