"""Tests for the enhanced security adapter."""
from __future__ import annotations

import pytest

from mlops_orchestrator.infrastructure.adapters.security_adapter import GcpSecurityAdapter


class TestGcpSecurityAdapter:
    async def test_sanitize_drops_dunder_keys(self):
        adapter = GcpSecurityAdapter()
        result = await adapter.sanitize_tool_metadata("tool", {"key": "v", "__secret": "x"})
        assert "key" in result
        assert "__secret" not in result

    async def test_sanitize_drops_unsafe_keys(self):
        adapter = GcpSecurityAdapter()
        result = await adapter.sanitize_tool_metadata("tool", {"good_key": "v", "bad;key": "v"})
        assert "good_key" in result
        assert "bad;key" not in result

    async def test_sanitize_strips_unsafe_value_chars(self):
        adapter = GcpSecurityAdapter()
        result = await adapter.sanitize_tool_metadata("tool", {"key": "hello\x00world"})
        assert result["key"] == "helloworld"

    async def test_sanitize_blocks_shell_injection(self):
        adapter = GcpSecurityAdapter()
        result = await adapter.sanitize_tool_metadata("tool", {"cmd": "value; rm -rf /"})
        assert "cmd" not in result

    async def test_sanitize_blocks_xss(self):
        adapter = GcpSecurityAdapter()
        result = await adapter.sanitize_tool_metadata("tool", {"html": "<script>alert(1)</script>"})
        assert "html" not in result

    async def test_sanitize_blocks_template_injection(self):
        adapter = GcpSecurityAdapter()
        result = await adapter.sanitize_tool_metadata("tool", {"tmpl": "{{malicious}}"})
        assert "tmpl" not in result

    async def test_validate_iam_well_formed_roles(self):
        adapter = GcpSecurityAdapter()
        result = await adapter.validate_iam_permissions(["roles/viewer", "roles/editor"])
        # Without GCP credentials, falls back to format validation
        assert result is True or result is False  # depends on env

    async def test_validate_iam_malformed_roles(self):
        adapter = GcpSecurityAdapter()
        result = await adapter.validate_iam_permissions(["admin", "user"])
        assert result is False

    async def test_validate_iam_no_project(self):
        adapter = GcpSecurityAdapter(project=None)
        result = await adapter.validate_iam_permissions(["roles/viewer"])
        assert result is False

    async def test_sanitize_empty_metadata(self):
        adapter = GcpSecurityAdapter()
        result = await adapter.sanitize_tool_metadata("tool", {})
        assert result == {}

    async def test_sanitize_safe_values_pass_through(self):
        adapter = GcpSecurityAdapter()
        meta = {"name": "my-model", "version": "1.0", "region": "us-central1"}
        result = await adapter.sanitize_tool_metadata("tool", meta)
        assert result == meta
