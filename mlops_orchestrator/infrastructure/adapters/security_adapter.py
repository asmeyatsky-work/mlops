from __future__ import annotations
import re

# Allowlist approach: only permit safe characters in keys and values
_SAFE_PATTERN = re.compile(r'^[a-zA-Z0-9_.\-/: @,=]+$')


class GcpSecurityAdapter:
    """
    Security adapter for metadata sanitization and IAM validation.
    Implements SecurityPort.
    """

    async def sanitize_tool_metadata(
        self, tool_name: str, metadata: dict[str, str]
    ) -> dict[str, str]:
        """Sanitize both keys and values using an allowlist approach."""
        sanitized: dict[str, str] = {}
        for key, value in metadata.items():
            if key.startswith("__"):
                continue
            if not _SAFE_PATTERN.match(key):
                continue
            if not _SAFE_PATTERN.match(value):
                value = re.sub(r'[^a-zA-Z0-9_.\-/: @,=]', '', value)
            sanitized[key] = value
        return sanitized

    async def validate_iam_permissions(self, required_roles: list[str]) -> bool:
        # In production, would call resourcemanager API to check IAM bindings
        # For now, validates that required roles are well-formed
        valid_prefixes = ("roles/", "projects/")
        return all(
            any(role.startswith(p) for p in valid_prefixes)
            for role in required_roles
        )
