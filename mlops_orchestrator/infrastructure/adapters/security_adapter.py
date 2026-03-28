from __future__ import annotations
import re


# Known injection patterns for tool metadata sanitization
_INJECTION_PATTERNS = [
    re.compile(r"<\s*script", re.IGNORECASE),
    re.compile(r"javascript:", re.IGNORECASE),
    re.compile(r"\{\{.*\}\}"),  # template injection
    re.compile(r"__import__"),
    re.compile(r"eval\s*\("),
    re.compile(r"exec\s*\("),
    re.compile(r"os\.system"),
    re.compile(r"subprocess"),
    re.compile(r"\bsudo\b"),
    re.compile(r"rm\s+-rf"),
]


class GcpSecurityAdapter:
    """
    Security adapter for metadata sanitization and IAM validation.
    Implements SecurityPort.
    """

    async def sanitize_tool_metadata(
        self, tool_name: str, metadata: dict[str, str]
    ) -> dict[str, str]:
        sanitized: dict[str, str] = {}
        for key, value in metadata.items():
            if key.startswith("__"):
                continue
            clean_value = value
            for pattern in _INJECTION_PATTERNS:
                clean_value = pattern.sub("[SANITIZED]", clean_value)
            sanitized[key] = clean_value
        return sanitized

    async def validate_iam_permissions(self, required_roles: list[str]) -> bool:
        # In production, would call resourcemanager API to check IAM bindings
        # For now, validates that required roles are well-formed
        valid_prefixes = ("roles/", "projects/")
        return all(
            any(role.startswith(p) for p in valid_prefixes)
            for role in required_roles
        )
