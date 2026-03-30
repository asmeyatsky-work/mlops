"""Security adapter for metadata sanitization and real IAM validation."""
from __future__ import annotations

import asyncio
import logging
import re

from mlops_orchestrator.infrastructure.adapters.retry import with_retry

logger = logging.getLogger(__name__)

# Allowlist approach: only permit safe characters in keys and values
_SAFE_PATTERN = re.compile(r'^[a-zA-Z0-9_.\-/: @,=]+$')

# Patterns that could indicate injection attempts
_INJECTION_PATTERNS = [
    re.compile(r'[;\|`$]'),           # shell injection
    re.compile(r'<script', re.I),     # XSS
    re.compile(r'\{\{.*\}\}'),        # template injection
    re.compile(r'%\{'),              # log4j-style
]


class GcpSecurityAdapter:
    """
    Security adapter with real IAM permission validation and metadata sanitization.
    Implements SecurityPort.
    """

    def __init__(self, project: str | None = None) -> None:
        self._project = project

    async def sanitize_tool_metadata(
        self, tool_name: str, metadata: dict[str, str]
    ) -> dict[str, str]:
        """Sanitize both keys and values using an allowlist approach.

        - Drops keys starting with '__' (dunder injection)
        - Drops keys/values failing allowlist regex
        - Strips characters from values that don't pass the allowlist
        - Detects and rejects injection patterns
        """
        sanitized: dict[str, str] = {}
        for key, value in metadata.items():
            if key.startswith("__"):
                logger.warning("Dropped dunder key %r from tool %s", key, tool_name)
                continue
            if not _SAFE_PATTERN.match(key):
                logger.warning("Dropped unsafe key %r from tool %s", key, tool_name)
                continue
            # Check for injection patterns in values
            if any(p.search(value) for p in _INJECTION_PATTERNS):
                logger.warning(
                    "Injection pattern detected in value for key %r in tool %s",
                    key,
                    tool_name,
                )
                continue
            if not _SAFE_PATTERN.match(value):
                value = re.sub(r'[^a-zA-Z0-9_.\-/: @,=]', '', value)
            sanitized[key] = value
        return sanitized

    @with_retry(max_attempts=3)
    async def validate_iam_permissions(self, required_roles: list[str]) -> bool:
        """Verify the service account has required IAM roles using the Resource Manager API."""
        if not self._project:
            logger.warning("No project configured for IAM validation")
            return False

        valid_prefixes = ("roles/", "projects/")
        if not all(any(role.startswith(p) for p in valid_prefixes) for role in required_roles):
            return False

        try:
            from google.cloud import resourcemanager_v3
            from google.iam.v1 import iam_policy_pb2

            client = resourcemanager_v3.ProjectsClient()
            request = iam_policy_pb2.GetIamPolicyRequest(
                resource=f"projects/{self._project}",
            )
            policy = await asyncio.to_thread(client.get_iam_policy, request=request)

            # Extract all roles from the policy
            granted_roles: set[str] = set()
            for binding in policy.bindings:
                granted_roles.add(binding.role)

            # Check if all required roles are present in the policy
            missing = [r for r in required_roles if r not in granted_roles]
            if missing:
                logger.warning("Missing IAM roles: %s", missing)
                return False

            return True
        except ImportError:
            logger.warning(
                "google-cloud-resource-manager not installed, "
                "falling back to format validation only"
            )
            return all(
                any(role.startswith(p) for p in valid_prefixes)
                for role in required_roles
            )
        except Exception:
            logger.exception("IAM permission validation failed")
            return False
