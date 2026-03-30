"""Authentication and authorization middleware for the MCP server."""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuthConfig:
    """Configuration for MCP server authentication."""
    enabled: bool = False
    api_keys: tuple[str, ...] = ()
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    jwt_issuer: str = "mlops-orchestrator"
    allowed_tools: dict[str, tuple[str, ...]] = field(default_factory=dict)


class AuthMiddleware:
    """API key and JWT authentication middleware.

    Supports:
    - API key validation (constant-time comparison)
    - JWT token validation (HS256)
    - Per-key tool-level authorization
    """

    def __init__(self, config: AuthConfig) -> None:
        self._config = config
        self._key_hashes = {
            self._hash_key(k): k for k in config.api_keys
        }

    @staticmethod
    def _hash_key(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()

    def validate_api_key(self, key: str) -> bool:
        """Validate an API key using constant-time comparison."""
        key_hash = self._hash_key(key)
        for stored_hash in self._key_hashes:
            if hmac.compare_digest(key_hash, stored_hash):
                return True
        return False

    def validate_jwt(self, token: str) -> dict | None:
        """Validate a JWT token. Returns claims dict or None."""
        if not self._config.jwt_secret:
            return None
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None

            # Decode header and payload (base64url)
            import base64

            def _b64decode(s: str) -> bytes:
                padding = 4 - len(s) % 4
                return base64.urlsafe_b64decode(s + "=" * padding)

            header = json.loads(_b64decode(parts[0]))
            payload = json.loads(_b64decode(parts[1]))

            # Verify algorithm
            if header.get("alg") != self._config.jwt_algorithm:
                return None

            # Verify signature
            signing_input = f"{parts[0]}.{parts[1]}".encode()
            expected_sig = hmac.new(
                self._config.jwt_secret.encode(),
                signing_input,
                hashlib.sha256,
            ).digest()
            actual_sig = _b64decode(parts[2])
            if not hmac.compare_digest(expected_sig, actual_sig):
                return None

            # Verify expiration
            if "exp" in payload and payload["exp"] < time.time():
                logger.warning("JWT token expired")
                return None

            # Verify issuer
            if self._config.jwt_issuer and payload.get("iss") != self._config.jwt_issuer:
                logger.warning("JWT issuer mismatch")
                return None

            return payload
        except Exception:
            logger.exception("JWT validation failed")
            return None

    def authorize_tool(self, api_key: str, tool_name: str) -> bool:
        """Check if an API key is authorized to use a specific tool.

        If no allowed_tools mapping exists for the key, all tools are permitted.
        """
        allowed = self._config.allowed_tools.get(api_key)
        if allowed is None:
            return True
        return tool_name in allowed

    def authenticate(self, credentials: str) -> bool:
        """Authenticate using either API key or JWT token.

        Accepts:
        - "Bearer <jwt_token>"
        - Raw API key
        """
        if not self._config.enabled:
            return True

        if not credentials:
            return False

        if credentials.startswith("Bearer "):
            token = credentials[7:]
            claims = self.validate_jwt(token)
            return claims is not None

        return self.validate_api_key(credentials)


class NoOpAuthMiddleware:
    """Auth middleware that always permits access. Used when auth is disabled."""

    def authenticate(self, credentials: str) -> bool:
        return True

    def authorize_tool(self, api_key: str, tool_name: str) -> bool:
        return True

    def validate_api_key(self, key: str) -> bool:
        return True

    def validate_jwt(self, token: str) -> dict | None:
        return {"sub": "anonymous"}
