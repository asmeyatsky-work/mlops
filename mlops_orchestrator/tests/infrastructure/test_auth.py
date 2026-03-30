"""Tests for authentication middleware."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

import pytest

from mlops_orchestrator.infrastructure.auth.auth_middleware import (
    AuthConfig,
    AuthMiddleware,
    NoOpAuthMiddleware,
)


def _make_jwt(payload: dict, secret: str, algorithm: str = "HS256") -> str:
    """Create a minimal JWT for testing."""
    header = base64.urlsafe_b64encode(
        json.dumps({"alg": algorithm, "typ": "JWT"}).encode()
    ).rstrip(b"=").decode()
    body = base64.urlsafe_b64encode(
        json.dumps(payload).encode()
    ).rstrip(b"=").decode()
    signing_input = f"{header}.{body}".encode()
    signature = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    sig_b64 = base64.urlsafe_b64encode(signature).rstrip(b"=").decode()
    return f"{header}.{body}.{sig_b64}"


class TestAuthMiddleware:
    def test_disabled_auth_always_passes(self):
        config = AuthConfig(enabled=False)
        auth = AuthMiddleware(config)
        assert auth.authenticate("") is True
        assert auth.authenticate("anything") is True

    def test_api_key_validation(self):
        config = AuthConfig(enabled=True, api_keys=("test-key-123",))
        auth = AuthMiddleware(config)
        assert auth.validate_api_key("test-key-123") is True
        assert auth.validate_api_key("wrong-key") is False

    def test_authenticate_with_api_key(self):
        config = AuthConfig(enabled=True, api_keys=("my-key",))
        auth = AuthMiddleware(config)
        assert auth.authenticate("my-key") is True
        assert auth.authenticate("bad-key") is False

    def test_empty_credentials_rejected(self):
        config = AuthConfig(enabled=True, api_keys=("key",))
        auth = AuthMiddleware(config)
        assert auth.authenticate("") is False

    def test_jwt_validation(self):
        secret = "test-secret-256"
        config = AuthConfig(
            enabled=True,
            jwt_secret=secret,
            jwt_issuer="mlops-orchestrator",
        )
        auth = AuthMiddleware(config)

        token = _make_jwt(
            {"sub": "user1", "iss": "mlops-orchestrator", "exp": time.time() + 3600},
            secret,
        )
        claims = auth.validate_jwt(token)
        assert claims is not None
        assert claims["sub"] == "user1"

    def test_jwt_expired(self):
        secret = "test-secret"
        config = AuthConfig(enabled=True, jwt_secret=secret, jwt_issuer="mlops-orchestrator")
        auth = AuthMiddleware(config)

        token = _make_jwt(
            {"sub": "user1", "iss": "mlops-orchestrator", "exp": time.time() - 100},
            secret,
        )
        assert auth.validate_jwt(token) is None

    def test_jwt_wrong_issuer(self):
        secret = "test-secret"
        config = AuthConfig(enabled=True, jwt_secret=secret, jwt_issuer="mlops-orchestrator")
        auth = AuthMiddleware(config)

        token = _make_jwt(
            {"sub": "user1", "iss": "wrong-issuer", "exp": time.time() + 3600},
            secret,
        )
        assert auth.validate_jwt(token) is None

    def test_jwt_bad_signature(self):
        config = AuthConfig(enabled=True, jwt_secret="real-secret", jwt_issuer="mlops-orchestrator")
        auth = AuthMiddleware(config)

        token = _make_jwt(
            {"sub": "user1", "iss": "mlops-orchestrator", "exp": time.time() + 3600},
            "wrong-secret",
        )
        assert auth.validate_jwt(token) is None

    def test_authenticate_with_bearer_token(self):
        secret = "test-secret"
        config = AuthConfig(
            enabled=True,
            jwt_secret=secret,
            jwt_issuer="mlops-orchestrator",
        )
        auth = AuthMiddleware(config)

        token = _make_jwt(
            {"sub": "user1", "iss": "mlops-orchestrator", "exp": time.time() + 3600},
            secret,
        )
        assert auth.authenticate(f"Bearer {token}") is True

    def test_tool_authorization_no_restrictions(self):
        config = AuthConfig(enabled=True, api_keys=("key-1",))
        auth = AuthMiddleware(config)
        assert auth.authorize_tool("key-1", "any_tool") is True

    def test_tool_authorization_with_restrictions(self):
        config = AuthConfig(
            enabled=True,
            api_keys=("key-1",),
            allowed_tools={"key-1": ("create_dataset", "train_model")},
        )
        auth = AuthMiddleware(config)
        assert auth.authorize_tool("key-1", "create_dataset") is True
        assert auth.authorize_tool("key-1", "deploy_to_vertex") is False

    def test_jwt_malformed_token(self):
        config = AuthConfig(enabled=True, jwt_secret="secret")
        auth = AuthMiddleware(config)
        assert auth.validate_jwt("not.a.valid.token.at.all") is None
        assert auth.validate_jwt("") is None
        assert auth.validate_jwt("single") is None

    def test_jwt_no_secret_configured(self):
        config = AuthConfig(enabled=True, jwt_secret="")
        auth = AuthMiddleware(config)
        assert auth.validate_jwt("any.token.here") is None


class TestNoOpAuthMiddleware:
    def test_always_authenticates(self):
        auth = NoOpAuthMiddleware()
        assert auth.authenticate("") is True
        assert auth.authenticate("anything") is True

    def test_always_authorizes_tools(self):
        auth = NoOpAuthMiddleware()
        assert auth.authorize_tool("", "any_tool") is True

    def test_validate_api_key(self):
        auth = NoOpAuthMiddleware()
        assert auth.validate_api_key("") is True

    def test_validate_jwt(self):
        auth = NoOpAuthMiddleware()
        assert auth.validate_jwt("") is not None
