"""Tests for tracing context, correlation id propagation, and PII redaction."""
from __future__ import annotations

import json
import logging

from mlops_orchestrator.infrastructure.logging_config import JSONFormatter, redact
from mlops_orchestrator.infrastructure.observability.tracing import (
    correlation_id_var,
    new_correlation_id,
    start_span,
)


def test_redact_strips_bearer_token():
    out = redact("Authorization: Bearer abcdef1234567890")
    assert "abcdef1234567890" not in out
    assert "<redacted>" in out


def test_redact_strips_email():
    out = redact("notify alice@example.com please")
    assert "alice@example.com" not in out


def test_redact_strips_jwt_shape():
    jwt = "eyJabcabcabc.eyJpYXQiOjEyMzQ.signaturesignature"
    out = redact(f"token={jwt}")
    assert jwt not in out


def test_json_formatter_includes_correlation_id():
    cid = "abc123"
    token = correlation_id_var.set(cid)
    try:
        rec = logging.LogRecord(
            name="x", level=logging.INFO, pathname="", lineno=0,
            msg="hello", args=(), exc_info=None,
        )
        out = json.loads(JSONFormatter().format(rec))
        assert out["correlation_id"] == cid
    finally:
        correlation_id_var.reset(token)


async def test_start_span_sets_correlation_id():
    async with start_span("test"):
        assert correlation_id_var.get() != ""


def test_new_correlation_id_is_unique():
    assert new_correlation_id() != new_correlation_id()
