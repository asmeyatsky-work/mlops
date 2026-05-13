"""Structured JSON logging with correlation id propagation and PII redaction."""
from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, UTC

from mlops_orchestrator.infrastructure.observability.tracing import current_correlation_id


# Patterns matched against formatted message and exception text; matches are
# replaced with ``<redacted>``. Conservative — only secrets/PII shapes.
_REDACTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)authorization\s*[:=]\s*(bearer\s+)?[A-Za-z0-9._\-]{8,}"),
    re.compile(r"(?i)api[_-]?key\s*[:=]\s*[A-Za-z0-9._\-]{8,}"),
    re.compile(r"(?i)password\s*[:=]\s*\S+"),
    re.compile(r"(?i)secret\s*[:=]\s*\S+"),
    re.compile(r"(?i)token\s*[:=]\s*[A-Za-z0-9._\-]{8,}"),
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),  # email
    re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"),  # JWT
)


def redact(text: str) -> str:
    """Replace known-sensitive substrings with ``<redacted>``."""
    if not text:
        return text
    out = text
    for pattern in _REDACTION_PATTERNS:
        out = pattern.sub("<redacted>", out)
    return out


class JSONFormatter(logging.Formatter):
    """Single-line JSON formatter that injects correlation id and redacts PII."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact(record.getMessage()),
        }
        cid = current_correlation_id()
        if cid:
            log_entry["correlation_id"] = cid
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = redact(self.formatException(record.exc_info))
        for attr in ("action", "resource_id", "duration_ms", "trace_id", "agent_id"):
            if hasattr(record, attr):
                log_entry[attr] = getattr(record, attr)
        return json.dumps(log_entry)


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    """Configure root logger with structured JSON or plain text output."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stderr)
    if json_output:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s — %(message)s")
        )
    root.addHandler(handler)
