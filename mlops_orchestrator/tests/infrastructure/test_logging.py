"""Tests for structured logging configuration."""
from __future__ import annotations

import json
import logging

from mlops_orchestrator.infrastructure.logging_config import JSONFormatter, configure_logging


class TestJSONFormatter:
    def test_formats_as_json(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello world", args=(), exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["message"] == "hello world"
        assert data["level"] == "INFO"
        assert data["logger"] == "test"
        assert "timestamp" in data

    def test_includes_exception(self):
        formatter = JSONFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys
            record = logging.LogRecord(
                name="test", level=logging.ERROR, pathname="", lineno=0,
                msg="error", args=(), exc_info=sys.exc_info(),
            )
        output = formatter.format(record)
        data = json.loads(output)
        assert "exception" in data
        assert "ValueError" in data["exception"]

    def test_includes_extra_fields(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="action", args=(), exc_info=None,
        )
        record.action = "create_dataset"  # type: ignore[attr-defined]
        record.resource_id = "rn-123"  # type: ignore[attr-defined]
        output = formatter.format(record)
        data = json.loads(output)
        assert data["action"] == "create_dataset"
        assert data["resource_id"] == "rn-123"


class TestConfigureLogging:
    def test_json_output(self):
        configure_logging(level="DEBUG", json_output=True)
        root = logging.getLogger()
        assert root.level == logging.DEBUG
        assert any(isinstance(h.formatter, JSONFormatter) for h in root.handlers)

    def test_plain_output(self):
        configure_logging(level="WARNING", json_output=False)
        root = logging.getLogger()
        assert root.level == logging.WARNING
        assert not any(isinstance(h.formatter, JSONFormatter) for h in root.handlers)
