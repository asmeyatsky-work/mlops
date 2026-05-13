"""OpenTelemetry-compatible tracing with no-op fallback.

If `opentelemetry-api` is installed, spans are real OTel spans. Otherwise
this module provides a lightweight in-process span that still propagates a
correlation id via a contextvar so logs and audit records can correlate.
"""
from __future__ import annotations

import contextvars
import functools
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, TypeVar

logger = logging.getLogger(__name__)

correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=""
)

try:
    from opentelemetry import trace as _otel_trace
    _tracer = _otel_trace.get_tracer("mlops_orchestrator")
    _OTEL_AVAILABLE = True
except Exception:
    _tracer = None
    _OTEL_AVAILABLE = False


def new_correlation_id() -> str:
    return uuid.uuid4().hex


def current_correlation_id() -> str:
    cid = correlation_id_var.get()
    return cid or ""


@asynccontextmanager
async def start_span(
    name: str, attributes: dict[str, Any] | None = None
) -> AsyncIterator[None]:
    """Start a span. Generates a correlation id if none is set in this context."""
    cid = correlation_id_var.get()
    token = None
    if not cid:
        cid = new_correlation_id()
        token = correlation_id_var.set(cid)

    attrs = {"correlation_id": cid, **(attributes or {})}

    if _OTEL_AVAILABLE and _tracer is not None:
        with _tracer.start_as_current_span(name, attributes=attrs):
            try:
                yield
            finally:
                if token is not None:
                    correlation_id_var.reset(token)
    else:
        logger.debug("span.start name=%s attrs=%s", name, attrs)
        try:
            yield
        finally:
            logger.debug("span.end name=%s", name)
            if token is not None:
                correlation_id_var.reset(token)


F = TypeVar("F", bound=Callable[..., Any])


def traced(span_name: str | None = None) -> Callable[[F], F]:
    """Decorator that wraps an async function in a span."""

    def decorator(func: F) -> F:
        name = span_name or func.__qualname__

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with start_span(name):
                return await func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
