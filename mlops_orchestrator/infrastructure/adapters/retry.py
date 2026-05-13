"""Retry, circuit breaker, and hard-timeout utilities for transient GCP failures."""
from __future__ import annotations

import asyncio
import functools
import logging
import random
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

_TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class CircuitBreakerOpen(RuntimeError):
    """Raised when the circuit breaker is open and rejects a call fast."""


class CircuitBreaker:
    """Simple async-friendly circuit breaker.

    States:
      - CLOSED: calls flow through; failures increment a counter.
      - OPEN: calls are rejected fast for ``reset_seconds``.
      - HALF_OPEN: a single probe is allowed; success closes, failure re-opens.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_seconds: float = 30.0,
        name: str = "default",
    ) -> None:
        self._failure_threshold = failure_threshold
        self._reset_seconds = reset_seconds
        self._failures = 0
        self._opened_at: float | None = None
        self._half_open_in_flight = False
        self._lock = asyncio.Lock()
        self._name = name

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "CLOSED"
        if time.monotonic() - self._opened_at >= self._reset_seconds:
            return "HALF_OPEN"
        return "OPEN"

    async def _before_call(self) -> None:
        async with self._lock:
            state = self.state
            if state == "OPEN":
                raise CircuitBreakerOpen(f"circuit '{self._name}' open; failing fast")
            if state == "HALF_OPEN":
                if self._half_open_in_flight:
                    raise CircuitBreakerOpen(
                        f"circuit '{self._name}' probe in flight"
                    )
                self._half_open_in_flight = True

    async def _on_success(self) -> None:
        async with self._lock:
            self._failures = 0
            self._opened_at = None
            self._half_open_in_flight = False

    async def _on_failure(self) -> None:
        async with self._lock:
            self._half_open_in_flight = False
            self._failures += 1
            if self._failures >= self._failure_threshold:
                self._opened_at = time.monotonic()
                logger.warning(
                    "circuit '%s' opened after %d failures",
                    self._name,
                    self._failures,
                )

    async def call(self, func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        await self._before_call()
        try:
            result = await func(*args, **kwargs)
        except Exception:
            await self._on_failure()
            raise
        await self._on_success()
        return result


def _is_transient(exc: Exception) -> bool:
    if hasattr(exc, "code"):
        return exc.code in _TRANSIENT_STATUS_CODES  # type: ignore[union-attr]
    if hasattr(exc, "status"):
        return exc.status in _TRANSIENT_STATUS_CODES  # type: ignore[union-attr]
    if isinstance(exc, (ConnectionError, TimeoutError, asyncio.TimeoutError)):
        return True
    return False


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    timeout_seconds: float | None = None,
    breaker: CircuitBreaker | None = None,
) -> Callable[[F], F]:
    """Retry with exponential-backoff + full jitter, optional hard timeout and breaker."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    async def _invoke() -> Any:
                        if breaker is not None:
                            return await breaker.call(func, *args, **kwargs)
                        return await func(*args, **kwargs)

                    if timeout_seconds is not None:
                        return await asyncio.wait_for(_invoke(), timeout_seconds)
                    return await _invoke()
                except CircuitBreakerOpen:
                    raise
                except Exception as exc:
                    last_exc = exc
                    if not _is_transient(exc) or attempt == max_attempts - 1:
                        raise
                    delay = min(max_delay, base_delay * (backoff_factor ** attempt))
                    jittered = random.uniform(0, delay)
                    logger.warning(
                        "Transient error in %s (attempt %d/%d), retrying in %.1fs: %s",
                        func.__qualname__,
                        attempt + 1,
                        max_attempts,
                        jittered,
                        exc,
                    )
                    await asyncio.sleep(jittered)
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator
