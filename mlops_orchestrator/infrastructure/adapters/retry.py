"""Retry utility with exponential backoff and jitter for transient GCP failures."""
from __future__ import annotations

import asyncio
import functools
import logging
import random
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Google Cloud SDK exceptions that indicate transient failures
_TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def _is_transient(exc: Exception) -> bool:
    """Determine if an exception is transient and retryable."""
    # google.api_core.exceptions
    if hasattr(exc, "code"):
        return exc.code in _TRANSIENT_STATUS_CODES  # type: ignore[union-attr]
    # kubernetes ApiException
    if hasattr(exc, "status"):
        return exc.status in _TRANSIENT_STATUS_CODES  # type: ignore[union-attr]
    # ConnectionError, TimeoutError
    if isinstance(exc, (ConnectionError, TimeoutError, asyncio.TimeoutError)):
        return True
    return False


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
) -> Callable[[F], F]:
    """Decorator that retries async functions on transient GCP/K8s errors.

    Uses exponential backoff with full jitter:
        delay = random(0, min(max_delay, base_delay * backoff_factor ** attempt))
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
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
