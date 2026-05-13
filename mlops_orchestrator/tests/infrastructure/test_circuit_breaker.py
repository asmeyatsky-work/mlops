"""Tests for the circuit breaker and hard-timeout retry behaviour."""
from __future__ import annotations

import asyncio

import pytest

from mlops_orchestrator.infrastructure.adapters.retry import (
    CircuitBreaker,
    CircuitBreakerOpen,
    with_retry,
)


class _Transient(Exception):
    code = 503


class TestCircuitBreaker:
    async def test_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=2, reset_seconds=60, name="test")

        async def boom():
            raise _Transient()

        for _ in range(2):
            with pytest.raises(_Transient):
                await cb.call(boom)
        assert cb.state == "OPEN"

        async def ok():
            return "ok"

        with pytest.raises(CircuitBreakerOpen):
            await cb.call(ok)

    async def test_half_open_probe_closes_circuit(self):
        cb = CircuitBreaker(failure_threshold=1, reset_seconds=0.01, name="t")

        async def boom():
            raise _Transient()

        with pytest.raises(_Transient):
            await cb.call(boom)
        assert cb.state == "OPEN"
        await asyncio.sleep(0.02)
        assert cb.state == "HALF_OPEN"

        async def ok():
            return "ok"

        assert await cb.call(ok) == "ok"
        assert cb.state == "CLOSED"


class TestHardTimeout:
    async def test_wait_for_enforces_per_attempt_timeout(self):
        @with_retry(max_attempts=1, timeout_seconds=0.05)
        async def slow():
            await asyncio.sleep(1.0)

        with pytest.raises(asyncio.TimeoutError):
            await slow()
