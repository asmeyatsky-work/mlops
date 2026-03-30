"""Tests for retry utility with exponential backoff."""
from __future__ import annotations

import asyncio

import pytest

from mlops_orchestrator.infrastructure.adapters.retry import with_retry, _is_transient


class FakeTransientError(Exception):
    """Simulates a GCP transient error with a status code."""
    def __init__(self, code: int):
        self.code = code
        super().__init__(f"transient error {code}")


class FakeK8sError(Exception):
    """Simulates a K8s ApiException with a status attribute."""
    def __init__(self, status: int):
        self.status = status
        super().__init__(f"k8s error {status}")


class TestIsTransient:
    def test_transient_status_codes(self):
        for code in [408, 429, 500, 502, 503, 504]:
            assert _is_transient(FakeTransientError(code))

    def test_non_transient_status_code(self):
        assert not _is_transient(FakeTransientError(400))
        assert not _is_transient(FakeTransientError(404))

    def test_connection_error(self):
        assert _is_transient(ConnectionError())

    def test_timeout_error(self):
        assert _is_transient(TimeoutError())
        assert _is_transient(asyncio.TimeoutError())

    def test_k8s_transient(self):
        assert _is_transient(FakeK8sError(503))

    def test_k8s_non_transient(self):
        assert not _is_transient(FakeK8sError(404))

    def test_generic_error_not_transient(self):
        assert not _is_transient(ValueError("nope"))


class TestWithRetry:
    async def test_succeeds_on_first_try(self):
        call_count = 0

        @with_retry(max_attempts=3, base_delay=0.01)
        async def succeed():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await succeed()
        assert result == "ok"
        assert call_count == 1

    async def test_retries_on_transient_error(self):
        call_count = 0

        @with_retry(max_attempts=3, base_delay=0.01)
        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise FakeTransientError(503)
            return "recovered"

        result = await fail_then_succeed()
        assert result == "recovered"
        assert call_count == 3

    async def test_raises_after_max_attempts(self):
        call_count = 0

        @with_retry(max_attempts=2, base_delay=0.01)
        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise FakeTransientError(503)

        with pytest.raises(FakeTransientError):
            await always_fail()
        assert call_count == 2

    async def test_non_transient_error_not_retried(self):
        call_count = 0

        @with_retry(max_attempts=3, base_delay=0.01)
        async def permanent_fail():
            nonlocal call_count
            call_count += 1
            raise ValueError("permanent")

        with pytest.raises(ValueError, match="permanent"):
            await permanent_fail()
        assert call_count == 1

    async def test_preserves_return_type(self):
        @with_retry(max_attempts=2, base_delay=0.01)
        async def return_dict():
            return {"key": "value"}

        result = await return_dict()
        assert result == {"key": "value"}
