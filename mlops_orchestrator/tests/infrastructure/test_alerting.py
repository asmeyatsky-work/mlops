"""Tests for alerting adapters."""
from __future__ import annotations

import pytest

from mlops_orchestrator.domain.ports.alerting_port import Alert
from mlops_orchestrator.infrastructure.adapters.alerting_adapters import (
    CompositeAlertAdapter,
    StubAlertAdapter,
)


@pytest.fixture
def stub_alert():
    return StubAlertAdapter()


@pytest.fixture
def sample_alert():
    return Alert(
        title="Test Alert",
        message="Something happened",
        severity="warning",
        source="test",
        metadata={"key": "value"},
    )


class TestStubAlertAdapter:
    async def test_send_alert(self, stub_alert, sample_alert):
        result = await stub_alert.send_alert(sample_alert)
        assert result is True
        assert len(stub_alert.sent_alerts) == 1
        assert stub_alert.sent_alerts[0].title == "Test Alert"

    async def test_clear(self, stub_alert, sample_alert):
        await stub_alert.send_alert(sample_alert)
        stub_alert.clear()
        assert len(stub_alert.sent_alerts) == 0

    async def test_multiple_alerts(self, stub_alert):
        for severity in ("info", "warning", "critical"):
            await stub_alert.send_alert(
                Alert(title=f"Alert {severity}", message="msg", severity=severity)
            )
        assert len(stub_alert.sent_alerts) == 3


class TestCompositeAlertAdapter:
    async def test_fans_out_to_all_adapters(self, sample_alert):
        a1 = StubAlertAdapter()
        a2 = StubAlertAdapter()
        composite = CompositeAlertAdapter([a1, a2])
        result = await composite.send_alert(sample_alert)
        assert result is True
        assert len(a1.sent_alerts) == 1
        assert len(a2.sent_alerts) == 1

    async def test_empty_adapters(self, sample_alert):
        composite = CompositeAlertAdapter([])
        result = await composite.send_alert(sample_alert)
        assert result is True

    async def test_returns_true_if_any_succeed(self, sample_alert):
        """Even if one adapter fails, returns True if another succeeds."""
        a1 = StubAlertAdapter()
        composite = CompositeAlertAdapter([a1])
        result = await composite.send_alert(sample_alert)
        assert result is True


class TestAlert:
    def test_frozen(self):
        alert = Alert(title="t", message="m", severity="info")
        with pytest.raises(AttributeError):
            alert.title = "modified"  # type: ignore[misc]

    def test_default_source(self):
        alert = Alert(title="t", message="m", severity="info")
        assert alert.source == "mlops-orchestrator"

    def test_metadata_optional(self):
        alert = Alert(title="t", message="m", severity="warning")
        assert alert.metadata is None

    def test_all_severity_levels(self):
        for sev in ("info", "warning", "critical"):
            alert = Alert(title="t", message="m", severity=sev)
            assert alert.severity == sev

    def test_with_metadata(self):
        alert = Alert(title="t", message="m", severity="info", metadata={"k": "v"})
        assert alert.metadata == {"k": "v"}
