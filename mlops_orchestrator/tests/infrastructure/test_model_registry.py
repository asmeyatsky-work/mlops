"""Tests for model registry adapter and commands."""
from __future__ import annotations

import pytest

from mlops_orchestrator.domain.ports.model_registry_port import ModelVersion
from mlops_orchestrator.infrastructure.adapters.stub_model_registry_adapter import (
    StubModelRegistryAdapter,
)
from mlops_orchestrator.application.commands.model_registry_command import (
    ModelRegistryCommand,
)
from mlops_orchestrator.infrastructure.adapters.stub_infrastructure_adapters import (
    StubAuditLogAdapter,
)


class TestStubModelRegistryAdapter:
    async def test_register_model(self):
        adapter = StubModelRegistryAdapter()
        version = await adapter.register_model(
            display_name="my-model",
            artifact_uri="gs://bucket/model",
            serving_container_image="img:latest",
            description="test model",
        )
        assert version.version == 1
        assert version.display_name == "my-model"
        assert version.stage == "development"
        assert "my-model" in version.model_id

    async def test_register_creates_versions(self):
        adapter = StubModelRegistryAdapter()
        v1 = await adapter.register_model("m", "gs://1", "img")
        v2 = await adapter.register_model("m", "gs://2", "img")
        assert v1.version == 1
        assert v2.version == 2
        assert v1.model_id == v2.model_id

    async def test_get_latest_version(self):
        adapter = StubModelRegistryAdapter()
        await adapter.register_model("m", "gs://1", "img")
        v2 = await adapter.register_model("m", "gs://2", "img")
        latest = await adapter.get_model_version(v2.model_id)
        assert latest is not None
        assert latest.version == 2

    async def test_get_specific_version(self):
        adapter = StubModelRegistryAdapter()
        v1 = await adapter.register_model("m", "gs://1", "img")
        await adapter.register_model("m", "gs://2", "img")
        result = await adapter.get_model_version(v1.model_id, version=1)
        assert result is not None
        assert result.version == 1

    async def test_get_nonexistent_model(self):
        adapter = StubModelRegistryAdapter()
        assert await adapter.get_model_version("nonexistent") is None

    async def test_list_versions(self):
        adapter = StubModelRegistryAdapter()
        v1 = await adapter.register_model("m", "gs://1", "img")
        await adapter.register_model("m", "gs://2", "img")
        versions = await adapter.list_versions(v1.model_id)
        assert len(versions) == 2

    async def test_list_versions_empty(self):
        adapter = StubModelRegistryAdapter()
        assert await adapter.list_versions("nonexistent") == []

    async def test_promote_version(self):
        adapter = StubModelRegistryAdapter()
        v1 = await adapter.register_model("m", "gs://1", "img")
        promoted = await adapter.promote_version(v1.model_id, 1, "production")
        assert promoted.stage == "production"
        # Verify it's persisted
        fetched = await adapter.get_model_version(v1.model_id, 1)
        assert fetched is not None
        assert fetched.stage == "production"

    async def test_promote_nonexistent_version(self):
        adapter = StubModelRegistryAdapter()
        v1 = await adapter.register_model("m", "gs://1", "img")
        with pytest.raises(ValueError, match="Version 99 not found"):
            await adapter.promote_version(v1.model_id, 99, "production")

    async def test_compare_versions(self):
        adapter = StubModelRegistryAdapter()
        await adapter.register_model("m", "gs://1", "img", labels={"acc": "0.9"})
        await adapter.register_model("m", "gs://2", "img", labels={"acc": "0.95"})
        model_id = (await adapter.get_model_version(
            "projects/stub-project/locations/us-central1/models/m"
        )).model_id
        comparison = await adapter.compare_versions(model_id, 1, 2)
        assert comparison["version_a"]["version"] == 1
        assert comparison["version_b"]["version"] == 2


class TestModelRegistryCommand:
    async def test_execute(self):
        adapter = StubModelRegistryAdapter()
        audit_log = StubAuditLogAdapter()
        cmd = ModelRegistryCommand(registry_port=adapter, audit_log=audit_log)

        version = await cmd.execute(
            display_name="test-model",
            artifact_uri="gs://bucket/model",
            serving_container_image="img:latest",
            description="test",
        )

        assert version.version == 1
        assert version.display_name == "test-model"
        assert len(audit_log.all_entries) == 1
        assert audit_log.all_entries[0]["action"] == "model_registered"

    async def test_execute_with_labels(self):
        adapter = StubModelRegistryAdapter()
        audit_log = StubAuditLogAdapter()
        cmd = ModelRegistryCommand(registry_port=adapter, audit_log=audit_log)

        version = await cmd.execute(
            display_name="labeled-model",
            artifact_uri="gs://bucket/model",
            serving_container_image="img",
            labels={"framework": "tensorflow"},
        )
        assert version.labels == {"framework": "tensorflow"}


class TestModelVersion:
    def test_frozen(self):
        mv = ModelVersion(
            model_id="m", version=1, resource_name="rn", display_name="n"
        )
        with pytest.raises(AttributeError):
            mv.version = 2  # type: ignore[misc]

    def test_default_stage(self):
        mv = ModelVersion(model_id="m", version=1, resource_name="rn", display_name="n")
        assert mv.stage == "development"
