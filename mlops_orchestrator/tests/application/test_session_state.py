"""Tests for session state."""
from __future__ import annotations

from types import MappingProxyType

from mlops_orchestrator.application.session.session_state import SessionState


class TestSessionState:
    def test_initial_state(self):
        s = SessionState()
        assert s.dataset_ids == ()
        assert s.latest_dataset == ""
        assert s.latest_model == ""
        assert s.active_project == ""

    def test_add_dataset(self):
        s = SessionState().add_dataset("ds-1")
        assert s.dataset_ids == ("ds-1",)
        assert s.latest_dataset == "ds-1"

    def test_add_model_uri(self):
        s = SessionState().add_model_uri("model-1")
        assert s.latest_model == "model-1"

    def test_add_job_handle(self):
        s = SessionState().add_job_handle("job-1")
        assert s.latest_job == "job-1"

    def test_add_endpoint(self):
        s = SessionState().add_endpoint("ep-1")
        assert s.latest_endpoint == "ep-1"

    def test_set_project(self):
        s = SessionState().set_project("my-proj")
        assert s.active_project == "my-proj"

    def test_set_metadata(self):
        s = SessionState().set_metadata("key", "val")
        assert s.metadata == {"key": "val"}

    def test_chaining(self):
        s = (
            SessionState()
            .set_project("p")
            .add_dataset("d1")
            .add_dataset("d2")
            .add_model_uri("m")
        )
        assert s.latest_dataset == "d2"
        assert len(s.dataset_ids) == 2

    def test_immutability(self):
        s1 = SessionState()
        s2 = s1.add_dataset("d")
        assert s1.dataset_ids == ()
        assert s2.dataset_ids == ("d",)

    def test_to_dict(self):
        s = SessionState().add_dataset("d").set_project("p").set_metadata("k", "v")
        d = s.to_dict()
        assert d["dataset_ids"] == ["d"]
        assert d["active_project"] == "p"
        assert d["metadata"] == {"k": "v"}

    # ── New tests ──────────────────────────────────────────────────────

    def test_metadata_property_returns_mapping_proxy_type(self):
        """metadata property returns a MappingProxyType (read-only view)."""
        s = SessionState().set_metadata("a", "1")
        meta = s.metadata
        assert isinstance(meta, MappingProxyType)
        assert meta["a"] == "1"

    def test_metadata_read_only_rejects_mutation(self):
        """Attempting to mutate the MappingProxyType raises TypeError."""
        s = SessionState().set_metadata("a", "1")
        meta = s.metadata
        try:
            meta["a"] = "2"  # type: ignore[index]
            assert False, "Should have raised TypeError"
        except TypeError:
            pass

    def test_set_metadata_overwrites_existing_key(self):
        """set_metadata with an existing key replaces its value."""
        s = SessionState().set_metadata("k", "old").set_metadata("k", "new")
        assert s.metadata["k"] == "new"

    def test_mutation_safety_returned_metadata_does_not_affect_state(self):
        """Mutating the dict obtained via to_dict() does not alter the state."""
        s = SessionState().set_metadata("x", "1")
        d = s.to_dict()
        d["metadata"]["x"] = "HACKED"
        # Original state must be unchanged
        assert s.metadata["x"] == "1"

    def test_to_dict_includes_all_fields(self):
        """to_dict() includes model_uris, job_handles, and endpoint_names."""
        s = (
            SessionState()
            .add_dataset("d")
            .add_model_uri("m")
            .add_job_handle("j")
            .add_endpoint("e")
            .set_project("p")
            .set_metadata("k", "v")
        )
        d = s.to_dict()
        assert d["dataset_ids"] == ["d"]
        assert d["model_uris"] == ["m"]
        assert d["job_handles"] == ["j"]
        assert d["endpoint_names"] == ["e"]
        assert d["active_project"] == "p"
        assert d["metadata"] == {"k": "v"}

    def test_latest_job_returns_empty_when_empty(self):
        """latest_job returns '' on a fresh state with no job handles."""
        s = SessionState()
        assert s.latest_job == ""

    def test_latest_endpoint_returns_empty_when_empty(self):
        """latest_endpoint returns '' on a fresh state with no endpoints."""
        s = SessionState()
        assert s.latest_endpoint == ""
