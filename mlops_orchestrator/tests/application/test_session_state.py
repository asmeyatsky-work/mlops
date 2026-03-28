"""Tests for session state."""
from __future__ import annotations

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
