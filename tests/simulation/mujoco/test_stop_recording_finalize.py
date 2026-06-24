"""stop_recording finalize / bucket-sync / hub-publish contract.

``Simulation.stop_recording`` closes an in-progress LeRobotDataset recording.
Beyond saving the episode it has three contractual side effects that an agent
relies on in the physical-AI data loop:

* the episode is saved and the dataset finalized (meta/ written) BEFORE any
  upload, so downstream streaming/training sees a complete dataset;
* when a ``bucket`` is given it syncs to the mutable HF Storage Bucket and the
  reported text reflects success or failure;
* when ``push_to_hub`` is set (per-call or from ``start_recording``) it publishes
  the versioned dataset repo and the text reflects success or failure.

These tests drive a fake recorder so the contract is pinned without the
``lerobot`` extra or any real Hub I/O - only the orchestration in
``recording.py`` runs.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


class _FakeRecorder:
    """Minimal stand-in for ``DatasetRecorder`` capturing orchestration order."""

    def __init__(self, *, sync_result=None, push_result=None):
        self.repo_id = "local/finalize_test"
        self.frame_count = 7
        self.episode_count = 1
        self.root = "/tmp/finalize_test"
        self.calls: list[str] = []
        self._sync_result = sync_result
        self._push_result = push_result
        self.sync_args: tuple | None = None
        self.push_tags = None

    def save_episode(self):
        self.calls.append("save_episode")

    def finalize(self):
        self.calls.append("finalize")

    def sync_to_bucket(self, bucket, run_id=None):
        self.calls.append("sync_to_bucket")
        self.sync_args = (bucket, run_id)
        return self._sync_result

    def push_to_hub(self, tags=None):
        self.calls.append("push_to_hub")
        self.push_tags = tags
        return self._push_result


@pytest.fixture
def recording_sim():
    s = Simulation(tool_name="stop_finalize_test", mesh=False)
    s.create_world()
    yield s
    s.cleanup()


def _arm(sim, recorder, *, push_to_hub=False):
    """Put the sim into a recording state backed by ``recorder``."""
    sim._world._backend_state["recording"] = True
    sim._world._backend_state["dataset_recorder"] = recorder
    sim._world._backend_state["push_to_hub"] = push_to_hub


class TestStopRecordingFinalize:
    def test_not_recording_is_idempotent(self, recording_sim):
        result = recording_sim.stop_recording()
        assert result["status"] == "success"
        assert "Was not recording" in result["content"][0]["text"]

    def test_missing_recorder_reports_error(self, recording_sim):
        # recording flagged on but no recorder object present
        recording_sim._world._backend_state["recording"] = True
        recording_sim._world._backend_state["dataset_recorder"] = None
        result = recording_sim.stop_recording()
        assert result["status"] == "error"
        assert "No dataset recorder active" in result["content"][0]["text"]

    def test_saves_and_finalizes_before_any_upload(self, recording_sim):
        rec = _FakeRecorder()
        _arm(recording_sim, rec)
        result = recording_sim.stop_recording()
        assert result["status"] == "success"
        # save happens, then finalize; no upload requested.
        assert rec.calls == ["save_episode", "finalize"]
        text = result["content"][0]["text"]
        assert "7 frames" in text
        assert rec.repo_id in text
        # state cleared so a subsequent stop is a no-op.
        assert recording_sim._world._backend_state["dataset_recorder"] is None
        assert recording_sim._world._backend_state["recording"] is False

    def test_bucket_sync_success_reports_uri_and_runs_after_finalize(self, recording_sim):
        rec = _FakeRecorder(sync_result={"status": "success", "bucket_uri": "hf://org/buck/run1"})
        _arm(recording_sim, rec)
        result = recording_sim.stop_recording(bucket="org/buck", run_id="run1")
        assert result["status"] == "success"
        # finalize must precede the bucket sync.
        assert rec.calls.index("finalize") < rec.calls.index("sync_to_bucket")
        assert rec.sync_args == ("org/buck", "run1")
        assert "Synced to bucket: hf://org/buck/run1" in result["content"][0]["text"]

    def test_bucket_sync_failure_is_surfaced(self, recording_sim):
        rec = _FakeRecorder(sync_result={"status": "error", "message": "bucket unreachable"})
        _arm(recording_sim, rec)
        result = recording_sim.stop_recording(bucket="org/buck")
        assert result["status"] == "success"
        assert "Bucket sync FAILED: bucket unreachable" in result["content"][0]["text"]

    def test_push_to_hub_per_call_publishes_with_tags(self, recording_sim):
        rec = _FakeRecorder(push_result={"status": "success"})
        _arm(recording_sim, rec)
        result = recording_sim.stop_recording(push_to_hub=True)
        assert result["status"] == "success"
        assert "push_to_hub" in rec.calls
        assert rec.push_tags == ["strands-robots", "sim"]
        assert "Pushed to HuggingFace Hub" in result["content"][0]["text"]

    def test_push_to_hub_inherited_from_start_recording(self, recording_sim):
        rec = _FakeRecorder(push_result={"status": "success"})
        # push not requested per-call, but armed at start_recording.
        _arm(recording_sim, rec, push_to_hub=True)
        result = recording_sim.stop_recording()
        assert "push_to_hub" in rec.calls
        assert "Pushed to HuggingFace Hub" in result["content"][0]["text"]

    def test_push_to_hub_failure_is_surfaced(self, recording_sim):
        rec = _FakeRecorder(push_result={"status": "error", "message": "auth denied"})
        _arm(recording_sim, rec)
        result = recording_sim.stop_recording(push_to_hub=True)
        assert result["status"] == "success"
        assert "push_to_hub FAILED: auth denied" in result["content"][0]["text"]
