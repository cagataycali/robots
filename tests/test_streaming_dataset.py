"""Unit tests for ``strands_robots.streaming_dataset.StreamingDatasetReader``
and ``DatasetRecorder.sync_to_bucket``.

Mirrors test_dataset_recorder.py: inject fakes so tests run WITHOUT lerobot or
the ``hf`` CLI installed. Covers version-tolerant kwarg forwarding, the
proprio-only ``drop_videos`` path, delta-grid validation, and the bucket-sync
CLI construction + meta/ guard.
"""

import subprocess

import pytest

import strands_robots.streaming_dataset as sd
from strands_robots.dataset_recorder import DatasetRecorder


class _FakeStreaming:
    """Fake StreamingLeRobotDataset capturing the kwargs it was built with."""

    def __init__(self, repo_id, **kw):
        self.repo_id = repo_id
        self.kw = kw
        self.num_frames = 1000
        self.num_episodes = 10
        self.fps = 30

    def __iter__(self):
        yield {"observation.state": [0.0], "action": [0.0], "task": "t"}


def test_open_forwards_supported_kwargs(monkeypatch):
    monkeypatch.setattr(sd, "StreamingLeRobotDataset", _FakeStreaming, raising=False)
    r = sd.StreamingDatasetReader.open(
        "org/ds", buffer_size=256, shuffle=False, max_num_shards=8,
        validate_deltas=False,
    )
    assert r.dataset.repo_id == "org/ds"
    assert r.dataset.kw["buffer_size"] == 256
    assert r.dataset.kw["shuffle"] is False
    assert r.dataset.kw["max_num_shards"] == 8
    assert r.num_episodes == 10
    assert r.fps == 30


def test_open_drops_unknown_kwargs(monkeypatch):
    """A narrow constructor (only repo_id) must not raise on extra kwargs."""

    class _Narrow:
        def __init__(self, repo_id):
            self.repo_id = repo_id
            self.num_frames = self.num_episodes = self.fps = 0

        def __iter__(self):
            yield {}

    monkeypatch.setattr(sd, "StreamingLeRobotDataset", _Narrow, raising=False)
    r = sd.StreamingDatasetReader.open(
        "org/ds", buffer_size=999, shuffle=True, validate_deltas=False
    )
    assert r.dataset.repo_id == "org/ds"


def test_drop_videos_strips_camera_deltas(monkeypatch):
    monkeypatch.setattr(sd, "StreamingLeRobotDataset", _FakeStreaming, raising=False)
    r = sd.StreamingDatasetReader.open(
        "org/ds",
        delta_timestamps={
            "observation.images.front": [-0.1, 0.0],
            "observation.state": [0.0],
            "action": [0.0],
        },
        drop_videos=True,
        validate_deltas=False,
    )
    dt = r.dataset.kw["delta_timestamps"]
    assert "observation.images.front" not in dt
    assert "observation.state" in dt and "action" in dt


def test_drop_videos_all_camera_keys_yields_none(monkeypatch):
    monkeypatch.setattr(sd, "StreamingLeRobotDataset", _FakeStreaming, raising=False)
    r = sd.StreamingDatasetReader.open(
        "org/ds",
        delta_timestamps={"observation.images.front": [-0.1, 0.0]},
        drop_videos=True,
        validate_deltas=False,
    )
    # All keys were camera keys → delta_timestamps drops out entirely.
    assert "delta_timestamps" not in r.dataset.kw


def test_dataloader_ignores_shuffle(monkeypatch):
    monkeypatch.setattr(sd, "StreamingLeRobotDataset", _FakeStreaming, raising=False)
    r = sd.StreamingDatasetReader.open("org/ds", validate_deltas=False)

    captured = {}

    class _FakeDataLoader:
        def __init__(self, dataset, batch_size, num_workers, **kw):
            captured["shuffle_in_kw"] = "shuffle" in kw
            captured["batch_size"] = batch_size

    class _FakeTorchUtilsData:
        DataLoader = _FakeDataLoader

    class _FakeTorch:
        utils = type("u", (), {"data": _FakeTorchUtilsData})

    import sys

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch)
    r.dataloader(batch_size=32, shuffle=True)  # shuffle must be swallowed
    assert captured["shuffle_in_kw"] is False
    assert captured["batch_size"] == 32


# ── sync_to_bucket ─────────────────────────────────────────────────────────


class _FakeDataset:
    def __init__(self, root):
        self.repo_id = "org/pick"
        self.root = root


def _recorder(tmp_path):
    rec = DatasetRecorder(dataset=_FakeDataset(str(tmp_path)))
    rec.episode_count = 3
    rec.frame_count = 300
    return rec


def test_sync_to_bucket_builds_cli(tmp_path, monkeypatch):
    (tmp_path / "meta").mkdir()  # satisfy the meta/ guard
    rec = _recorder(tmp_path)

    import shutil

    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/hf")

    calls = []

    def fake_run(cmd, capture_output=True, text=True):
        calls.append(cmd)

        class R:
            returncode = 0
            stdout = ""
            stderr = ""

        return R()

    monkeypatch.setattr(subprocess, "run", fake_run)

    res = rec.sync_to_bucket("my-org/robot-fave", run_id="run-021")
    assert res["status"] == "success"
    assert res["bucket_uri"] == "hf://buckets/my-org/robot-fave/run-021"
    assert any(c[:3] == ["hf", "buckets", "create"] for c in calls)
    assert any(
        c[:2] == ["hf", "sync"] and c[-1].startswith("hf://buckets/") for c in calls
    )


def test_sync_to_bucket_requires_meta(tmp_path, monkeypatch):
    rec = _recorder(tmp_path)  # NO meta/ dir
    import shutil

    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/hf")
    res = rec.sync_to_bucket("my-org/robot-fave")
    assert res["status"] == "error"
    assert "meta/" in res["message"]


def test_sync_to_bucket_missing_hf_cli(tmp_path, monkeypatch):
    (tmp_path / "meta").mkdir()
    rec = _recorder(tmp_path)
    import shutil

    monkeypatch.setattr(shutil, "which", lambda _: None)
    res = rec.sync_to_bucket("my-org/robot-fave")
    assert res["status"] == "error"
    assert "hf` CLI" in res["message"] or "hf CLI" in res["message"]


# ── stream_dataset facade ──────────────────────────────────────────────────


def test_recording_mixin_stream_dataset_delegates(monkeypatch):
    """sim.stream_dataset(...) must delegate to StreamingDatasetReader.open,
    keeping streaming a native facade method (not user-side plumbing)."""
    from strands_robots.simulation.mujoco.recording import RecordingMixin

    captured = {}

    def fake_open(repo_id, **kw):
        captured["repo_id"] = repo_id
        captured["kw"] = kw
        return "READER"

    monkeypatch.setattr(
        sd.StreamingDatasetReader, "open", staticmethod(fake_open), raising=True
    )

    mixin = RecordingMixin()
    out = mixin.stream_dataset(
        "org/ds", root="/tmp/x", shuffle=False, drop_videos=True
    )
    assert out == "READER"
    assert captured["repo_id"] == "org/ds"
    assert captured["kw"]["root"] == "/tmp/x"
    assert captured["kw"]["shuffle"] is False
    assert captured["kw"]["drop_videos"] is True
