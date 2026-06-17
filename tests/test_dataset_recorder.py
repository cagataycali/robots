"""Unit tests for ``strands_robots.dataset_recorder.DatasetRecorder``.

These tests exercise the wrapper logic that does NOT require a real LeRobot
dataset by injecting a fake dataset object, so they run on a minimal env
(``lerobot`` not installed). They cover the partial-episode discard behaviour
and the add_frame() control-loop transform (schema-ordered flattening, camera
normalization, drop accounting), plus episode/finalize/push lifecycle.
"""

import logging

import numpy as np
import pytest

from strands_robots.dataset_recorder import DatasetRecorder


class _FakeDatasetWithClear:
    """Fake LeRobot dataset exposing ``clear_episode_buffer`` (preferred path)."""

    def __init__(self):
        self.repo_id = "local/fake"
        self.cleared = 0

    def clear_episode_buffer(self):
        self.cleared += 1


class _FakeDatasetWithCreate:
    """Fake dataset exposing only ``create_episode_buffer`` (fallback path)."""

    def __init__(self):
        self.repo_id = "local/fake"
        self.episode_buffer = {"frames": [1, 2, 3]}
        self.created = 0

    def create_episode_buffer(self):
        self.created += 1
        return {}


class _FakeDatasetNoClear:
    """Fake dataset exposing no buffer-reset surface (warn-only path)."""

    def __init__(self):
        self.repo_id = "local/fake"


def _recorder_for(dataset) -> DatasetRecorder:
    rec = DatasetRecorder(dataset=dataset)
    rec.episode_frame_count = 5  # simulate 5 frames buffered for the open episode
    rec.frame_count = 5
    return rec


def test_clear_episode_buffer_prefers_native_clear():
    ds = _FakeDatasetWithClear()
    rec = _recorder_for(ds)

    assert rec.clear_episode_buffer() is True
    assert ds.cleared == 1
    # Next episode starts at frame 0; cumulative frame_count is untouched
    # (those frames were only ever in the open buffer, not flushed to disk).
    assert rec.episode_frame_count == 0


def test_clear_episode_buffer_falls_back_to_create_buffer():
    ds = _FakeDatasetWithCreate()
    rec = _recorder_for(ds)

    assert rec.clear_episode_buffer() is True
    assert ds.created == 1
    assert ds.episode_buffer == {}
    assert rec.episode_frame_count == 0


def test_clear_episode_buffer_warns_when_no_surface(caplog):
    ds = _FakeDatasetNoClear()
    rec = _recorder_for(ds)

    import logging

    with caplog.at_level(logging.WARNING, logger="strands_robots.dataset_recorder"):
        result = rec.clear_episode_buffer()

    assert result is False
    # Counter still resets so reporting does not carry over the discarded frames.
    assert rec.episode_frame_count == 0
    assert any("partial episode" in r.message for r in caplog.records)


def test_clear_episode_buffer_swallows_dataset_error(caplog):
    """A failure inside the dataset's clear must not mask the original abort."""

    class _Boom:
        repo_id = "local/fake"

        def clear_episode_buffer(self):
            raise RuntimeError("buffer is wedged")

    rec = _recorder_for(_Boom())

    import logging

    with caplog.at_level(logging.WARNING, logger="strands_robots.dataset_recorder"):
        result = rec.clear_episode_buffer()

    assert result is False
    assert rec.episode_frame_count == 0


def test_clear_episode_buffer_ascii_only_warnings(caplog):
    """Recorder log strings must be plain ASCII (project string-hygiene rule)."""
    rec = _recorder_for(_FakeDatasetNoClear())

    import logging

    with caplog.at_level(logging.WARNING, logger="strands_robots.dataset_recorder"):
        rec.clear_episode_buffer()

    for record in caplog.records:
        record.getMessage().encode("ascii")  # raises if any non-ASCII glyph leaked


# add_frame() control-loop transformation behaviour
#
# These tests inject a fake LeRobot dataset that captures every frame handed to
# add_frame(), so they assert the recorder's observable output (the frame dict
# shape, vector ordering, camera normalization, drop accounting) without a real
# lerobot install.


class _CapturingDataset:
    """Fake LeRobot dataset that records frames and exposes a feature schema."""

    def __init__(self, features: dict, *, fail_add: bool = False):
        self.repo_id = "local/fake"
        self.root = "/tmp/local-fake"
        self.features = features
        self.frames: list[dict] = []
        self.saved = 0
        self.finalized = 0
        self.pushed: dict | None = None
        self._fail_add = fail_add

    def add_frame(self, frame):
        if self._fail_add:
            raise RuntimeError("disk full")
        self.frames.append(frame)

    def save_episode(self):
        self.saved += 1

    def finalize(self):
        self.finalized += 1

    def push_to_hub(self, tags=None, private=False):
        self.pushed = {"tags": tags, "private": private}


def _state_action_features(state_names, action_names) -> dict:
    return {
        "observation.state": {"dtype": "float32", "names": state_names},
        "action": {"dtype": "float32", "names": action_names},
    }


def test_add_frame_orders_state_and_action_by_schema():
    """State/action vectors follow the feature-schema order, not dict order."""
    ds = _CapturingDataset(_state_action_features(["j1", "j2", "j3"], ["j1", "j2", "j3"]))
    rec = DatasetRecorder(dataset=ds, task="pick")

    # Pass observation/action keys in a deliberately scrambled order.
    rec.add_frame(
        observation={"j3": 3.0, "j1": 1.0, "j2": 2.0},
        action={"j2": 0.2, "j3": 0.3, "j1": 0.1},
    )

    assert rec.frame_count == 1
    assert rec.episode_frame_count == 1
    frame = ds.frames[0]
    assert np.allclose(frame["observation.state"], [1.0, 2.0, 3.0])
    assert np.allclose(frame["action"], [0.1, 0.2, 0.3])
    assert frame["observation.state"].dtype == np.float32
    assert frame["task"] == "pick"


def test_add_frame_fills_missing_keys_with_zero():
    """A joint absent from the observation contributes 0.0 at its schema slot."""
    ds = _CapturingDataset(_state_action_features(["j1", "j2", "j3"], ["j1"]))
    rec = DatasetRecorder(dataset=ds)

    rec.add_frame(observation={"j1": 1.0, "j3": 3.0}, action={"j1": 0.5}, task="t")

    frame = ds.frames[0]
    assert np.allclose(frame["observation.state"], [1.0, 0.0, 3.0])
    assert np.allclose(frame["action"], [0.5])


def test_add_frame_flattens_vector_valued_entries():
    """List / ndarray observation values are flattened into the state vector."""
    ds = _CapturingDataset(_state_action_features(["pose", "grip"], ["cmd"]))
    rec = DatasetRecorder(dataset=ds)

    rec.add_frame(
        observation={"pose": [1.0, 2.0, 3.0], "grip": 0.5},
        action={"cmd": np.array([0.1, 0.2])},
        task="t",
    )

    frame = ds.frames[0]
    assert np.allclose(frame["observation.state"], [1.0, 2.0, 3.0, 0.5])
    assert np.allclose(frame["action"], [0.1, 0.2])


def test_add_frame_converts_float_images_to_uint8():
    """A float image in [0, 1] is scaled to uint8 HWC for LeRobot."""
    feats = {"observation.images.cam": {"dtype": "video"}}
    ds = _CapturingDataset(feats)
    rec = DatasetRecorder(dataset=ds)

    img = np.ones((4, 4, 3), dtype=np.float32)  # all-white in [0, 1]
    rec.add_frame(observation={"cam": img}, action={}, task="t")

    out = ds.frames[0]["observation.images.cam"]
    assert out.dtype == np.uint8
    assert out.shape == (4, 4, 3)
    assert np.array_equal(out, np.full((4, 4, 3), 255, dtype=np.uint8))


def test_add_frame_normalizes_namespaced_camera_keys():
    """A 'arm0/wrist' camera key is rewritten to the declared 'arm0__wrist'."""
    feats = {"observation.images.arm0__wrist": {"dtype": "video"}}
    ds = _CapturingDataset(feats)
    rec = DatasetRecorder(dataset=ds)

    img = np.zeros((2, 2, 3), dtype=np.uint8)
    rec.add_frame(observation={"arm0/wrist": img}, action={}, task="t")

    frame = ds.frames[0]
    assert "observation.images.arm0__wrist" in frame
    assert "observation.images.arm0/wrist" not in frame


def test_add_frame_strips_undeclared_cameras():
    """A camera not in the feature schema is dropped to avoid 'extra features'."""
    feats = {"observation.images.declared": {"dtype": "video"}}
    ds = _CapturingDataset(feats)
    rec = DatasetRecorder(dataset=ds)

    img = np.zeros((2, 2, 3), dtype=np.uint8)
    rec.add_frame(
        observation={"declared": img, "ghost": img},
        action={},
        task="t",
    )

    frame = ds.frames[0]
    assert "observation.images.declared" in frame
    assert "observation.images.ghost" not in frame


def test_add_frame_strict_reraises_on_dataset_error():
    """strict=True (default) propagates a dataset add_frame failure."""
    ds = _CapturingDataset(_state_action_features(["j1"], ["j1"]), fail_add=True)
    rec = DatasetRecorder(dataset=ds, strict=True)

    with pytest.raises(RuntimeError, match="disk full"):
        rec.add_frame(observation={"j1": 1.0}, action={"j1": 0.1}, task="t")
    assert rec.frame_count == 0


def test_add_frame_non_strict_counts_drops_without_raising(caplog):
    """strict=False swallows the error, counts the drop, and logs ASCII-only."""
    ds = _CapturingDataset(_state_action_features(["j1"], ["j1"]), fail_add=True)
    rec = DatasetRecorder(dataset=ds, strict=False)

    with caplog.at_level(logging.WARNING, logger="strands_robots.dataset_recorder"):
        rec.add_frame(observation={"j1": 1.0}, action={"j1": 0.1}, task="t")

    assert rec.dropped_frame_count == 1
    assert rec.frame_count == 0
    for record in caplog.records:
        record.getMessage().encode("ascii")


def test_add_frame_noop_when_closed():
    """A closed recorder ignores add_frame instead of corrupting the dataset."""
    ds = _CapturingDataset(_state_action_features(["j1"], ["j1"]))
    rec = DatasetRecorder(dataset=ds)
    rec._closed = True

    rec.add_frame(observation={"j1": 1.0}, action={"j1": 0.1}, task="t")

    assert ds.frames == []
    assert rec.frame_count == 0


def test_save_episode_reports_per_episode_frame_count():
    """save_episode reports frames since the last save and resets the counter."""
    ds = _CapturingDataset(_state_action_features(["j1"], ["j1"]))
    rec = DatasetRecorder(dataset=ds)

    rec.add_frame(observation={"j1": 1.0}, action={"j1": 0.1}, task="t")
    rec.add_frame(observation={"j1": 2.0}, action={"j1": 0.2}, task="t")
    result = rec.save_episode()

    assert result["status"] == "success"
    assert result["episode"] == 1
    assert result["episode_frames"] == 2
    assert result["total_frames"] == 2
    assert rec.episode_frame_count == 0  # reset for the next episode

    # A second episode reports only its own frames; total stays monotonic.
    rec.add_frame(observation={"j1": 3.0}, action={"j1": 0.3}, task="t")
    result2 = rec.save_episode()
    assert result2["episode_frames"] == 1
    assert result2["total_frames"] == 3


def test_save_episode_poisons_recorder_on_failure():
    """A failed save closes the recorder so later frames cannot corrupt data."""

    class _BadSave(_CapturingDataset):
        def save_episode(self):
            raise RuntimeError("encode failed")

    ds = _BadSave(_state_action_features(["j1"], ["j1"]))
    rec = DatasetRecorder(dataset=ds)
    rec.add_frame(observation={"j1": 1.0}, action={"j1": 0.1}, task="t")

    result = rec.save_episode()

    assert result["status"] == "error"
    assert rec._closed is True
    # save_episode on a closed recorder returns a clean error, not a crash.
    assert rec.save_episode()["status"] == "error"


def test_finalize_is_idempotent_and_closes():
    ds = _CapturingDataset(_state_action_features(["j1"], ["j1"]))
    rec = DatasetRecorder(dataset=ds)

    rec.finalize()
    rec.finalize()  # second call is a no-op

    assert rec._closed is True
    assert ds.finalized == 1


def test_push_to_hub_success_and_failure():
    ds = _CapturingDataset(_state_action_features(["j1"], ["j1"]))
    rec = DatasetRecorder(dataset=ds)
    rec.episode_count = 2
    rec.frame_count = 50

    ok = rec.push_to_hub(tags=["sim"], private=True)
    assert ok == {
        "status": "success",
        "repo_id": "local/fake",
        "episodes": 2,
        "frames": 50,
    }
    assert ds.pushed == {"tags": ["sim"], "private": True}

    class _BadPush(_CapturingDataset):
        def push_to_hub(self, tags=None, private=False):
            raise RuntimeError("network down")

    bad = DatasetRecorder(dataset=_BadPush(_state_action_features(["j1"], ["j1"])))
    err = bad.push_to_hub()
    assert err["status"] == "error"
    assert "network down" in err["message"]


def test_repo_id_root_and_repr_properties():
    ds = _CapturingDataset(_state_action_features(["j1"], ["j1"]))
    rec = DatasetRecorder(dataset=ds)
    rec.episode_count = 1
    rec.frame_count = 7

    assert rec.repo_id == "local/fake"
    assert rec.root == "/tmp/local-fake"
    rep = repr(rec)
    assert "local/fake" in rep and "episodes=1" in rep and "frames=7" in rep
