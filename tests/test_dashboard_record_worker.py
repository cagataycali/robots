"""RecordWorker: the state machine behind /api/record (teleop episode capture).

Drives the worker deterministically - no thread, no hardware, no lerobot:
``autostart_loop=False`` and manual ``tick()`` calls stand in for the control
loop, a fake backend stands in for the arms, a fake recorder for the dataset.
What these tests pin down is the WIRE CONTRACT the frontend was built against
(FRONTEND_HANDOFF.md) and the invariants that make recording honest:
idempotent commands, 0-frame episodes surfaced not half-kept, redo drops the
buffer, close never half-writes.
"""

import numpy as np
import pytest

from strands_robots.dashboard.record_worker import RecordWorker, _save_thumbnail


class FakeBackend:
    def __init__(self):
        self.camera_keys = ["top"]
        self.applied = []
        self.closed = False
        self._n = 0

    def leader_action(self):
        self._n += 1
        return {"shoulder_pan": float(self._n)}

    def follower_apply(self, action):
        self.applied.append(action)
        return action

    def follower_observation(self):
        return {
            "shoulder_pan": float(self._n),
            "top": np.zeros((8, 8, 3), dtype=np.uint8),
        }

    def close(self):
        self.closed = True


class FakeRecorder:
    def __init__(self):
        self.frames = []
        self.saved = 0
        self.cleared = 0
        self.finalized = False
        self.pushed = None

    def add_frame(self, obs, action, task=None):
        self.frames.append((obs, action, task))

    def save_episode(self):
        self.saved += 1
        return {"status": "ok", "episode_index": self.saved - 1}

    def clear_episode_buffer(self):
        self.cleared += 1
        return True

    def finalize(self):
        self.finalized = True

    def push_to_hub(self, repo_id=None):
        self.pushed = repo_id


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def make_worker(**over):
    backend = over.pop("backend", FakeBackend())
    recorder = over.pop("recorder", FakeRecorder())
    clock = over.pop("clock", Clock())
    kw = dict(
        dataset="cagatay/so101-pick", task="pick up the cube",
        leader="arm-leader", follower="arm-follower",
        target_episodes=3, fps=30,
        backend=backend, recorder_factory=lambda **_: recorder,
        thumb_dir=over.pop("thumb_dir", "/tmp/rec-thumbs-test"),
        clock=clock, autostart_loop=False,
    )
    kw.update(over)
    return RecordWorker(**kw), backend, recorder, clock


# ------------------------------------------------------------ construction

def test_open_validation_refuses_the_obvious():
    for bad in (
        dict(dataset="  "), dict(task=""), dict(follower="arm-leader"),
        dict(fps=0), dict(fps=500), dict(target_episodes=0),
    ):
        with pytest.raises(ValueError):
            make_worker(**bad)


def test_fresh_session_matches_the_wire_contract():
    w, _, _, _ = make_worker()
    s = w.session()
    assert s["dataset"] == "cagatay/so101-pick"
    assert s["task"] == "pick up the cube"
    assert s["leader"] == "arm-leader" and s["follower"] == "arm-follower"
    assert s["target_episodes"] == 3 and s["fps"] == 30
    assert s["phase"] == "idle" and s["episodes"] == []


# ------------------------------------------------------------- recording

def test_record_stop_keeps_an_episode_with_frames_and_duration(tmp_path):
    w, backend, recorder, clock = make_worker(thumb_dir=tmp_path)
    w.start_episode()
    assert w.session()["phase"] == "recording"
    for _ in range(5):
        clock.t += 1 / 30
        assert w.tick() is True
    s = w.stop_episode()
    assert s["phase"] == "idle"
    (ep,) = s["episodes"]
    assert ep["index"] == 0 and ep["frames"] == 5 and not ep["discarded"]
    assert ep["duration_s"] > 0
    assert recorder.saved == 1 and len(recorder.frames) == 5
    # every frame carried the task (language conditioning depends on it)
    assert all(t == "pick up the cube" for _, _, t in recorder.frames)
    # teleop reached the follower
    assert len(backend.applied) == 5


def test_teleop_runs_between_episodes_but_records_nothing():
    w, backend, recorder, _ = make_worker()
    assert w.tick() is False  # idle: arms move, dataset does not
    assert len(backend.applied) == 1 and recorder.frames == []


def test_commands_are_idempotent_as_the_contract_promises():
    w, _, recorder, clock = make_worker()
    w.start_episode()
    first = w.session()
    assert w.start_episode()["episodes"] == first["episodes"]  # start while recording
    clock.t += 0.1
    w.tick()
    w.stop_episode()
    before = w.session()
    assert w.stop_episode() == before  # stop while idle
    assert w.redo_episode() == before  # redo while idle
    assert recorder.saved == 1


def test_zero_frame_episode_is_surfaced_not_half_kept():
    w, _, recorder, _ = make_worker()
    w.start_episode()
    s = w.stop_episode()  # stop before any tick
    assert s["episodes"] == []
    assert "0 frames" in s["error"]
    assert recorder.saved == 0
    # the error clears on the next successful start
    assert w.start_episode()["error"] is None


def test_redo_drops_the_buffer_and_reuses_the_index():
    w, _, recorder, clock = make_worker()
    w.start_episode()
    clock.t += 0.1
    w.tick()
    s = w.redo_episode()
    assert s["phase"] == "idle" and s["episodes"] == []
    assert recorder.cleared == 1 and recorder.saved == 0
    s = w.start_episode()
    assert s["episodes"][-1]["index"] == 0  # redo did not burn index 0


def test_discard_marks_saved_episode_and_close_reports_it(tmp_path):
    w, backend, recorder, clock = make_worker(thumb_dir=tmp_path)
    for _ in range(2):
        w.start_episode()
        clock.t += 0.1
        w.tick()
        w.stop_episode()
    w.discard(0)
    s = w.session()
    assert [e["discarded"] for e in s["episodes"]] == [True, False]
    with pytest.raises(KeyError):
        w.discard(99)
    r = w.close()
    assert r["ok"] is True
    assert "1 episode(s) kept" in r["detail"] and "1 discarded" in r["detail"]
    assert r["discarded_indices"] == [0]
    assert recorder.finalized and backend.closed


def test_close_mid_recording_keeps_nothing_half_written():
    w, backend, recorder, clock = make_worker()
    w.start_episode()
    clock.t += 0.1
    w.tick()
    r = w.close()
    assert r["ok"] is True and "0 episode(s) kept" in r["detail"]
    assert recorder.cleared == 1 and recorder.saved == 0
    assert backend.closed
    # closed session: dataset nulls out, commands refuse, close is idempotent
    assert w.session()["dataset"] is None
    with pytest.raises(RuntimeError):
        w.start_episode()
    assert w.close()["detail"] == "session already closed"


def test_upload_failure_is_reported_but_dataset_survives():
    class PushBoom(FakeRecorder):
        def push_to_hub(self, repo_id=None):
            raise RuntimeError("hub said no")

    w, backend, recorder, clock = make_worker(recorder=PushBoom())
    w.start_episode()
    clock.t += 0.1
    w.tick()
    w.stop_episode()
    r = w.close(upload=True, repo_id="cagatay/remote")
    assert r["ok"] is False
    assert "saved but upload failed" in r["detail"]
    assert recorder.finalized  # local dataset was finalized before the push
    assert backend.closed


def test_first_frame_writes_thumbnails_with_contract_urls(tmp_path):
    w, _, _, clock = make_worker(thumb_dir=tmp_path)
    w.start_episode()
    clock.t += 0.1
    w.tick()
    clock.t += 0.1
    w.tick()
    s = w.stop_episode()
    (ep,) = s["episodes"]
    assert ep["thumbnails"] == {"top": "/api/record/thumb/0/top"}
    assert (tmp_path / "0_top.jpg").exists()


def test_thumbnail_helper_never_raises_on_junk(tmp_path):
    assert _save_thumbnail("not an image", tmp_path / "x.jpg") is False
    assert _save_thumbnail(np.zeros((4,)), tmp_path / "y.jpg") is False
