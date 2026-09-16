"""``start_cameras_recording`` / ``stop_cameras_recording`` dispatched as actions must not hold the lock their recorder renders under.

The recorder thread's ``render`` serializes its mjData read under
``Simulation._lock`` - the same RLock ``_dispatch_action`` holds around most
actions. Dispatched under that blanket lock:

* ``start_cameras_recording`` waited its whole readiness timeout (6 s for one
  camera) for a warmup render that was waiting for the lock start held, then
  logged "not ready" and returned success over a recorder that had captured
  nothing;
* ``stop_cameras_recording`` joined a thread blocked in ``render`` on the lock
  stop held, so the join expired every time: ``status="error"``, "did not stop
  within 5.0s ... render() call is most likely blocking", nothing encoded, the
  recording left registered.

Measured through ``Robot("so101", mode="sim")`` (the agent path) while the
same calls on a bare ``Simulation`` - no dispatch lock - worked. Pinned here:
both verbs are in ``_SELF_LOCKING_ACTIONS``; through ``_dispatch_action`` start
is ready well inside its timeout, frames are captured and stop encodes them;
start still refuses under the lock it takes itself; a camera that buffered
nothing is reported as "no clip written" with ``path=None`` instead of naming a
file that does not exist.
"""

from __future__ import annotations

import os
import time

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation


@pytest.fixture
def sim():
    s = Simulation(tool_name="cams_rec_lock_test", mesh=False)
    s.create_world()
    assert s.add_robot(name="so101", data_config="so101")["status"] == "success"
    yield s
    s.cleanup()


def _text(result: dict) -> str:
    return "\n".join(c["text"] for c in result["content"] if isinstance(c, dict) and "text" in c)


def _json(result: dict) -> dict:
    return next(c["json"] for c in result["content"] if isinstance(c, dict) and "json" in c)


def test_both_recorder_verbs_run_outside_the_blanket_lock():
    assert {"start_cameras_recording", "stop_cameras_recording"} <= Simulation._SELF_LOCKING_ACTIONS


def test_dispatched_start_is_ready_fast_and_dispatched_stop_encodes_frames(sim, tmp_path):
    t0 = time.monotonic()
    started = sim._dispatch_action(
        "start_cameras_recording",
        {"output_dir": str(tmp_path), "cameras": ["default"], "fps": 20, "width": 64, "height": 48},
    )
    elapsed = time.monotonic() - t0
    assert started["status"] == "success", started
    # The readiness timeout is 5 s + 1 s/camera; under the blanket lock the
    # whole of it elapsed. Ready means the warmup render got the lock.
    assert elapsed < 4.0, f"start took {elapsed:.1f}s - the warmup render is waiting for a lock start holds"

    for _ in range(8):
        sim._dispatch_action("step", {"n_steps": 10})
        time.sleep(0.1)

    t0 = time.monotonic()
    stopped = sim._dispatch_action("stop_cameras_recording", {})
    assert stopped["status"] == "success", stopped
    assert time.monotonic() - t0 < 4.0
    artifacts = _json(stopped)["artifacts"]
    assert len(artifacts) == 1 and artifacts[0]["frames"] >= 3, stopped
    assert os.path.isfile(artifacts[0]["path"]), artifacts
    assert sim._dispatch_action("get_cameras_recording_status", {})["status"] == "success"


def test_dispatched_start_still_refuses_what_it_refused_before(sim, tmp_path):
    result = sim._dispatch_action(
        "start_cameras_recording",
        {"output_dir": str(tmp_path), "cameras": ["no_such_camera"], "fps": 10},
    )
    assert result["status"] == "error", result
    assert getattr(sim, "_cams_rec_state", None) is None


def test_a_camera_with_no_frames_names_no_clip(sim, tmp_path):
    state = {
        "name": "rec_empty",
        "cameras": ["default"],
        "fps": 10,
        "buffers": {"default": []},
        "errors": {"default": 0},
        "paths": {"default": str(tmp_path / "rec_empty__default.mp4")},
        "output_dir": str(tmp_path),
        "started_mono": time.monotonic(),
    }
    result = sim._flush_cameras_recording_state(state)
    assert result["status"] == "success", result
    text = _text(result)
    assert "no clip written" in text, text
    assert "rec_empty__default.mp4" not in text, text
    artifact = _json(result)["artifacts"][0]
    assert artifact["path"] is None and artifact["frames"] == 0, artifact
    assert not os.path.exists(str(tmp_path / "rec_empty__default.mp4"))
