# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``destroy()`` keeps a held camera recording instead of nulling it in silence.

``stop_cameras_recording`` leaves a raw-camera recording registered when the clip
encoder is absent: the frames are host-side NumPy arrays still in
``_cams_rec_state["buffers"]``, and the refusal
(:func:`~strands_robots.simulation.recording.encoder_absent_flush_refusal`)
promises they are recoverable by installing the encoder and calling the verb
again. ``destroy()`` used to set ``_cams_rec_state = None`` outright, discarding
exactly those frames under ``status="success"`` - the silent loss the refusal
had ruled out. The buffers are host arrays that survive the stage teardown and
encode fine, so teardown owes the same rule the flush does.

These cells pin that rule at teardown for the two ways a recording is still held
when ``destroy()`` runs - a stop refused for want of an encoder, and a recording
that was never stopped - against the real ``destroy`` / ``stop_cameras_recording``
on a skeleton engine built with ``__new__`` (no Isaac Kit runtime):

* encoder present  -> the held frames are encoded to their registered path
  before teardown, the files carry every buffered frame, and the registration is
  then cleared;
* encoder absent   -> a WARNING names the recording and its per-camera frame
  counts, and nothing is written - never a silent ``None``.
"""

from __future__ import annotations

import logging
import threading
import types

import numpy as np
import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac.config import IsaacConfig  # noqa: E402
from strands_robots.simulation.isaac.simulation import IsaacSimulation, _CameraState  # noqa: E402
from tests._blocked_module import blocked  # noqa: E402

_CAMERAS = ["front", "wrist"]
_FRAMES = 4


def _json_block(envelope: dict) -> dict:
    for block in envelope["content"]:
        if "json" in block:
            return block["json"]
    raise AssertionError(f"no json block in {envelope}")


class _FakeCameraHandle:
    def __init__(self, rgba: np.ndarray) -> None:
        self.rgba = rgba

    def get_rgba(self) -> np.ndarray:
        return self.rgba


def _engine(output_dir, *, width: int = 32, height: int = 24):
    """A skeleton engine that can both record cameras and be destroyed."""
    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._config = IsaacConfig(render_mode="headless")
    engine._lock = threading.RLock()
    # A world stub so destroy()'s teardown path runs without the Kit runtime.
    engine._world = types.SimpleNamespace(stop=lambda: None, clear_instance=lambda: None)
    engine._world_created = True
    engine._robots = {}
    engine._action_controllers = {}
    engine._objects = {}
    engine._prim_registry = []
    engine._cams_rec_state = None
    engine._recording_state_dict = {}
    engine._num_envs_active = 1
    engine._sim_time = 0.0
    engine._step_count = 0
    engine._main_tid = threading.get_ident()
    engine._obs_noise = None
    engine._obs_noise_rng = None
    engine._cameras = {}
    for name in _CAMERAS:
        cam = _CameraState(name=name, prim_path=f"/World/Cameras/{name}", width=width, height=height)
        cam.handle = _FakeCameraHandle(np.zeros((height, width, 4), dtype=np.uint8))
        engine._cameras[name] = cam
    # A gradient rather than a flat fill: an encoder that silently wrote nothing
    # would still produce a plausible file from constant frames.
    rgb = np.tile(np.linspace(0, 255, width, dtype=np.uint8)[None, :, None], (height, 1, 3))
    engine._render_frame = lambda camera_name, **_kw: (rgb, None, {})  # type: ignore[method-assign]
    return engine


def _buffered(engine) -> dict[str, int]:
    state = engine._cams_rec_state
    return {} if not state else {cam: len(state["buffers"][cam]) for cam in state["cameras"]}


def _arm_and_capture(engine, output_dir, frames: int = _FRAMES, name: str = "wedge") -> None:
    started = engine.start_cameras_recording(cameras=list(_CAMERAS), output_dir=str(output_dir), fps=30, name=name)
    assert started["status"] == "success", started
    on_frame = _json_block(started)["on_frame"]
    for step in range(frames):
        on_frame(step, {}, {})
    assert _buffered(engine) == {cam: frames for cam in _CAMERAS}, "premise: frames were buffered"


def _refused_stop(engine, output_dir) -> None:
    """A stop that refused for want of an encoder, leaving the frames registered."""
    _arm_and_capture(engine, output_dir)
    with blocked("imageio"):
        refused = engine.stop_cameras_recording()
    assert refused["status"] == "error", refused
    assert _buffered(engine) == {cam: _FRAMES for cam in _CAMERAS}, "the refusal kept the frames"


def _never_stopped(engine, output_dir) -> None:
    """A recording that was armed and captured frames but never stopped."""
    _arm_and_capture(engine, output_dir)


@pytest.fixture(params=[_refused_stop, _never_stopped], ids=["refused-stop", "never-stopped"])
def held_recording(request, tmp_path):
    """An engine holding a live camera recording by each route into destroy()."""
    engine = _engine(tmp_path)
    request.param(engine, tmp_path)
    return engine


class TestEncoderPresentEncodesBeforeTeardown:
    """Teardown's last chance: the held frames are encoded, then deregistered."""

    def test_the_held_frames_are_encoded(self, held_recording, tmp_path) -> None:
        imageio = pytest.importorskip("imageio.v2")
        state = held_recording._cams_rec_state
        paths = dict(state["paths"])

        assert held_recording.destroy()["status"] == "success"

        assert held_recording._cams_rec_state is None, "the encoded recording is deregistered"
        for path in paths.values():
            with imageio.get_reader(path) as reader:
                assert sum(1 for _ in reader) == _FRAMES, path


class TestEncoderAbsentNamesTheLoss:
    """No encoder to write with, so the loss is named - never a silent None."""

    def test_a_warning_names_the_recording_and_frame_counts(self, held_recording, caplog) -> None:
        with blocked("imageio"), caplog.at_level(logging.WARNING):
            assert held_recording.destroy()["status"] == "success"

        assert held_recording._cams_rec_state is None
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        loss = [m for m in warnings if "wedge" in m and "camera recording" in m]
        assert loss, f"no warning named the discarded camera recording: {warnings}"
        assert "front" in loss[0] and "wrist" in loss[0], loss[0]

    def test_nothing_is_written(self, held_recording, tmp_path) -> None:
        with blocked("imageio"):
            held_recording.destroy()

        assert list(tmp_path.glob("*.mp4")) == [], "nothing is encoded without an encoder"


class TestAFailingEncoderDoesNotAbortTeardown:
    """An encoder that dies mid-write is warned about; the world is still torn down."""

    @pytest.mark.parametrize(
        "error",
        [RuntimeError("no ffmpeg"), ValueError("bad frame"), OSError("[Errno 32] Broken pipe")],
        ids=lambda e: type(e).__name__,
    )
    def test_destroy_returns_success_and_clears_the_world(self, held_recording, monkeypatch, error) -> None:
        def failing_encode(*_args, **_kwargs):
            raise error

        monkeypatch.setattr("strands_robots.rendering.video.encode_clip", failing_encode)

        assert held_recording.destroy()["status"] == "success"
        assert held_recording._world is None and not held_recording._world_created
        assert held_recording._cams_rec_state is None
