# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""An Isaac camera recording is encoded whole, or it stays there to be encoded.

``IsaacSimulation.stop_cameras_recording`` hands each camera's buffer to
:func:`strands_robots.rendering.encode_clip`, whose encoder is two dependencies -
``imageio`` and the MP4 plugin ``imageio`` itself leaves optional - so the probe
can refuse before any writer is opened. Nothing is encoded and nothing is
touched, and the refusal quotes the encoder's own remedy: install it and call the
verb again.

That remedy is only followable while the frames are still reachable, and the
flush had already dropped ``_cams_rec_state`` before reading a single buffer. So
the refusal destroyed exactly what it said was recoverable: every camera's frames
went with the registration, a second call answered "Was not recording cameras."
about them, and a new ``start`` replaced the attribute under ``status="success"``.

The MuJoCo recorder decided this already - only a flush that encoded deregisters
(``_flush_and_deregister_cameras_recording``) - and words it in the shared
:func:`~strands_robots.simulation.recording.encoder_absent_flush_refusal`. These
cells pin the same three answers for Isaac: what the refusal reports, that a
later call encodes the frames it kept, and that a start cannot discard them.

The engine is a skeleton ``IsaacSimulation`` built with ``__new__`` (the fixture
shape ``test_cameras_recording_preflight_guards.py`` uses) so the recording
lifecycle runs without the Isaac Sim Kit runtime.
"""

from __future__ import annotations

import contextlib
import sys
import threading

import numpy as np
import pytest

from strands_robots import utils
from strands_robots.simulation.isaac.config import IsaacConfig
from strands_robots.simulation.isaac.simulation import IsaacSimulation, _CameraState

_CAMERAS = ["front", "wrist"]
_FRAMES = 4


@contextlib.contextmanager
def _blocked_encoder():
    """An install with no ``imageio``, for one block.

    Two steps, because :func:`~strands_robots.utils.require_optional` memoises a
    module it has already imported: the ``sys.modules`` entry is what makes the
    import fail, and dropping the cache entry is what stops an earlier import in
    the same session from answering instead.
    """
    cached = utils._lazy_modules.pop("imageio", None)
    sys.modules["imageio"] = None  # type: ignore[assignment]
    try:
        yield
    finally:
        del sys.modules["imageio"]
        if cached is not None:
            utils._lazy_modules["imageio"] = cached


@pytest.fixture
def no_encoder():
    """:func:`_blocked_encoder` for a whole cell."""
    with _blocked_encoder():
        yield


def _json_block(envelope: dict) -> dict:
    """The ``json`` payload of a tool envelope."""
    for block in envelope["content"]:
        if "json" in block:
            return block["json"]
    raise AssertionError(f"no json block in {envelope}")


class _FakeCameraHandle:
    """Stub RTX camera handle returning a fixed RGBA buffer."""

    def __init__(self, rgba: np.ndarray) -> None:
        self.rgba = rgba

    def get_rgba(self) -> np.ndarray:
        return self.rgba


class _Recorder:
    """A skeleton Isaac engine with a live camera recording and a frame source."""

    def __init__(self, output_dir, *, width: int = 32, height: int = 24) -> None:
        engine = IsaacSimulation.__new__(IsaacSimulation)
        engine._config = IsaacConfig()
        engine._lock = threading.RLock()
        engine._world = None
        engine._world_created = True
        engine._robots = {}
        engine._objects = {}
        engine._prim_registry = []
        engine._cams_rec_state = None
        engine._sim_time = 0.0
        engine._step_count = 0
        engine._main_tid = threading.get_ident()
        engine._cameras = {}
        for name in _CAMERAS:
            cam = _CameraState(name=name, prim_path=f"/World/Cameras/{name}", width=width, height=height)
            cam.handle = _FakeCameraHandle(np.zeros((height, width, 4), dtype=np.uint8))
            engine._cameras[name] = cam
        # A gradient rather than a flat fill: an encoder that silently wrote
        # nothing would still produce a plausible file from constant frames.
        self._rgb = np.tile(
            np.linspace(0, 255, width, dtype=np.uint8)[None, :, None],
            (height, 1, 3),
        )
        engine._render_frame = lambda camera_name, **_kw: (self._rgb, None, {})  # type: ignore[method-assign]
        self.sim = engine
        self.output_dir = output_dir

    def start(self, name: str = "wedge") -> dict:
        return self.sim.start_cameras_recording(
            cameras=list(_CAMERAS), output_dir=str(self.output_dir), fps=30, name=name
        )

    def buffer_a_few(self, frames: int = _FRAMES) -> int:
        """Arm a recording and capture ``frames`` frames on every camera."""
        started = self.start()
        assert started["status"] == "success", started
        on_frame = _json_block(started)["on_frame"]
        for step in range(frames):
            on_frame(step, {}, {})
        assert self.buffered() == {cam: frames for cam in _CAMERAS}, "premise: frames were buffered"
        return frames

    def buffered(self) -> dict[str, int]:
        state = self.sim._cams_rec_state
        if not state:
            return {}
        return {cam: len(state["buffers"][cam]) for cam in state["cameras"]}


@pytest.fixture
def recorder(tmp_path):
    return _Recorder(tmp_path)


class TestAFlushWithNoEncoder:
    """Nothing was written, so nothing may be dropped."""

    def test_the_refusal_reports_what_it_kept(self, recorder, no_encoder) -> None:
        """The verdict, and the counts that make the advice followable."""
        buffered = recorder.buffer_a_few()

        result = recorder.sim.stop_cameras_recording()

        assert result["status"] == "error", result
        text = result["content"][0]["text"]
        assert "pip install imageio" in text, text
        assert "stop_cameras_recording()" in text, text
        payload = _json_block(result)
        assert payload["stopped"] is False
        assert payload["recording"] == "wedge"
        assert payload["buffered_frames"] == {cam: buffered for cam in _CAMERAS}

    def test_the_frames_are_still_there(self, recorder, no_encoder, tmp_path) -> None:
        """The registration is the only route back to them, so it survives."""
        buffered = recorder.buffer_a_few()

        recorder.sim.stop_cameras_recording()

        assert recorder.sim._cams_rec_state is not None, "the registration is the only route back"
        assert recorder.buffered() == {cam: buffered for cam in _CAMERAS}
        assert list(tmp_path.glob("*.mp4")) == [], "nothing is encoded without an encoder"

    def test_a_later_call_encodes_them(self, recorder, tmp_path) -> None:
        """Following the message's own advice recovers every frame."""
        imageio = pytest.importorskip("imageio.v2")
        with _blocked_encoder():
            buffered = recorder.buffer_a_few()
            first = recorder.sim.stop_cameras_recording()
            assert first["status"] == "error", first

        second = recorder.sim.stop_cameras_recording()

        assert second["status"] == "success", second
        artifacts = _json_block(second)["artifacts"]
        assert [a["frames"] for a in artifacts] == [buffered] * len(_CAMERAS), artifacts
        for artifact in artifacts:
            with imageio.get_reader(artifact["path"]) as reader:
                assert sum(1 for _ in reader) == buffered
        assert recorder.sim._cams_rec_state is None, "the encoded recording is deregistered"

    def test_a_start_cannot_discard_them(self, recorder, no_encoder) -> None:
        """A start reads the registration, so it refuses instead of replacing."""
        buffered = recorder.buffer_a_few()
        recorder.sim.stop_cameras_recording()

        again = recorder.start(name="second")

        assert again["status"] == "error", again
        assert "stop_cameras_recording() first" in again["content"][0]["text"]
        assert recorder.sim._cams_rec_state["name"] == "wedge"
        assert recorder.buffered() == {cam: buffered for cam in _CAMERAS}


class TestAFlushThatEncoded:
    """The ordinary path is unchanged: encode, then deregister."""

    def test_it_deregisters_and_the_next_call_is_the_no_op(self, recorder, tmp_path) -> None:
        """Only a flush that encoded may report the recording gone."""
        pytest.importorskip("imageio.v2")
        buffered = recorder.buffer_a_few()

        result = recorder.sim.stop_cameras_recording()

        assert result["status"] == "success", result
        assert [a["frames"] for a in _json_block(result)["artifacts"]] == [buffered] * len(_CAMERAS)
        assert len(list(tmp_path.glob("*.mp4"))) == len(_CAMERAS)
        assert recorder.sim._cams_rec_state is None
        assert "Was not recording" in recorder.sim.stop_cameras_recording()["content"][0]["text"]

    def test_nothing_registered_is_still_the_no_op(self, recorder) -> None:
        """The idempotent success is about an absent registration, not an idle one."""
        result = recorder.sim.stop_cameras_recording()

        assert result["status"] == "success", result
        assert "Was not recording" in result["content"][0]["text"]


def test_both_recorders_word_one_absence_the_same_way() -> None:
    """One owner for the refusal, so the two recorders cannot drift apart."""
    from strands_robots.simulation.recording import encoder_absent_flush_refusal

    refusal = encoder_absent_flush_refusal(ImportError("no imageio"), "tag", {"front": 3})

    assert refusal["status"] == "error"
    assert "no imageio" in refusal["content"][0]["text"]
    assert "left registered holding {'front': 3}" in refusal["content"][0]["text"]
    assert _json_block(refusal) == {"stopped": False, "recording": "tag", "buffered_frames": {"front": 3}}
