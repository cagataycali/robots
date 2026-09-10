# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""An MP4 request without an MP4 backend is refused as the missing dependency.

:func:`~strands_robots.rendering.encode_clip` probes ``imageio`` before it
writes, and ``imageio`` is not one dependency but two: the package itself, and
the plugin that actually encodes the container. ``imageio`` declares that plugin
-- ``imageio_ffmpeg`` -- as an optional extra of its own, so an install can
supply the module ``encode_clip`` imports and still have no MP4 writer behind
it. Every extra here declares both, but nothing stops a caller installing
``imageio`` on its own - and an extra that declared it alone once shipped.

With the plugin absent ``imageio`` does not refuse the ``.mp4`` request. It
falls through to whatever other plugin claims the container, and the libx264
knobs then reach a writer that has never heard of them, so the request fails as
``PyAVPlugin.write() got an unexpected keyword argument 'quality'`` -- or
``TiffWriter.write() got an unexpected keyword argument 'fps'`` when PyAV is
absent too. That is a ``TypeError`` naming a writer the caller never asked for,
and it is neither what ``encode_clip`` documents (``ImportError``) nor what its
callers handle:

* ``SimulationRendering._flush_cameras_recording_state`` (MuJoCo) has one
  ``except ImportError`` branch that keeps the recording registered with every
  buffered frame, so the caller can install the encoder and stop again. A
  ``TypeError`` misses it, lands in the best-effort ``except Exception``, and
  the frames are dropped under ``status="success"`` instead.
* ``IsaacSimulation.stop_cameras_recording`` documents that it never raises,
  and catches ``ImportError``, ``RuntimeError`` and ``ValueError``. A
  ``TypeError`` escapes the tool envelope entirely -- after the recording state
  has already been cleared, so no second stop can recover the buffers.

The rollout recorder is the third writer with the same shape:
``_RolloutVideoWriter.open`` probed ``imageio`` and then handed
``imageio.get_writer`` the same libx264 knobs, which is the path
``run_policy(video=...)`` takes, with an ``.mp4`` default.

So the container decides which modules have to be present, one place decides it
(:func:`~strands_robots.rendering.require_clip_encoder`), and every writer routes
its probe through that instead of repeating the rule -- which is how the rule
came to hold for one writer and not the next. GIF needs no plugin (Pillow writes
it, ``imageio`` hard-requires Pillow and this package depends on it outright),
which is the control here: the same absent plugin must leave a GIF encode
working.

The second half is the report. Both flushes used to answer with a fixed
``"imageio not installed. pip install imageio imageio-ffmpeg"`` line, which
names the wrong module whenever it is the plugin that is missing, and hands over
two unbounded distributions instead of the extra whose bounds this project owns.
They quote the encoder's own refusal now, which ``require_optional`` builds from
the module actually absent.
"""

from __future__ import annotations

import ast
import io
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest

imageio = pytest.importorskip("imageio", reason="imageio not installed - pip install imageio imageio-ffmpeg")

from strands_robots import utils  # noqa: E402
from strands_robots.rendering import encode_clip  # noqa: E402

#: The ``imageio`` plugin that writes MP4, spelled out here rather than imported
#: from the encoder: what a caller experiences is this module being absent, and
#: a pin that reads the name from the code under test would agree with it
#: whatever it said.
_MP4_BACKEND_MODULE = "imageio_ffmpeg"
#: The distribution that supplies it, and the extra of this package that pins
#: that distribution - what a caller must be told, rather than the import name.
_MP4_BACKEND_DISTRIBUTION = "imageio-ffmpeg"
_ENCODER_EXTRA = "sim-mujoco"

#: The claim the old fixed line made. Reachable only when ``imageio`` itself is
#: gone, so a refusal about the plugin must not repeat it.
_WRONG_DIAGNOSIS = "imageio not installed"


def _frames(n: int = 4, w: int = 32, h: int = 24) -> list[np.ndarray]:
    """``n`` distinct RGB frames of one shape."""
    return [np.full((h, w, 3), (i * 40) % 256, dtype=np.uint8) for i in range(n)]


@contextmanager
def _no_mp4_backend() -> Iterator[None]:
    """An install with ``imageio`` but not its MP4 plugin, for one block.

    Two steps, because :func:`~strands_robots.utils.require_optional` memoises a
    module it has already imported: the ``sys.modules`` entry is what makes the
    import fail, and dropping the cache entry is what stops an earlier import in
    the same session from answering instead. Mirrors
    ``tests/simulation/mujoco/test_camera_recorder_stop_reports_the_join.py``'s
    absent-``imageio`` block.
    """
    cached = utils._lazy_modules.pop(_MP4_BACKEND_MODULE, None)
    saved = sys.modules.get(_MP4_BACKEND_MODULE, "absent")
    sys.modules[_MP4_BACKEND_MODULE] = None  # type: ignore[assignment]
    try:
        yield
    finally:
        if saved == "absent":
            del sys.modules[_MP4_BACKEND_MODULE]
        else:
            sys.modules[_MP4_BACKEND_MODULE] = saved  # type: ignore[assignment]
        if cached is not None:
            utils._lazy_modules[_MP4_BACKEND_MODULE] = cached


@pytest.fixture
def no_mp4_backend() -> Iterator[None]:
    """:func:`_no_mp4_backend` for a whole cell."""
    with _no_mp4_backend():
        yield


class TestAnMp4RequestNeedsTheMp4Backend:
    """The plugin is a dependency of the request, refused before any writer opens."""

    # Every container this module routes to the MP4 writer: the suffix check is
    # "not .gif", so the case of the extension and the container name are both
    # part of the domain.
    @pytest.mark.parametrize("suffix", [".mp4", ".MP4", ".mkv"])
    def test_the_refusal_names_the_absent_plugin_and_the_extra_that_supplies_it(
        self, tmp_path: Path, no_mp4_backend: None, suffix: str
    ) -> None:
        """An ImportError about the plugin, not a TypeError about a foreign writer."""
        out = tmp_path / f"clip{suffix}"
        with pytest.raises(ImportError) as excinfo:
            encode_clip(_frames(), out, fps=10)
        exc = excinfo.value
        assert exc.name == _MP4_BACKEND_MODULE, (
            f"the refusal reports {exc.name!r}, so a caller cannot tell which module is missing"
        )
        text = str(exc)
        assert _MP4_BACKEND_MODULE in text, text
        assert f"pip install 'strands-robots[{_ENCODER_EXTRA}]'" in text, text
        assert _MP4_BACKEND_DISTRIBUTION in text, text
        assert _WRONG_DIAGNOSIS not in text, f"the refusal blames a module that is installed: {text}"
        assert not out.exists(), f"a refused encode left a file at {out}"

    def test_the_frames_are_not_read_before_the_refusal(self, tmp_path: Path, no_mp4_backend: None) -> None:
        """The caller keeps its buffer, so installing the plugin and retrying works.

        ``frames`` is an iterable: consuming it to build the frame list and only
        then discovering the encoder is unusable would exhaust a generator the
        caller cannot rewind.
        """
        read = 0

        def counted() -> Iterator[np.ndarray]:
            nonlocal read
            for frame in _frames():
                read += 1
                yield frame

        with pytest.raises(ImportError):
            encode_clip(counted(), tmp_path / "clip.mp4", fps=10)
        assert read == 0, f"the refusal consumed {read} frames from the caller's iterable"

    def test_a_present_backend_still_encodes_an_mp4(self, tmp_path: Path) -> None:
        """The premise: nothing about the guard refuses an install that can encode."""
        pytest.importorskip(_MP4_BACKEND_MODULE)
        out = encode_clip(_frames(), tmp_path / "clip.mp4", fps=10)
        assert out.exists() and out.stat().st_size > 0


class TestAGifNeedsNoMp4Backend:
    """The control: the guard is the container's, not the module's."""

    @pytest.mark.parametrize("suffix", [".gif", ".GIF"])
    def test_a_gif_encodes_with_the_mp4_plugin_absent(self, tmp_path: Path, no_mp4_backend: None, suffix: str) -> None:
        """Pillow writes GIF, so an absent MP4 plugin has nothing to do with it."""
        pytest.importorskip("PIL.Image")
        out = encode_clip(_frames(), tmp_path / f"clip{suffix}", fps=10)
        assert out.exists() and out.stat().st_size > 0


class TestEveryCameraFlushQuotesTheEncoder:
    """A flush reports the refusal it caught, not a fixed guess at its cause."""

    # The backend modules that flush buffered camera frames through
    # ``encode_clip`` - the same roster as
    # ``tests/simulation/test_camera_flush_encoder_refusal.py``.
    _FLUSH_MODULES = ("mujoco/rendering.py", "isaac/simulation.py")

    @staticmethod
    def _import_handlers(module_path: Path) -> list[ast.ExceptHandler]:
        """Every ``except ImportError`` inside a function that calls ``encode_clip``."""
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        handlers: list[ast.ExceptHandler] = []
        for func in ast.walk(tree):
            if not isinstance(func, ast.FunctionDef):
                continue
            calls_encoder = any(
                isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "encode_clip"
                for node in ast.walk(func)
            )
            if not calls_encoder:
                continue
            for node in ast.walk(func):
                if isinstance(node, ast.Try):
                    handlers += [
                        h for h in node.handlers if h.type is not None and "ImportError" in ast.unparse(h.type)
                    ]
        return handlers

    def test_no_flush_substitutes_its_own_diagnosis_for_the_encoders(self) -> None:
        """Two modules can be the missing one; only the encoder knows which."""
        import strands_robots.simulation as simulation_pkg

        package_dir = Path(simulation_pkg.__file__).parent
        checked = 0
        for relative in self._FLUSH_MODULES:
            module_path = package_dir / relative
            assert module_path.exists(), f"{relative} moved; update this guard"
            handlers = self._import_handlers(module_path)
            assert handlers, f"{relative} no longer guards the encoder import; update this guard"
            for handler in handlers:
                body = "\n".join(ast.unparse(stmt) for stmt in handler.body)
                assert handler.name, (
                    f"{relative}:{handler.lineno} discards the encoder's refusal, so it cannot report "
                    "which module is missing"
                )
                assert handler.name in body, f"{relative}:{handler.lineno} binds {handler.name!r} without reporting it"
                assert _WRONG_DIAGNOSIS not in body, (
                    f"{relative}:{handler.lineno} asserts {_WRONG_DIAGNOSIS!r}, which is false when it is "
                    "the MP4 plugin that is absent"
                )
                checked += 1
        assert checked >= len(self._FLUSH_MODULES)


def _render_result(width: int, height: int) -> dict[str, Any]:
    """A ``render()`` envelope carrying a real PNG the recorder can decode.

    A gradient rather than a flat fill: the recorder's warmup discards frames
    whose per-column standard deviation reads as a cold GL gradient. Same shape
    as ``tests/simulation/mujoco/test_daemon_camera_recording.py``'s harness, so
    no GL context is needed.
    """
    from PIL import Image

    row = np.linspace(0, 255, width, dtype=np.uint8)
    arr = np.repeat(row[None, :], height, axis=0)
    arr = np.stack([arr, arr[::-1], arr], axis=-1).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return {
        "status": "success",
        "content": [
            {"text": f"{width}x{height}"},
            {"image": {"format": "png", "source": {"bytes": buf.getvalue()}}},
        ],
    }


class TestTheMujocoFlushKeepsTheFramesItCannotEncode:
    """The consequence at the caller, on a real daemon recording."""

    @pytest.fixture
    def recording(self, tmp_path: Path) -> Iterator[Any]:
        """A started camera recording holding real buffered frames."""
        pytest.importorskip("mujoco", reason="mujoco not installed - pip install strands-robots[sim-mujoco]")
        from strands_robots.simulation import Simulation

        sim = Simulation()
        sim.create_world()
        sim.add_robot("arm", data_config="so101", position=[0.0, 0.0, 0.0])
        sim.add_camera("cam_a", position=[-0.3, -0.3, 0.4], target=[0.0, 0.0, 0.1])
        sim.render = lambda camera_name, width=None, height=None, **_kw: _render_result(  # type: ignore[method-assign]
            width or 32, height or 24
        )
        started = sim.start_cameras_recording(cameras=["cam_a"], output_dir=str(tmp_path), fps=30, name="clip")
        assert started["status"] == "success", started
        deadline = time.monotonic() + 30.0
        while len((sim._cams_rec_state or {}).get("buffers", {}).get("cam_a", [])) < 3:
            assert time.monotonic() < deadline, "premise: the recorder buffers frames"
            time.sleep(0.02)
        try:
            yield sim
        finally:
            sim.stop_cameras_recording()
            sim.destroy()

    def test_the_stop_refuses_and_leaves_the_recording_registered(self, recording: Any, tmp_path: Path) -> None:
        """The buffers survive an install the encoder cannot use, and are recoverable."""
        with _no_mp4_backend():
            result = recording.stop_cameras_recording()

        assert result["status"] == "error", result
        payload = next(block["json"] for block in result["content"] if "json" in block)
        assert payload["stopped"] is False, payload
        assert payload["buffered_frames"]["cam_a"] >= 3, payload
        text = result["content"][0]["text"]
        assert _MP4_BACKEND_MODULE in text, text
        assert f"pip install 'strands-robots[{_ENCODER_EXTRA}]'" in text, text
        assert _WRONG_DIAGNOSIS not in text, f"the flush blames a module that is installed: {text}"
        assert not list(tmp_path.glob("*.mp4")), "a refused flush wrote a clip"
        assert recording._cams_rec_state is not None, "the recording was deregistered with its frames unencoded"

    def test_a_second_stop_with_the_backend_present_encodes_the_same_frames(
        self, recording: Any, tmp_path: Path
    ) -> None:
        """What the refusal promises: install the plugin, stop again, get the clip."""
        pytest.importorskip(_MP4_BACKEND_MODULE)
        with _no_mp4_backend():
            buffered = recording.stop_cameras_recording()
        frames = next(b["json"] for b in buffered["content"] if "json" in b)["buffered_frames"]["cam_a"]

        result = recording.stop_cameras_recording()

        assert result["status"] == "success", result
        clips = list(tmp_path.glob("*.mp4"))
        assert len(clips) == 1 and clips[0].stat().st_size > 0, clips
        written = next(b["json"] for b in result["content"] if "json" in b)
        assert written["artifacts"][0]["frames"] == frames, written


class TestTheRolloutRecorderNeedsItToo:
    """``run_policy(video=...)`` writes MP4 through the same libx264 knobs."""

    def test_a_rollout_reports_the_absent_plugin_and_writes_nothing(self, tmp_path: Path, no_mp4_backend: None) -> None:
        """The envelope names the module and the extra, before the loop runs.

        ``_RolloutVideoWriter.open`` returns every other setup failure - a bad
        path, an unrenderable camera - as this envelope, so the one condition
        that is a fact about the install answers the same way. Previously the
        rollout ran, the foreign writer refused ``quality`` mid-loop, and the
        caller was told ``Policy failed: PyAVPlugin.write() got an unexpected
        keyword argument 'quality'`` beside a 0-byte MP4.
        """
        pytest.importorskip("mujoco", reason="mujoco not installed - pip install strands-robots[sim-mujoco]")
        from strands_robots.simulation import Simulation

        sim = Simulation()
        sim.create_world()
        sim.add_robot("arm", data_config="so101", position=[0.0, 0.0, 0.0])
        sim.add_camera("cam_a", position=[-0.3, -0.3, 0.4], target=[0.0, 0.0, 0.1])
        try:
            result = sim.run_policy(
                robot_name="arm",
                policy_provider="mock",
                n_steps=6,
                control_frequency=20.0,
                video={"path": str(tmp_path / "rollout.mp4"), "fps": 20, "camera": "cam_a"},
            )
        finally:
            sim.destroy()

        assert result["status"] == "error", result
        text = result["content"][0]["text"]
        assert _MP4_BACKEND_MODULE in text, text
        assert f"pip install 'strands-robots[{_ENCODER_EXTRA}]'" in text, text
        assert _WRONG_DIAGNOSIS not in text, f"the rollout blames a module that is installed: {text}"
        assert not list(tmp_path.iterdir()), f"a refused rollout left files behind: {list(tmp_path.iterdir())}"


class TestOneOwnerDecidesWhichEncoderModulesAContainerNeeds:
    """The rule lives in one function, so a new writer cannot get it wrong."""

    def test_no_writer_probes_imageio_on_its_own(self) -> None:
        """Every ``require_optional("imageio")`` in the package is the shared owner's."""
        import strands_robots

        package_dir = Path(strands_robots.__file__).parent
        owner = package_dir / "rendering" / "video.py"
        offenders = []
        for module_path in sorted(package_dir.rglob("*.py")):
            tree = ast.parse(module_path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                    continue
                if node.func.id != "require_optional" or not node.args:
                    continue
                first = node.args[0]
                if isinstance(first, ast.Constant) and first.value == "imageio" and module_path != owner:
                    offenders.append(f"{module_path.relative_to(package_dir)}:{node.lineno}")
        assert not offenders, (
            "these sites probe imageio without deciding which backend their container needs, "
            f"so an MP4 request reaches a writer that refuses its knobs: {offenders}"
        )
