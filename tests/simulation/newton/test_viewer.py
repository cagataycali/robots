"""Live interactive viewer parity for the Newton backend.

Exercises NewtonSimEngine.open_viewer / close_viewer, which bring the Newton
backend to viewer parity with the MuJoCo backend. The real Newton viewers
(ViewerGL window, ViewerViser browser dashboard, ViewerNull sink) are replaced
with lightweight fakes so the tests never open a window or bind a port; the
engine itself is built against real Newton + Warp (gated below) so the model
binding and per-step frame sync run against a genuine model/state.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import pytest

_HAS_NEWTON = importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None

pytestmark = pytest.mark.skipif(not _HAS_NEWTON, reason="newton/warp not installed")


class _FakeViewer:
    """Records the viewer contract NewtonSimEngine drives (no window/port)."""

    kind = "base"

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.model: Any = None
        self.frames = 0
        self.closed = False
        self._running = True
        self.url = "http://localhost:8080/fake"

    def set_model(self, model: Any, max_worlds: int | None = None) -> None:
        self.model = model

    def is_running(self) -> bool:
        return self._running

    def begin_frame(self, time: float) -> None:
        pass

    def log_state(self, state: Any) -> None:
        self.frames += 1

    def end_frame(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class _FakeGL(_FakeViewer):
    kind = "gl"


class _FakeViser(_FakeViewer):
    kind = "viser"


class _FakeNull(_FakeViewer):
    kind = "null"


@pytest.fixture
def engine():
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    sim = NewtonSimEngine(solver="mujoco")
    sim.create_world()
    sim.add_robot("so100")
    yield sim
    sim.destroy()


@pytest.fixture
def patched_viewers(engine, monkeypatch):
    """Replace the Newton viewer classes with fakes (no window / no port).

    Patches ``ViewerGL`` / ``ViewerViser`` / ``ViewerNull`` on the live
    ``newton.viewer`` module reached through the engine, and tracks
    instantiation counts so tests can assert which viewer was chosen and that
    the headless / error paths never construct a real viewer.
    """
    vmod = engine._nt.viewer
    counts = {"gl": 0, "viser": 0, "null": 0}

    def _factory(cls):
        def _make(*args: Any, **kwargs: Any):
            counts[cls.kind] += 1
            return cls(*args, **kwargs)

        return _make

    monkeypatch.setattr(vmod, "ViewerGL", _factory(_FakeGL))
    monkeypatch.setattr(vmod, "ViewerViser", _factory(_FakeViser))
    monkeypatch.setattr(vmod, "ViewerNull", _factory(_FakeNull))
    return counts


class TestOpenViewer:
    def test_null_binds_model_and_primes_a_frame(self, engine, patched_viewers):
        result = engine.open_viewer("null")
        assert result["status"] == "success"
        assert engine._viewer is not None
        # Bound to the live finalized model, and one frame primed on open.
        assert engine._viewer.model is engine._model
        assert engine._viewer.frames >= 1
        assert patched_viewers["null"] == 1

    def test_viser_reports_browser_url_and_port(self, engine, patched_viewers):
        result = engine.open_viewer("viser", port=9123)
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "viser" in text and engine._viewer.url in text
        # Port threaded through to the real constructor kwargs.
        assert engine._viewer.kwargs.get("port") == 9123
        assert patched_viewers["viser"] == 1

    def test_gl_with_display_opens_window(self, engine, patched_viewers, monkeypatch):
        monkeypatch.setenv("DISPLAY", ":0")
        result = engine.open_viewer("gl", width=800, height=600)
        assert result["status"] == "success"
        assert engine._viewer.kwargs == {"width": 800, "height": 600}
        assert patched_viewers["gl"] == 1

    def test_auto_selects_gl_when_display_present(self, engine, patched_viewers, monkeypatch):
        monkeypatch.setenv("DISPLAY", ":0")
        engine.open_viewer("auto")
        assert engine._viewer_kind == "gl"
        assert patched_viewers["gl"] == 1 and patched_viewers["viser"] == 0

    def test_auto_selects_viser_when_headless(self, engine, patched_viewers, monkeypatch):
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        engine.open_viewer("auto")
        assert engine._viewer_kind == "viser"
        assert patched_viewers["viser"] == 1 and patched_viewers["gl"] == 0


class TestOpenViewerErrors:
    def test_gl_headless_errors_without_constructing(self, engine, patched_viewers, monkeypatch):
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        result = engine.open_viewer("gl")
        assert result["status"] == "error"
        assert "viser" in result["content"][0]["text"]
        # No real viewer constructed on the error path.
        assert patched_viewers["gl"] == 0
        assert engine._viewer is None

    def test_unknown_viewer_kind_errors(self, engine, patched_viewers):
        result = engine.open_viewer("hologram")
        assert result["status"] == "error"
        assert "hologram" in result["content"][0]["text"]

    def test_no_world_errors(self):
        from strands_robots.simulation.newton.simulation import NewtonSimEngine

        sim = NewtonSimEngine(solver="mujoco")
        assert sim.open_viewer("null")["status"] == "error"

    def test_double_open_is_idempotent(self, engine, patched_viewers):
        engine.open_viewer("null")
        first = engine._viewer
        result = engine.open_viewer("null")
        assert result["status"] == "success"
        assert "already open" in result["content"][0]["text"]
        assert engine._viewer is first
        assert patched_viewers["null"] == 1


class TestViewerSync:
    def test_step_pushes_frames(self, engine, patched_viewers):
        engine.open_viewer("null")
        before = engine._viewer.frames
        engine.step(3)
        assert engine._viewer.frames > before

    def test_send_action_pushes_a_frame(self, engine, patched_viewers):
        engine.open_viewer("null")
        before = engine._viewer.frames
        engine.send_action({"Rotation": 0.3}, robot_name="so100", n_substeps=2)
        assert engine._viewer.frames > before

    def test_dead_viewer_is_released_without_breaking_stepping(self, engine, patched_viewers):
        engine.open_viewer("null")
        engine._viewer._running = False  # simulate user closing the window
        result = engine.step(1)
        assert result["status"] == "success"
        assert engine._viewer is None

    def test_rebuild_rebinds_viewer_to_new_model(self, engine, patched_viewers):
        engine.open_viewer("null")
        engine.reset()  # triggers a model rebuild
        assert engine._viewer is not None
        assert engine._viewer.model is engine._model


class TestCloseViewer:
    def test_close_releases_handle(self, engine, patched_viewers):
        engine.open_viewer("null")
        handle = engine._viewer
        result = engine.close_viewer()
        assert result["status"] == "success"
        assert handle.closed is True
        assert engine._viewer is None

    def test_close_without_open_is_noop(self, engine, patched_viewers):
        assert engine.close_viewer()["status"] == "success"

    def test_destroy_closes_open_viewer(self, engine, patched_viewers):
        engine.open_viewer("null")
        handle = engine._viewer
        engine.destroy()
        assert handle.closed is True
        assert engine._viewer is None


class TestDescribeAdvertisesViewer:
    def test_describe_lists_viewer_methods(self, engine, patched_viewers):
        methods = engine.describe()["methods"]
        assert "open_viewer" in methods
        assert "close_viewer" in methods
