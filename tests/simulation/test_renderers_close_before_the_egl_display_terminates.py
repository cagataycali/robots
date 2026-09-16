"""Renderers a script leaves alive are closed before mujoco tears the EGL display down.

``mujoco.Renderer`` frees its GL context in ``__del__``. Under ``MUJOCO_GL=egl``
the display those contexts belong to is terminated by an ``atexit`` hook mujoco
registers when it first opens the display. A renderer still alive at interpreter
exit - the main-thread cache of any script that never calls ``cleanup()``, which
is every ``Robot(...).run_policy(video=...)`` script - is finalised after that
hook ran, so its ``free()`` raises ``EGLError: EGL_NOT_INITIALIZED`` out of
``eglMakeCurrent``, and the context object's own ``__del__`` raises it again.
Measured rendering the v0.5.2 assets on an L40S and a Jetson Thor (mujoco
3.13.0): a successful rollout ended in some thirty lines of ignored traceback.

The mixin now records every renderer it builds with the thread that built it and
registers one ``atexit`` hook the first time - after mujoco's, so it runs first -
that closes the renderers owned by the thread running the hooks. Cross-thread
close is never attempted (a GL context is bound to its creating thread; closing
one from another SIGSEGVs in ``cgl.free()`` on macOS), and it is not needed: a
worker's thread-local cache is dropped when the worker ends, before the hooks.

The hook's contract is pinned here with stand-in renderers, so it holds with no
GL at all; the one GL cell checks that the renderers the public ``render``
surface builds are the ones recorded. The EGL behaviour itself was verified live
on the L40S: the repro's traceback is gone with this change and present without.
"""

from __future__ import annotations

import threading

import pytest

from strands_robots.simulation.mujoco import rendering


class _Renderer:
    """A weakly referenceable stand-in that records ``close()``."""

    def __init__(self, *, fail: bool = False) -> None:
        self.closed = 0
        self._fail = fail

    def close(self) -> None:
        self.closed += 1
        if self._fail:
            raise RuntimeError("EGL_NOT_INITIALIZED")


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> list:
    """A fresh live-renderer table and a captured ``atexit.register``."""
    registered: list = []
    monkeypatch.setattr(rendering, "_LIVE_RENDERERS", type(rendering._LIVE_RENDERERS)())
    monkeypatch.setattr(rendering, "_EXIT_HOOK_REGISTERED", False)
    monkeypatch.setattr(rendering.atexit, "register", lambda fn, *a, **k: registered.append(fn))
    return registered


class TestTheExitHook:
    def test_it_is_registered_once_on_the_first_renderer(self, registry: list) -> None:
        a, b = _Renderer(), _Renderer()

        rendering._track_renderer(a)
        rendering._track_renderer(b)

        assert registry == [rendering._close_renderers_at_exit]

    def test_it_closes_every_renderer_the_exiting_thread_built(self, registry: list) -> None:
        a, b = _Renderer(), _Renderer()
        rendering._track_renderer(a)
        rendering._track_renderer(b)

        rendering._close_renderers_at_exit()

        assert (a.closed, b.closed) == (1, 1)

    def test_it_leaves_another_threads_renderer_alone(self, registry: list) -> None:
        theirs = _Renderer()
        worker = threading.Thread(target=rendering._track_renderer, args=(theirs,))
        worker.start()
        worker.join()
        mine = _Renderer()
        rendering._track_renderer(mine)

        rendering._close_renderers_at_exit()

        assert mine.closed == 1
        assert theirs.closed == 0, "a GL context is bound to its creating thread"

    def test_a_renderer_already_collected_is_not_in_the_way(self, registry: list) -> None:
        rendering._track_renderer(_Renderer())  # dropped at the end of this expression
        survivor = _Renderer()
        rendering._track_renderer(survivor)

        rendering._close_renderers_at_exit()

        assert survivor.closed == 1
        assert len(rendering._LIVE_RENDERERS) == 1

    def test_a_close_that_raises_does_not_stop_the_others(self, registry: list) -> None:
        bad, good = _Renderer(fail=True), _Renderer()
        rendering._track_renderer(bad)
        rendering._track_renderer(good)

        rendering._close_renderers_at_exit()

        assert (bad.closed, good.closed) == (1, 1)

    def test_a_renderer_that_cannot_be_weakly_referenced_is_skipped(self, registry: list) -> None:
        rendering._track_renderer(object())  # no __weakref__ slot

        assert registry == []
        rendering._close_renderers_at_exit()


class TestTheRenderSurfaceIsWhatIsTracked:
    def test_a_renderer_built_through_render_is_recorded_for_this_thread(self, tmp_path, registry: list) -> None:
        pytest.importorskip("mujoco")
        from tests.simulation.mujoco._gl_probe import gl_available

        if not gl_available():
            pytest.skip("no offscreen GL context on this host")
        from strands_robots.simulation.mujoco.simulation import Simulation

        xml = tmp_path / "box.xml"
        xml.write_text(
            '<mujoco><worldbody><body name="b"><joint type="hinge"/><geom type="box" size="0.1 0.1 0.1"/></body>'
            '<camera name="cam" pos="1 0 0.5" xyaxes="0 1 0 -0.3 0 1"/></worldbody></mujoco>'
        )
        sim = Simulation(tool_name="exit_hook", mesh=False)
        try:
            sim.create_world()
            assert sim.add_robot(name="r", urdf_path=str(xml))["status"] == "success"
            result = sim.render(camera_name="r/cam", width=32, height=32)
            assert result["status"] == "success", result

            owners = list(rendering._LIVE_RENDERERS.values())
            assert owners == [threading.get_ident()]
            # Other modules register their own hooks during a render; ours is among them once.
            assert registry.count(rendering._close_renderers_at_exit) == 1
        finally:
            sim.cleanup(policy_stop_timeout=0.5)
