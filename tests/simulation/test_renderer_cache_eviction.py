"""The per-thread MuJoCo renderer cache is bounded and evicts oldest-first.

``RenderingMixin._get_renderer`` caches a ``mujoco.Renderer`` per
``(width, height)`` on a ``threading.local`` (each renderer binds a GL context
to its creating thread). Without a bound, a caller that renders at many
resolutions - a common pattern when sweeping preview / record / eval sizes -
would accumulate GL contexts for the lifetime of the ``Simulation`` and leak
GPU memory. The cache therefore holds at most four renderers per thread and
evicts the first-inserted (FIFO) one when a fifth distinct resolution arrives;
a GL driver that fails to free the evicted context must not break the render.

These tests pin that contract through the public ``render`` surface, counting
``mujoco.Renderer`` construction (the observable resource-allocation side
effect) rather than inspecting the private cache:

* distinct resolutions build one renderer each;
* re-rendering a cached resolution reuses it (no new build);
* exceeding the cap evicts the oldest first, so re-rendering the evicted
  resolution rebuilds it while a still-cached one does not (proves FIFO and
  that the cache stays bounded);
* a ``close()`` that raises while evicting is swallowed - the render succeeds.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

import mujoco as mj  # noqa: E402

from strands_robots.simulation.mujoco.backend import _can_render  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

requires_gl = pytest.mark.skipif(
    not _can_render(),
    reason="No OpenGL context available (EGL/OSMesa required for offscreen rendering)",
)

ARM_XML = """
<mujoco model="arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base">
      <joint name="pan" type="hinge" axis="0 0 1"/>
      <geom type="cylinder" size="0.05 0.05"/>
    </body>
    <camera name="side" pos="0.8 -0.8 0.4" xyaxes="0.707 0.707 0 -0.2 0.2 0.96"/>
  </worldbody>
  <actuator>
    <position name="pan_act" joint="pan" kp="30"/>
  </actuator>
</mujoco>
"""

_CAM = "arm1/side"
# Five distinct, small offscreen sizes (all well under any framebuffer cap).
# The cache bound is four, so _E forces an eviction of the oldest (_A).
_A, _B, _C, _D, _E = (32, 32), (48, 48), (64, 64), (80, 80), (96, 96)


@pytest.fixture
def sim_with_arm(tmp_path):
    xml_path = tmp_path / "arm.xml"
    xml_path.write_text(ARM_XML)
    sim = Simulation(tool_name="renderer_cache", mesh=False)
    try:
        sim.create_world()
        r = sim.add_robot(name="arm1", urdf_path=str(xml_path))
        assert r["status"] == "success", r
        yield sim
    finally:
        sim.cleanup(policy_stop_timeout=0.5)


@pytest.fixture
def build_counter(monkeypatch):
    """Count ``mujoco.Renderer`` constructions (the cached GL resource)."""
    orig_init = mj.Renderer.__init__
    calls = {"n": 0}

    def counting_init(self, *args, **kwargs):
        calls["n"] += 1
        return orig_init(self, *args, **kwargs)

    monkeypatch.setattr(mj.Renderer, "__init__", counting_init)
    return calls


def _render(sim, size):
    r = sim.render(camera_name=_CAM, width=size[0], height=size[1])
    assert r["status"] == "success", r
    return r


@requires_gl
class TestRendererCacheEviction:
    def test_distinct_resolutions_each_build_one_renderer(self, sim_with_arm, build_counter):
        for size in (_A, _B, _C, _D):
            _render(sim_with_arm, size)
        assert build_counter["n"] == 4

    def test_cached_resolution_is_reused_not_rebuilt(self, sim_with_arm, build_counter):
        for size in (_A, _B, _C, _D):
            _render(sim_with_arm, size)
        # Re-render the same four: all cached, so no renderer is constructed.
        for size in (_A, _B, _C, _D):
            _render(sim_with_arm, size)
        assert build_counter["n"] == 4

    def test_exceeding_cap_evicts_oldest_first_in_first_out(self, sim_with_arm, build_counter):
        for size in (_A, _B, _C, _D):
            _render(sim_with_arm, size)
        # Fifth distinct resolution: the cache is at the cap of four, so the
        # oldest (_A) is evicted rather than the cache growing to five.
        _render(sim_with_arm, _E)
        assert build_counter["n"] == 5

        # A still-cached resolution reuses its renderer (no rebuild)...
        _render(sim_with_arm, _D)
        assert build_counter["n"] == 5

        # ...while the evicted oldest (_A) must be rebuilt, proving it was the
        # one dropped (first-inserted-first-out), not a more-recent resolution.
        _render(sim_with_arm, _A)
        assert build_counter["n"] == 6

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
    def test_close_failure_during_eviction_is_fail_soft(self, sim_with_arm, monkeypatch):
        """A GL context that fails to free on eviction must not break rendering.

        Eviction best-effort-closes the evicted renderer; if the driver's
        ``close()`` raises, the exception is swallowed so the render that
        triggered the eviction still returns a frame.
        """
        for size in (_A, _B, _C, _D):
            _render(sim_with_arm, size)

        def _boom(self):
            raise RuntimeError("GL context free failed")

        monkeypatch.setattr(mj.Renderer, "close", _boom)

        # The fifth resolution evicts _A; its close() now raises but the
        # render must still succeed.
        result = sim_with_arm.render(camera_name=_CAM, width=_E[0], height=_E[1])
        assert result["status"] == "success", result


@requires_gl
class TestCacheCapScalesWithTheScene:
    """The cap must fit a legitimately configured multi-camera rig.

    ``_get_sim_observation`` requests each camera's own configured resolution, so
    with a flat cap of four every ``get_observation`` on a rig with more distinct
    resolutions evicted and rebuilt GL contexts in a loop. A ``mujoco.Renderer``
    costs ~226 ms to construct against ~0.03 ms for a cache hit, so the loop went
    from render-bound to construction-bound with no log line at all. Measured::

        3 cams, 4 distinct keys ->     9.6 ms/obs
        4 cams, 5 distinct keys ->  1340.8 ms/obs   (140x)
        4 cams at ONE resolution ->   10.3 ms/obs   (isolates the cause to key count)

    After the fix the same 4-camera rig costs 10.6 ms/obs (126x faster) and a
    5-camera one 15.0 ms.

    The cap is derived from the scene's cameras plus headroom for the free camera
    and a video size - bounded by configuration, not by caller behaviour, so an
    unbounded ``render(width=..., height=...)`` sweep still evicts rather than
    leaking contexts (pinned below).
    """

    def test_cap_is_the_floor_for_a_single_camera_scene(self, sim_with_arm):
        # The inline arm has one camera; create_world adds the default. Both are
        # under the floor, so the historical cap of four is unchanged.
        assert sim_with_arm._max_renderers_per_thread() == 4

    def test_cap_grows_with_distinct_camera_resolutions(self, sim_with_arm):
        for index, (width, height) in enumerate([(224, 224), (128, 128), (96, 96), (64, 64)]):
            assert (
                sim_with_arm.add_camera(
                    f"extra{index}",
                    position=[0.3, 0.1 * index, 0.3],
                    target=[0.0, 0.0, 0.1],
                    width=width,
                    height=height,
                )["status"]
                == "success"
            )

        # Four new distinct resolutions plus the default camera's own, plus the
        # two-slot headroom for the free camera and a video size.
        assert sim_with_arm._max_renderers_per_thread() >= 6

    def test_cameras_sharing_one_resolution_do_not_inflate_the_cap(self, sim_with_arm):
        before = sim_with_arm._max_renderers_per_thread()
        for index in range(4):
            assert (
                sim_with_arm.add_camera(
                    f"same{index}",
                    position=[0.3, 0.1 * index, 0.3],
                    target=[0.0, 0.0, 0.1],
                    width=224,
                    height=224,
                )["status"]
                == "success"
            )

        # One distinct resolution added, so the cap moves by at most one.
        assert sim_with_arm._max_renderers_per_thread() <= before + 1

    def test_a_multi_resolution_scene_does_not_evict_across_observations(self, sim_with_arm, build_counter):
        """The defect, at the level it actually bit: get_observation in a loop."""
        for index, (width, height) in enumerate([(224, 224), (128, 128), (96, 96), (64, 64)]):
            assert (
                sim_with_arm.add_camera(
                    f"cam{index}",
                    position=[0.3, 0.1 * index, 0.3],
                    target=[0.0, 0.0, 0.1],
                    width=width,
                    height=height,
                )["status"]
                == "success"
            )

        sim_with_arm.get_observation("arm1")
        after_first = build_counter["n"]
        assert after_first > 0, "no renderer was built at all"

        for _ in range(3):
            sim_with_arm.get_observation("arm1")

        assert build_counter["n"] == after_first, (
            f"{build_counter['n'] - after_first} renderer rebuild(s) across three observations - "
            f"the cache is still thrashing"
        )

    def test_no_world_falls_back_to_the_floor(self):
        engine = Simulation(tool_name="renderer_cache_no_world", mesh=False)
        try:
            assert engine._max_renderers_per_thread() == 4
        finally:
            engine.cleanup(policy_stop_timeout=0.5)


@requires_gl
class TestThrashIsReported:
    """Silence was half the defect: a 140x slowdown produced no log line.

    The repo already treats a comparable ~100x render regression as
    warning-worthy (``_warn_if_software_rendering`` fires on llvmpipe with an
    actionable message); this follows the same one-shot pattern.
    """

    def test_re_requesting_an_evicted_resolution_warns(self, sim_with_arm, caplog):
        sizes = [(32 + 8 * i, 32 + 8 * i) for i in range(6)]
        with caplog.at_level("WARNING"):
            for _ in range(2):
                for size in sizes:
                    _render(sim_with_arm, size)

        warnings = [r.getMessage() for r in caplog.records if "renderer cache is thrashing" in r.getMessage()]
        assert warnings, [r.getMessage() for r in caplog.records]

    def test_the_warning_is_one_shot(self, sim_with_arm, caplog):
        sizes = [(32 + 8 * i, 32 + 8 * i) for i in range(6)]
        with caplog.at_level("WARNING"):
            for _ in range(4):
                for size in sizes:
                    _render(sim_with_arm, size)

        warnings = [r.getMessage() for r in caplog.records if "renderer cache is thrashing" in r.getMessage()]
        assert len(warnings) == 1, f"{len(warnings)} warnings for a repeated thrash"

    def test_the_warning_names_the_resolutions_and_is_actionable(self, sim_with_arm, caplog):
        sizes = [(32 + 8 * i, 32 + 8 * i) for i in range(6)]
        with caplog.at_level("WARNING"):
            for _ in range(2):
                for size in sizes:
                    _render(sim_with_arm, size)

        warning = next(r.getMessage() for r in caplog.records if "renderer cache is thrashing" in r.getMessage())
        assert "same width/height" in warning, warning
        assert warning.isascii()

    def test_a_first_fill_does_not_warn(self, sim_with_arm, caplog):
        """Filling an empty cache is not thrashing, even past the cap."""
        with caplog.at_level("WARNING"):
            for size in [(32 + 8 * i, 32 + 8 * i) for i in range(6)]:
                _render(sim_with_arm, size)

        assert not [r for r in caplog.records if "renderer cache is thrashing" in r.getMessage()]

    def test_a_recompile_does_not_look_like_thrash(self, sim_with_arm, caplog):
        """The cache is dropped on recompile; refilling it is not a thrash."""
        for size in (_A, _B):
            _render(sim_with_arm, size)

        with caplog.at_level("WARNING"):
            assert (
                sim_with_arm.add_object(
                    "box", shape="box", position=[0.2, 0.0, 0.05], size=[0.02, 0.02, 0.02], mass=0.1
                )["status"]
                == "success"
            )
            for size in (_A, _B):
                _render(sim_with_arm, size)

        assert not [r for r in caplog.records if "renderer cache is thrashing" in r.getMessage()]


@requires_gl
class TestUnboundedSweepStillEvicts:
    def test_a_resolution_sweep_stays_capped(self, sim_with_arm):
        """The cap must bound caller behaviour, not just scene configuration."""
        for size in [(32 + 8 * i, 32 + 8 * i) for i in range(12)]:
            _render(sim_with_arm, size)

        resident = len(getattr(sim_with_arm._renderer_tls, "renderers", {}))
        assert resident <= sim_with_arm._max_renderers_per_thread(), resident
