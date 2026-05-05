"""T4: Renderer TLS cache hygiene - destroy and cleanup empty the cache; same
(w,h) reuses an existing renderer. Unit-level (no RSS measurement; see
tests_integ/test_resource_hygiene.py for the process-memory checks)."""

from __future__ import annotations

import pytest

from strands_robots.simulation.mujoco.backend import _can_render  # noqa: E402

requires_gl = pytest.mark.skipif(
    not _can_render(),
    reason="No GL context available (headless CI without EGL/OSMesa)",
)

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    s = Simulation(tool_name="renderer_hygiene_test", mesh=False)
    yield s
    s.cleanup()


@requires_gl
class TestRendererTLSCache:
    def test_destroy_empties_main_thread_renderer_cache(self, sim):
        sim.create_world()
        sim.render(width=160, height=120)
        cached = getattr(sim._renderer_tls, "renderers", {})
        assert cached, "renderer should have been cached after render()"

        sim.destroy()
        cached_after = getattr(sim._renderer_tls, "renderers", {})
        assert not cached_after, "destroy() must empty the main-thread renderer cache"

    def test_render_reuses_renderer_for_identical_dims(self, sim):
        sim.create_world()
        sim.render(width=160, height=120)
        first = sim._renderer_tls.renderers[(160, 120)]
        sim.render(width=160, height=120)
        second = sim._renderer_tls.renderers[(160, 120)]
        assert first is second

    def test_render_creates_new_renderer_for_different_dims(self, sim):
        sim.create_world()
        sim.render(width=160, height=120)
        sim.render(width=320, height=240)
        keys = set(sim._renderer_tls.renderers.keys())
        assert (160, 120) in keys
        assert (320, 240) in keys

    def test_create_world_after_destroy_rebuilds_cache(self, sim):
        sim.create_world()
        sim.render(width=160, height=120)
        sim.destroy()
        sim.create_world()
        sim.render(width=160, height=120)
        assert (160, 120) in sim._renderer_tls.renderers
