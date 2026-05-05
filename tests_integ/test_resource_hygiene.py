"""T4/D3: Resource hygiene - no RSS leak on create_world/destroy cycles or
repeated render at fixed dims.

Skipped when psutil isn't installed. Runs as part of `hatch run test-integ`.
Marked slow because it does 50+ cycles and ~500 renders.
"""

from __future__ import annotations

import gc
import importlib.util

import pytest

from strands_robots.simulation.mujoco.simulation import Simulation

psutil = None
if importlib.util.find_spec("psutil") is not None:
    import psutil  # type: ignore  # noqa: F401

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(psutil is None, reason="psutil not installed"),
]


def _rss_mb() -> float:
    import psutil as _ps

    return _ps.Process().memory_info().rss / (1024 * 1024)


class TestResourceHygiene:
    def test_no_leak_on_create_destroy_cycle(self):
        """50 create_world -> destroy cycles should not grow RSS by more than ~50 MB."""
        sim = Simulation(tool_name="hygiene_cycle", mesh=False)
        # warmup
        sim.create_world()
        sim.destroy()
        gc.collect()

        start_rss = _rss_mb()
        for _ in range(50):
            sim.create_world()
            sim.destroy()
        gc.collect()
        end_rss = _rss_mb()
        sim.cleanup()

        delta = end_rss - start_rss
        assert delta < 50.0, f"RSS grew by {delta:.1f} MB over 50 create/destroy cycles"

    def test_no_leak_on_many_renders(self):
        """500 renders at fixed dims should not grow RSS by more than ~100 MB.

        Renderer reuse must kick in (same (w,h) key) so we don't allocate a
        new GL context per call.
        """
        sim = Simulation(tool_name="hygiene_render", mesh=False)
        sim.create_world()

        # warmup
        sim.render(width=320, height=240)
        gc.collect()

        start_rss = _rss_mb()
        for _ in range(500):
            sim.render(width=320, height=240)
        gc.collect()
        end_rss = _rss_mb()
        sim.cleanup()

        delta = end_rss - start_rss
        assert delta < 100.0, f"RSS grew by {delta:.1f} MB over 500 renders"


class TestRendererCacheBehaviour:
    """Unit-level checks that the TLS cache is cleared on destroy/cleanup."""

    def test_destroy_empties_main_thread_renderer_cache(self):
        sim = Simulation(tool_name="hygiene_tls", mesh=False)
        sim.create_world()
        # Touch the renderer cache on the main thread.
        sim.render(width=160, height=120)
        # Inspect TLS cache is non-empty
        renderers = getattr(sim._renderer_tls, "renderers", {})
        assert renderers, "expected a renderer cached on the main thread"

        sim.destroy()
        renderers_after = getattr(sim._renderer_tls, "renderers", {})
        assert not renderers_after, "destroy() should have closed and cleared the main-thread renderer cache"
        sim.cleanup()

    def test_render_reuses_renderer_for_same_dims(self):
        sim = Simulation(tool_name="hygiene_reuse", mesh=False)
        sim.create_world()
        sim.render(width=160, height=120)
        rcache = sim._renderer_tls.renderers
        r_first = rcache[(160, 120)]
        sim.render(width=160, height=120)
        r_second = rcache[(160, 120)]
        assert r_first is r_second, "renderer should be reused for identical (w,h)"
        sim.cleanup()
