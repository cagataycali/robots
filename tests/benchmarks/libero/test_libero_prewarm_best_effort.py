"""Best-effort contract of :meth:`LiberoAdapter.prewarm`.

``prewarm`` runs once before episode 0 to prime rendering: it wraps the
scene-supplied Panda, installs cameras / render options / the OSC action
controller, applies ``init_states[0]``, forwards the MjData and warms the GL
state. Every one of those steps only *degrades rendering* on failure - the
authoritative per-episode ``on_episode_start`` retries them - so prewarm must
never abort when a single step raises. It swallows each sub-step's exception,
logs it (WARNING for the priming steps, DEBUG for the throwaway warmup render),
and proceeds to the next step.

These tests pin that contract by forcing every sub-step to raise and asserting
prewarm still returns cleanly, runs every subsequent step, and emits one log
line per failed step.
"""

from __future__ import annotations

import logging
from typing import Any

from strands_robots.benchmarks.libero import LiberoAdapter

PICK_CUBE_BDDL = """
(define (problem libero_spatial_pick_cube)
  (:domain kitchen)
  (:language "pick up the cube")
  (:objects cube_1 - object)
  (:goal (grasped cube_1)))
"""

# Prewarm sub-steps in call order, paired with the substring their failure
# logs. All but the final warmup render log at WARNING; the warmup render is a
# throwaway GL-priming call and logs at DEBUG.
_SUBSTEPS = [
    ("_register_default_robot", "_register_default_robot raised", logging.WARNING),
    ("_install_libero_cameras", "_install_libero_cameras raised", logging.WARNING),
    ("_install_render_options", "_install_render_options raised", logging.WARNING),
    ("_install_action_controller", "_install_action_controller raised", logging.WARNING),
    ("_apply_init_state_for_prewarm", "init-state apply failed", logging.WARNING),
    ("_forward_mj_data", "mj_forward failed", logging.WARNING),
    ("_warmup_render", "warmup render failed", logging.DEBUG),
]


def _exploding_adapter(*, scene_path: str | None = "/tmp/libero_scene.xml") -> tuple[LiberoAdapter, list[str]]:
    """Build a real adapter, then replace every prewarm sub-step with a raiser
    that records that it ran. Returns the adapter and the shared calls list."""
    adapter = LiberoAdapter.from_text(
        PICK_CUBE_BDDL,
        scene_path=scene_path,
        install_cameras=True,
        auto_generate_scene=False,
    )
    calls: list[str] = []

    def _make_raiser(name: str) -> Any:
        def _step(_sim: Any) -> None:
            calls.append(name)
            raise RuntimeError(f"boom in {name}")

        return _step

    for name, _substring, _level in _SUBSTEPS:
        setattr(adapter, name, _make_raiser(name))
    return adapter, calls


def test_prewarm_swallows_every_substep_failure():
    """Every prewarm sub-step raising must not propagate: prewarm returns
    ``None`` and never re-raises, so a rendering-priming glitch cannot abort an
    eval before episode 0."""
    adapter, _calls = _exploding_adapter()

    result = adapter.prewarm(object())  # type: ignore[arg-type]

    assert result is None


def test_prewarm_runs_all_substeps_despite_failures():
    """A failure in one step must not short-circuit the rest: prewarm keeps
    going and invokes every subsequent priming step exactly once, in order."""
    adapter, calls = _exploding_adapter()

    adapter.prewarm(object())  # type: ignore[arg-type]

    assert calls == [name for name, _, _ in _SUBSTEPS]


def test_prewarm_logs_one_line_per_failed_substep(caplog):
    """Each failed step logs its own diagnostic at the documented level -
    WARNING for the priming steps, DEBUG for the throwaway warmup render - so
    the degraded rendering is visible in logs without crashing the run."""
    adapter, _calls = _exploding_adapter()

    with caplog.at_level(logging.DEBUG, logger="strands_robots.benchmarks.libero.adapter"):
        adapter.prewarm(object())  # type: ignore[arg-type]

    for _name, substring, level in _SUBSTEPS:
        matches = [r for r in caplog.records if substring in r.getMessage()]
        assert matches, f"expected a log line containing {substring!r}"
        assert any(r.levelno == level for r in matches), (
            f"{substring!r} should be logged at {logging.getLevelName(level)}"
        )


def test_prewarm_with_no_scene_path_is_a_noop():
    """An adapter built from a BDDL without a scene has nothing to prime, so
    prewarm returns immediately without touching any sub-step."""
    adapter, calls = _exploding_adapter(scene_path=None)

    adapter.prewarm(object())  # type: ignore[arg-type]

    assert calls == []
