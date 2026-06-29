"""Graceful-degradation behavior of :class:`LiberoAdapter` helper methods.

These cover the defensive / optional-dependency branches that the
happy-path adapter tests do not exercise:

* ``_apply_init_jitter`` must silently skip when the sim does not expose
  ``move_object`` / ``get_body_state``, when a per-body state lookup
  raises, when the reported state is not a success payload, when no usable
  position is present, and when the ``move_object`` write itself raises -
  per-episode jitter is best-effort and must never abort an eval.
* ``_extract_init_targets`` ignores nodes that are not BDDL predicate
  combinators.
* ``_existing_camera_names`` falls back to the registry-only camera set
  when there is no compiled MuJoCo model, when ``mujoco`` is not
  importable, and when model-side camera enumeration raises.
* ``_action_controller_remediation`` returns the generic robosuite hint
  for an unrelated install failure (the numba/coverage clash branch is
  covered by the happy-path suite).
"""

from __future__ import annotations

import random
import sys
import types
from typing import Any, cast

from strands_robots.benchmarks.libero import LiberoAdapter
from strands_robots.benchmarks.libero.adapter import _extract_init_targets
from strands_robots.simulation.base import SimEngine

# BDDL whose (:init (on cube_1 table_1)) makes cube_1 the single jitter target.
PICK_CUBE_BDDL = """
(define (problem libero_spatial_pick_cube)
  (:domain kitchen)
  (:language "pick up the red cube and place it on the plate")
  (:objects cube_1 plate_1 table_1 - object)
  (:init (on cube_1 table_1))
  (:goal (on cube_1 plate_1)))
"""


def _adapter(jitter: float = 0.01) -> LiberoAdapter:
    return LiberoAdapter.from_text(PICK_CUBE_BDDL, init_jitter=jitter)


class _RecordingSim:
    """Sim stub recording ``move_object`` calls with a configurable state."""

    def __init__(self, body_state: object) -> None:
        self._body_state = body_state
        self.moves: list[tuple[str, list[float]]] = []
        self.move_should_raise = False

    def get_body_state(self, *, body_name: str) -> object:
        if isinstance(self._body_state, Exception):
            raise self._body_state
        return self._body_state

    def move_object(self, *, name: str, position: list[float]) -> dict[str, str]:
        self.moves.append((name, list(position)))
        if self.move_should_raise:
            raise RuntimeError("move failed")
        return {"status": "success"}


class TestApplyInitJitterDegradesGracefully:
    def test_skips_when_sim_lacks_move_object(self) -> None:
        adapter = _adapter()

        class _NoMove:
            def get_body_state(self, *, body_name: str) -> dict[str, str]:
                raise AssertionError("get_body_state must not be reached")

        # No move_object attribute -> returns before touching get_body_state.
        adapter._apply_init_jitter(cast(SimEngine, _NoMove()), random.Random(0))

    def test_skips_when_sim_lacks_get_body_state(self) -> None:
        adapter = _adapter()

        class _OnlyMove:
            def move_object(self, *, name: str, position: list[float]) -> None:
                raise AssertionError("move_object must not be reached")

        adapter._apply_init_jitter(cast(SimEngine, _OnlyMove()), random.Random(0))

    def test_continues_when_get_body_state_raises(self) -> None:
        adapter = _adapter()
        sim = _RecordingSim(RuntimeError("lookup boom"))
        adapter._apply_init_jitter(cast(SimEngine, sim), random.Random(0))
        assert sim.moves == []

    def test_skips_body_with_non_success_state(self) -> None:
        adapter = _adapter()
        sim = _RecordingSim({"status": "error"})
        adapter._apply_init_jitter(cast(SimEngine, sim), random.Random(0))
        assert sim.moves == []

    def test_skips_body_without_a_usable_position(self) -> None:
        adapter = _adapter()
        sim = _RecordingSim({"status": "success", "content": []})
        adapter._apply_init_jitter(cast(SimEngine, sim), random.Random(0))
        assert sim.moves == []

    def test_swallows_move_object_failure(self) -> None:
        adapter = _adapter()
        sim = _RecordingSim({"status": "success", "content": [{"json": {"position": [0.1, 0.2, 0.3]}}]})
        sim.move_should_raise = True
        # The write is attempted (cube_1 is the init subject) but its failure
        # is swallowed rather than aborting the eval.
        adapter._apply_init_jitter(cast(SimEngine, sim), random.Random(0))
        assert sim.moves and sim.moves[0][0] == "cube_1"


def test_extract_init_targets_ignores_non_predicate_nodes() -> None:
    assert _extract_init_targets(cast(Any, 12345)) == []


class TestExistingCameraNamesFallbacks:
    def test_registry_only_when_no_compiled_model(self) -> None:
        world = types.SimpleNamespace(cameras={"front": object()}, _model=None)
        sim = types.SimpleNamespace(_world=world)
        assert LiberoAdapter._existing_camera_names(cast(SimEngine, sim)) == {"front"}

    def test_falls_back_when_mujoco_unimportable(self, monkeypatch) -> None:
        world = types.SimpleNamespace(cameras={"front": object()}, _model=object())
        sim = types.SimpleNamespace(_world=world)
        # Setting the module entry to None makes `import mujoco` raise ImportError.
        monkeypatch.setitem(sys.modules, "mujoco", None)
        assert LiberoAdapter._existing_camera_names(cast(SimEngine, sim)) == {"front"}

    def test_swallows_model_side_enumeration_error(self) -> None:
        # A non-castable ncam makes the model-side enumeration raise; the error
        # is caught and the registry-only set is returned.
        world = types.SimpleNamespace(cameras={"front": object()}, _model=types.SimpleNamespace(ncam="not-an-int"))
        sim = types.SimpleNamespace(_world=world)
        assert LiberoAdapter._existing_camera_names(cast(SimEngine, sim)) == {"front"}


def test_action_controller_remediation_generic_hint() -> None:
    hint = LiberoAdapter._action_controller_remediation(RuntimeError("unrelated failure"))
    assert "robosuite" in hint
    assert "numba" not in hint
