"""Regression: ``_apply_canonical_state`` must run with the documented default rng.

The init-state branch of :meth:`LiberoAdapter._apply_canonical_state` samples a
row via ``random.Random()`` when the caller passes no ``rng`` (the documented
``rng=None`` default) on episodes 1+. ``random`` used to be imported only under
``if TYPE_CHECKING``, so at runtime the fallback raised ``NameError: name
'random' is not defined`` - a hard crash on the eval path any time a second
episode ran without an explicit rng. These tests pin that the default-rng path
works end to end against a real (tiny) compiled MuJoCo model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from strands_robots.benchmarks.libero import LiberoAdapter

mujoco = pytest.importorskip("mujoco")

PICK_CUBE_BDDL = """
(define (problem libero_spatial_pick_cube)
  (:domain kitchen)
  (:language "pick up the cube")
  (:objects cube_1 - object)
  (:goal (grasped cube_1)))
"""

# One slide joint -> nq == nv == 1, so a flat init state is [time, qpos, qvel].
_SCENE_XML = """
<mujoco>
  <worldbody>
    <body name="slider">
      <joint name="slide_x" type="slide" axis="1 0 0"/>
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


class _World:
    def __init__(self, model: Any, data: Any) -> None:
        self._model = model
        self._data = data
        self._backend_state: dict[str, Any] = {}


class _Sim:
    """Minimal sim exposing the compiled model/data the branch reads."""

    def __init__(self, world: Any) -> None:
        self._world = world


def _make_sim() -> Any:
    model = mujoco.MjModel.from_xml_string(_SCENE_XML)
    data = mujoco.MjData(model)
    return _Sim(_World(model, data))


def _make_adapter(init_states: np.ndarray) -> LiberoAdapter:
    return LiberoAdapter.from_text(
        PICK_CUBE_BDDL,
        auto_generate_scene=False,
        init_jitter=0.0,
        init_states=init_states,
    )


def test_apply_canonical_state_default_rng_episode1_no_nameerror():
    """Episode 1+ with default rng samples a row without raising NameError."""
    # Two rows so the RNG branch has a range to sample from.
    init_states = np.array([[0.0, 0.1, 0.0], [0.0, 0.2, 0.0]], dtype=np.float64)
    adapter = _make_adapter(init_states)
    sim = _make_sim()

    # Force the "episode 1+" branch that hits the ``random.Random()`` fallback.
    adapter._episode_count = 1

    # Must not raise (pre-fix: NameError: name 'random' is not defined).
    adapter._apply_canonical_state(sim)

    # A row was applied: qpos is one of the provided init states.
    assert float(sim._world._data.qpos[0]) in {0.1, 0.2}


def test_apply_canonical_state_seeded_rng_is_reproducible():
    """A supplied seed reproduces the same sampled init state (episode 1+)."""
    import random

    init_states = np.array(
        [[0.0, float(i), 0.0] for i in range(8)],
        dtype=np.float64,
    )

    def sample(seed: int) -> float:
        adapter = _make_adapter(init_states)
        adapter._episode_count = 1
        sim = _make_sim()
        adapter._apply_canonical_state(sim, random.Random(seed))
        return float(sim._world._data.qpos[0])

    assert sample(1234) == sample(1234)
