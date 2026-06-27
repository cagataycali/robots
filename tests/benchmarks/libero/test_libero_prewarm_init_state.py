"""Behavior tests for ``LiberoAdapter._apply_init_state_for_prewarm``.

Prewarm runs once BEFORE episode 0 and applies ``init_states[0]`` to the
MuJoCo buffer so the recorder's first frame shows LIBERO's canonical "ready"
pose rather than the joint-default zeros ``load_scene`` leaves behind (#168).

Unlike the per-episode ``_apply_init_state_branch`` (strict=True), prewarm is a
best-effort hint: every missing/invalid precondition degrades to a skip rather
than raising, so a non-MuJoCo backend or a mis-sized state never crashes the
eval pipeline. These tests pin that contract: the happy path writes
time/qpos/qvel and marks the one-shot prewarm flag, while each degrade branch
no-ops without mutating the buffer.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

from strands_robots.benchmarks.libero import LiberoAdapter

PICK_CUBE_BDDL = """
(define (problem libero_spatial_pick_cube)
  (:domain kitchen)
  (:language "pick up the cube")
  (:objects cube_1 - object)
  (:goal (grasped cube_1)))
"""


class _StubModel:
    def __init__(self, nq: int, nv: int) -> None:
        self.nq = nq
        self.nv = nv


class _StubData:
    def __init__(self, nq: int, nv: int) -> None:
        self.time = -1.0
        self.qpos = np.full(nq, -7.0, dtype=np.float64)
        self.qvel = np.full(nv, -7.0, dtype=np.float64)


class _StubWorld:
    def __init__(self, model: Any, data: Any) -> None:
        self._model = model
        self._data = data
        self._backend_state: dict[str, Any] = {}


class _StubSim:
    """Minimal stand-in exposing only what the method reads."""

    def __init__(self, world: Any, lock: Any = None) -> None:
        self._world = world
        if lock is not None:
            self._lock = lock


def _make_adapter(state: np.ndarray | None, scene_path: str = "/tmp/libero_scene.xml") -> LiberoAdapter:
    return LiberoAdapter.from_text(
        PICK_CUBE_BDDL,
        scene_path=scene_path,
        init_jitter=0.0,
        auto_generate_scene=False,
        init_states=state,
    )


def test_applies_init_state_and_marks_prewarm_flag():
    """Happy path: time/qpos/qvel are written from ``init_states[0]`` and the
    one-shot prewarm flag is recorded under the scene path."""
    nq, nv = 3, 2
    state = np.array([[0.5, 1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float64)  # 1 + nq + nv = 6
    adapter = _make_adapter(state, scene_path="/tmp/scene_a.xml")
    data = _StubData(nq, nv)
    sim = _StubSim(_StubWorld(_StubModel(nq, nv), data))

    adapter._apply_init_state_for_prewarm(sim)

    assert data.time == 0.5
    np.testing.assert_array_equal(data.qpos, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(data.qvel, [4.0, 5.0])
    assert sim._world._backend_state["libero_prewarm_path"] == "/tmp/scene_a.xml"


def test_applies_under_sim_lock_when_present():
    """When the sim exposes a ``_lock`` the mutation happens inside it; the
    result is identical to the lock-free path."""
    nq, nv = 2, 2
    state = np.array([[0.0, 9.0, 8.0, 7.0, 6.0]], dtype=np.float64)
    adapter = _make_adapter(state)
    data = _StubData(nq, nv)
    sim = _StubSim(_StubWorld(_StubModel(nq, nv), data), lock=threading.Lock())

    adapter._apply_init_state_for_prewarm(sim)

    np.testing.assert_array_equal(data.qpos, [9.0, 8.0])
    np.testing.assert_array_equal(data.qvel, [7.0, 6.0])


def test_no_init_states_is_a_silent_skip():
    """Bare-Panda construction (no ``init_states``) must not touch the buffer."""
    nq, nv = 2, 2
    adapter = _make_adapter(None)
    data = _StubData(nq, nv)
    sim = _StubSim(_StubWorld(_StubModel(nq, nv), data))

    adapter._apply_init_state_for_prewarm(sim)

    np.testing.assert_array_equal(data.qpos, [-7.0, -7.0])
    assert "libero_prewarm_path" not in sim._world._backend_state


def test_missing_world_is_a_skip():
    """A backend without ``_world`` degrades without raising."""
    adapter = _make_adapter(np.array([[0.0, 1.0, 2.0, 3.0, 4.0]], dtype=np.float64))

    class _Worldless:
        _world = None

    adapter._apply_init_state_for_prewarm(_Worldless())  # no exception


def test_missing_model_or_data_is_a_skip():
    """A world whose model/data are unset (non-MuJoCo backend) is a skip."""
    adapter = _make_adapter(np.array([[0.0, 1.0, 2.0, 3.0, 4.0]], dtype=np.float64))
    sim = _StubSim(_StubWorld(None, None))

    adapter._apply_init_state_for_prewarm(sim)  # no exception


def test_zero_dof_model_is_a_skip():
    """A compiled-but-empty model (nq==0) must not attempt the copy."""
    adapter = _make_adapter(np.array([[0.0, 1.0, 2.0, 3.0, 4.0]], dtype=np.float64))
    data = _StubData(0, 0)
    sim = _StubSim(_StubWorld(_StubModel(0, 0), data))

    adapter._apply_init_state_for_prewarm(sim)

    assert "libero_prewarm_path" not in sim._world._backend_state


def test_width_mismatch_warns_and_skips(caplog):
    """A state sized for a different model logs a WARNING (the call-order
    diagnostic) and skips rather than corrupting the buffer."""
    nq, nv = 3, 2  # expected width 6
    state = np.array([[0.0, 1.0, 2.0]], dtype=np.float64)  # width 3 - mismatched
    adapter = _make_adapter(state)
    data = _StubData(nq, nv)
    sim = _StubSim(_StubWorld(_StubModel(nq, nv), data))

    with caplog.at_level(logging.WARNING):
        adapter._apply_init_state_for_prewarm(sim)

    np.testing.assert_array_equal(data.qpos, [-7.0, -7.0, -7.0])
    assert "width" in caplog.text.lower()
    assert "libero_prewarm_path" not in sim._world._backend_state


_TRIVIAL_MJCF = """
<mujoco>
  <worldbody>
    <body name="link">
      <joint name="j" type="hinge" axis="0 0 1"/>
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_forward_mj_data_runs_mj_forward_on_valid_world():
    """``_forward_mj_data`` populates derived state via ``mj_forward`` when the
    world exposes a compiled model + data."""
    import mujoco

    model = mujoco.MjModel.from_xml_string(_TRIVIAL_MJCF)
    data = mujoco.MjData(model)
    adapter = _make_adapter(None)
    sim = _StubSim(_StubWorld(model, data))

    adapter._forward_mj_data(sim)  # no exception; xpos now populated

    assert data.xpos.shape[0] == model.nbody


def test_forward_mj_data_uses_lock_when_present():
    """The forward runs inside the sim lock when one is exposed."""
    import mujoco

    model = mujoco.MjModel.from_xml_string(_TRIVIAL_MJCF)
    data = mujoco.MjData(model)
    adapter = _make_adapter(None)
    sim = _StubSim(_StubWorld(model, data), lock=threading.Lock())

    adapter._forward_mj_data(sim)  # no exception


def test_forward_mj_data_skips_missing_world():
    """A backend without ``_world`` is a clean skip."""
    adapter = _make_adapter(None)

    class _Worldless:
        _world = None

    adapter._forward_mj_data(_Worldless())  # no exception


def test_forward_mj_data_skips_missing_model_or_data():
    """A world without compiled model/data is a clean skip."""
    adapter = _make_adapter(None)
    sim = _StubSim(_StubWorld(None, None))

    adapter._forward_mj_data(sim)  # no exception
