"""Every joint position in a Newton observation carries its ``<joint>.vel``.

``get_observation`` is the observation a policy runs on - ``PolicyRunner`` feeds
it to ``get_actions`` every control step, and a velocity-feedback locomotion
policy reads the velocity out of it BY NAME: the shipped Microduck observation
builder indexes ``obs[f"{joint}.vel"]`` for all 14 joints, and the WBC /
ProtoMotions readers use the same spelling. The MuJoCo backend emits that
companion beside each position; this backend emitted positions only, so a policy
that walks on MuJoCo raised ``KeyError: 'left_hip_yaw.vel'`` on its first tick
here and ``run_policy`` returned ``steps_used=0``. The velocity was already read
(``get_robot_state`` reports it from ``joint_qd``); only the observation was
missing it.

Solver-free: the engine is built via ``__new__`` with only the attributes the
state paths read, so no Newton/Warp stack is required and these pins run on
every CI job rather than being skipped.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from strands_robots.policies.microduck.observation import build_observation
from strands_robots.simulation.models import SimRobot, SimWorld
from strands_robots.simulation.newton.simulation import NewtonSimEngine

_BASE_JOINT = "floating_base_joint"
_HINGES = ["hip_yaw", "hip_pitch", "knee", "ankle"]


class _Arr:
    """The one method the engine calls on a Warp array."""

    def __init__(self, values: list[float]) -> None:
        self._values = np.asarray(values, dtype=np.float64)

    def numpy(self) -> np.ndarray:
        return self._values


def _engine(*, free_base: bool) -> NewtonSimEngine:
    """A Newton engine bound to one robot, with or without a free root.

    A free root shifts the hinge DOF indices away from their coordinate indices
    (7 coordinates, 6 DOFs), which is exactly the case a position index reused
    as a velocity index gets wrong. ``joint_q[i] = 100 + i`` and
    ``joint_qd[i] = 200 + i`` so every reported value names the index it was
    read from.
    """
    joints = ([_BASE_JOINT] if free_base else []) + _HINGES
    world = SimWorld()
    world.robots["duck"] = SimRobot(name="duck", urdf_path="d.xml", data_config="duck", joint_names=list(joints))

    engine = NewtonSimEngine.__new__(NewtonSimEngine)
    engine._world = world
    engine._model = object()  # non-None sentinel: "world created"
    engine._lock = threading.RLock()
    engine._obs_noise = None
    engine._obs_noise_rng = None
    engine._robot_free_base_joint = {"duck": _BASE_JOINT} if free_base else {}
    q_offset, d_offset = (7, 6) if free_base else (0, 0)
    engine._joint_coord_index = {("duck", j): q_offset + i for i, j in enumerate(_HINGES)}
    engine._joint_dof_index = {("duck", j): d_offset + i for i, j in enumerate(_HINGES)}
    if free_base:
        engine._joint_coord_index[("duck", _BASE_JOINT)] = 0
        engine._joint_dof_index[("duck", _BASE_JOINT)] = 0
    engine._state_0 = type(
        "_State",
        (),
        {"joint_q": _Arr([100.0 + i for i in range(16)]), "joint_qd": _Arr([200.0 + i for i in range(16)])},
    )()
    return engine


class TestEachPositionCarriesItsVelocity:
    """The observation's joint block is (position, ``.vel``) pairs."""

    @pytest.mark.parametrize("free_base", [False, True])
    def test_every_position_has_a_velocity_companion(self, free_base):
        obs = _engine(free_base=free_base).get_observation("duck", skip_images=True)
        positions = [k for k, v in obs.items() if isinstance(v, float) and not k.endswith(".vel")]
        assert positions == _HINGES
        assert [f"{j}.vel" for j in _HINGES] == [k for k in obs if k.endswith(".vel")]

    @pytest.mark.parametrize(
        ("free_base", "q_offset", "d_offset"),
        [(False, 0, 0), (True, 7, 6)],
    )
    def test_the_velocity_is_read_at_the_dof_index(self, free_base, q_offset, d_offset):
        """Velocities come from ``joint_qd`` at the DOF index, not the coordinate one.

        With a free root the two differ by one, so reusing the position index
        would report the neighbouring joint's velocity.
        """
        obs = _engine(free_base=free_base).get_observation("duck", skip_images=True)
        for i, jname in enumerate(_HINGES):
            assert obs[jname] == pytest.approx(100.0 + q_offset + i)
            assert obs[f"{jname}.vel"] == pytest.approx(200.0 + d_offset + i)

    def test_the_free_root_has_no_velocity_scalar(self):
        """The 6-DoF root is surfaced as ``base_*``, so it gains no ``.vel`` either."""
        obs = _engine(free_base=True).get_observation("duck", skip_images=True)
        assert _BASE_JOINT not in obs
        assert f"{_BASE_JOINT}.vel" not in obs
        assert obs["base_lin_vel"] == pytest.approx([200.0, 201.0, 202.0])

    def test_it_is_the_velocity_get_robot_state_reports(self):
        """One reading, two surfaces - the pair cannot disagree."""
        engine = _engine(free_base=True)
        obs = engine.get_observation("duck", skip_images=True)
        state = engine.get_robot_state("duck")["content"][1]["json"]["state"]
        assert {j: obs[f"{j}.vel"] for j in _HINGES} == {j: state[j]["velocity"] for j in _HINGES}


class TestALocomotionPolicyCanReadTheObservation:
    """The consumer that failed: the Microduck observation builder.

    It indexes ``obs[f"{joint}.vel"]`` for every joint, unguarded, so a
    position-only observation raised ``KeyError`` before a single action was
    applied.
    """

    def test_the_microduck_builder_consumes_a_newton_observation(self):
        obs = _engine(free_base=True).get_observation("duck", skip_images=True)
        obs["base_quat"] = [1.0, 0.0, 0.0, 0.0]
        vector = build_observation(
            obs,
            joint_names=_HINGES,
            default_pose=np.zeros(len(_HINGES), dtype=np.float32),
            last_action=np.zeros(len(_HINGES), dtype=np.float32),
            command=np.zeros(3, dtype=np.float32),
        )
        # 3 base_ang_vel + 3 gravity + 4 joint_pos + 4 joint_vel + 4 last_action + 3 command
        assert vector.shape == (21,)
        assert vector[10:14] == pytest.approx([206.0, 207.0, 208.0, 209.0])
