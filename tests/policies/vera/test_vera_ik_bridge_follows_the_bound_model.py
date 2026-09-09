"""The IK bridge a rollout solves on is a bridge for the model that is bound.

``VeraPolicy`` turns an end-effector-delta chunk into joint targets through a
:class:`~strands_robots.policies.vera.sim_ik.MinkIKBridge`, which holds a
``mink.Configuration`` and the two tasks built from ONE compiled ``MjModel``.
The simulation rebinds a policy on every rollout
(``bind_policy_sim_context`` -> ``set_sim_context``) and hands it the model
compiled *now* - a new object after any scene change - so which model a cached
bridge was built from is part of whether that bridge is still usable.

The sibling contracts pin the cache without ever varying that input:
:mod:`tests.policies.vera.test_vera_ik_bridge_lazy_build` pins "builds once; a
later inference reuses it" and the rebuild after an explicit
``set_ik_target``, both against a single model; the adjacent qpos-address cache
in :mod:`tests.policies.vera.test_vera_ik_qpos_addressing` is keyed on the model
*and* the state keys and says so. A bridge served for a superseded model is
silent at the boundary in both directions - it refuses the seed built from the
bound model by naming the previous model's ``nq``, or it solves in the previous
world's geometry and returns an action dict shaped exactly like a good one.

The bridge is stood in for so these run without the optional ``mink`` stack, but
the models are really compiled and the rebind is really the engine's.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest

pytest.importorskip("mujoco")

import mujoco  # noqa: E402

from strands_robots import Simulation  # noqa: E402
from strands_robots.policies.vera import sim_ik as sim_ik_mod  # noqa: E402
from strands_robots.policies.vera.provider import VeraPolicy  # noqa: E402
from strands_robots.simulation.ik import pose_vector_error  # noqa: E402

#: One step of eef-delta: a 2 cm descent with the gripper column open.
_CHUNK = np.zeros((1, 8), np.float32)
_CHUNK[:, 2] = -0.02
_CHUNK[:, 7] = 1.0

#: A second compiled model, distinct from the scene's, carrying the two frames
#: the keying table names. Only its identity matters to the table.
_OTHER_XML = """
<mujoco><worldbody>
  <body name="panda/link7"><geom type="sphere" size="0.05"/>
    <body name="panda/hand" pos="0.1 0 0"><geom type="sphere" size="0.05"/></body>
  </body>
</worldbody></mujoco>
"""


class _Client:
    """A VERA client that serves one eef-delta chunk, so no server is needed."""

    def get_server_metadata(self) -> dict[str, Any]:
        return {
            "action_space": "eef_delta",
            "context_frames": 1,
            "gripper_dim_index": 7,
            "gripper_is_raw": True,
            "view_keys": ["image"],
        }

    def configure(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {}

    def reset(self, *args: Any, **kwargs: Any) -> None:
        return None

    def close(self, *args: Any, **kwargs: Any) -> None:
        return None

    def infer(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"action": _CHUNK}


class _FkBridge:
    """A bridge stand-in that is honestly bound to ONE model.

    Forward kinematics is real MuJoCo forward kinematics on the model it was
    constructed with, and both configuration-reading methods hold their input to
    that model's ``nq`` through the same :func:`pose_vector_error` the shipped
    :class:`~strands_robots.simulation.ik.MinkIKBridge` uses. Both are what make
    a bridge built from a superseded model observable at all.
    """

    #: Every bridge constructed during a test, in order.
    built: list[_FkBridge] = []

    def __init__(self, model: Any, ee_frame_name: str, ee_frame_type: str = "body", **_: Any) -> None:
        self.model = model
        self.ee_frame_name = ee_frame_name
        self.ee_frame_type = ee_frame_type
        self._data = mujoco.MjData(model)
        _FkBridge.built.append(self)

    def ee_pose(self, qpos: Any) -> np.ndarray:
        q = np.asarray(qpos, np.float64)
        if text := pose_vector_error("ee_pose", "qpos", q, self.model.nq):
            raise ValueError(text)
        self._data.qpos[:] = q
        mujoco.mj_forward(self.model, self._data)
        body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.ee_frame_name)
        pose = np.eye(4)
        pose[:3, 3] = self._data.xpos[body]
        return pose

    def solve(self, target_pose: Any, q_init: Any) -> np.ndarray:
        q = np.asarray(q_init, np.float64)
        if text := pose_vector_error("solve", "q_init", q, self.model.nq):
            raise ValueError(text)
        return q


@pytest.fixture
def bridges(monkeypatch: pytest.MonkeyPatch) -> list[_FkBridge]:
    """Route every build through :class:`_FkBridge` and record them."""
    _FkBridge.built = []
    monkeypatch.setattr(sim_ik_mod, "MinkIKBridge", _FkBridge)
    return _FkBridge.built


def _policy() -> VeraPolicy:
    client: Any = _Client()
    policy = VeraPolicy(client=client, auto_launch_server=False)
    policy._runner = None
    return policy


def _rollout(policy: VeraPolicy, sim: Simulation) -> list[dict[str, Any]]:
    """Bind the policy the way the engine does, then infer once."""
    joints = sim.robot_joint_names("panda")
    policy.set_robot_state_keys(joints)
    sim.bind_policy_sim_context(policy, "panda")
    policy.reset()
    observation: dict[str, Any] = {"image": np.zeros((8, 8, 3), np.uint8), **dict.fromkeys(joints, 0.0)}
    return asyncio.run(policy.get_actions(observation, "descend"))


def _bound_model(sim: Simulation) -> Any:
    """The compiled model the engine hands a policy for this scene."""
    world = sim._world
    assert world is not None, "the fixture must have created a world"
    return world._model


def _panda_world(**add_robot: Any) -> Simulation:
    sim = Simulation(backend="mujoco", mesh=False)
    sim.create_world()
    sim.add_robot(name="panda", **add_robot)
    return sim


class TestEveryInputTheBridgeIsBuiltFromIsKeyed:
    """A differing-value table over the three inputs the build reads."""

    @pytest.mark.parametrize(
        ("frame", "frame_type", "new_model", "rebuilt"),
        [
            pytest.param("panda/hand", "body", False, False, id="nothing-differs-is-served-from-the-cache"),
            pytest.param("panda/hand", "body", True, True, id="the-model-differs"),
            pytest.param("panda/link7", "body", False, True, id="the-frame-name-differs"),
            pytest.param("panda/hand", "site", False, True, id="the-frame-type-differs"),
        ],
    )
    def test_a_differing_input_re_derives_the_bridge(
        self, bridges: list[_FkBridge], frame: str, frame_type: str, new_model: bool, rebuilt: bool
    ) -> None:
        sim = _panda_world()
        try:
            first_model = _bound_model(sim)
            second_model = mujoco.MjModel.from_xml_string(_OTHER_XML)
            policy = _policy()
            policy.set_robot_state_keys(sim.robot_joint_names("panda"))
            policy.set_ik_target(first_model, "panda/hand", "body")

            first = policy._ensure_ik_bridge(first_model, "panda/hand")
            policy._ee_frame_type = frame_type
            second = policy._ensure_ik_bridge(second_model if new_model else first_model, frame)

            assert (second is not first) is rebuilt
            assert len(bridges) == (2 if rebuilt else 1)
        finally:
            sim.cleanup()


class TestARolloutAfterASceneChangeSolvesInTheBoundWorld:
    """The rebind the engine performs, driven end to end through ``get_actions``."""

    def test_a_seed_from_the_bound_model_is_not_refused_against_the_previous_dof_count(
        self, bridges: list[_FkBridge]
    ) -> None:
        """Adding an object recompiles the model, and the seed grows with it.

        The seed is ``model.nq`` long for whichever model is bound, so a bridge
        built from the previous one refuses it - naming ``q_init`` and the
        superseded model's width, which reads as a caller bug for a value the
        caller never chose.
        """
        sim = _panda_world()
        try:
            policy = _policy()
            assert _rollout(policy, sim), "the first rollout must emit actions"
            before = _bound_model(sim).nq

            sim.add_object(name="cube", position=[0.4, 0.0, 0.1], size=[0.03, 0.03, 0.03])
            assert _bound_model(sim).nq > before, "the free-floating object must add DOFs"

            actions = _rollout(policy, sim)

            assert actions, "the rollout after the scene change must still emit actions"
            assert bridges[-1].model is _bound_model(sim)
            assert len(bridges) == 2, "the recompiled model must be built against"
        finally:
            sim.cleanup()

    def test_the_targets_are_solved_in_the_world_the_robot_is_in(self, bridges: list[_FkBridge]) -> None:
        """Same DOF count, different geometry: the silent half.

        Rebuilding the scene with the arm somewhere else leaves ``nq`` untouched,
        so nothing refuses anything; the previous world's bridge simply reads and
        writes end-effector poses half a metre from where the arm now is.
        """
        offset = 0.5
        policy = _policy()
        for position in ([0.0, 0.0, 0.0], [offset, 0.0, 0.0]):
            sim = _panda_world(position=position)
            try:
                assert _rollout(policy, sim), "each rollout must emit actions"
                bound = _bound_model(sim)
            finally:
                sim.cleanup()

        assert len(bridges) == 2, "the second world must get its own bridge"
        assert bridges[-1].model is bound
        rest = np.zeros(bound.nq)
        moved = float(np.linalg.norm(bridges[-1].ee_pose(rest)[:3, 3] - bridges[0].ee_pose(rest)[:3, 3]))
        assert moved == pytest.approx(offset, abs=1e-6), (
            "the two worlds must disagree by the offset, or this test pins nothing"
        )

    def test_an_unchanged_scene_is_still_served_the_built_bridge(self, bridges: list[_FkBridge]) -> None:
        """The control: a second rollout in the same world must not rebuild.

        The bridge resolves a QP backend and constructs a configuration plus two
        tasks, so reuse is what the cache is for.
        """
        sim = _panda_world()
        try:
            policy = _policy()
            assert _rollout(policy, sim), "the first rollout must emit actions"
            assert _rollout(policy, sim), "the second rollout must emit actions"

            assert len(bridges) == 1, "an unchanged model and frame must be served from the cache"
        finally:
            sim.cleanup()
