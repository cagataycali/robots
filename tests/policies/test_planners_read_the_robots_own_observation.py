# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The planner providers read the observation a simulated robot actually sends.

A robot publishes its proprioception as per-joint scalars keyed by joint name
(``obs["joint1"] = qpos`` beside ``obs["joint1.vel"] = qvel``) and writes no flat
``observation.state`` vector. The two planner providers read only that flat key,
so every simulated rollout planned from no start state at all:

================================================ ============================ ==========================
``sim.run_policy(policy_provider="curobo")``      pre-fix                      post-fix
================================================ ============================ ==========================
start state handed to cuRobo                      ``None``                     the robot's 7 arm joints
``plan_pose`` on cuRobo main                      ``AttributeError`` on ndim   plans
waypoint keys (9-wide plan, 8 declared keys)      ``joint_0..joint_8``         the robot's joint names
Panda TCP against a ``target_pose`` 231 mm away   0 steps, never moved         6.2 mm, 500 steps
================================================ ============================ ==========================

Two rosters are in play and both are read from the planner itself:
``kinematics.joint_names`` is the space cuRobo plans in (7 for the Panda, whose
``franka.yml`` locks the fingers) and ``kinematics.all_articulated_joint_names``
is the layout of the waypoints it returns (9). A robot declares one action key
per actuator (8, the fingers sharing one), so the declared roster keys neither -
and the keys the robot published its state under do, without inventing a name.

Stubs only - no GPU, no cuRobo, no MuJoCo.
"""

from __future__ import annotations

import asyncio
import types
from typing import Any

import pytest

from strands_robots.policies._state_keys import joint_positions_from_observation, observation_joint_keys
from strands_robots.policies.curobo import CuroboPolicy
from strands_robots.policies.moveit2 import MoveIt2Policy

#: The Panda as MuJoCo reports it: 7 arm joints then the two fingers.
SIM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "finger_joint1", "finger_joint2")
#: What ``robot_action_keys`` declares for it - one key per actuator.
SIM_ACTION_KEYS = [f"actuator{i + 1}" for i in range(8)]
#: cuRobo's two rosters for ``franka.yml``.
PLAN_SPACE = [f"panda_joint{i + 1}" for i in range(7)]
WAYPOINT_LAYOUT = [*PLAN_SPACE, "panda_finger_joint1", "panda_finger_joint2"]

HOME = (0.0, -0.3, 0.1, -1.5708, 0.2, 1.5708, -0.7853, 0.04, 0.04)


def sim_observation(positions: tuple[float, ...] = HOME) -> dict[str, Any]:
    """The unified sim observation: a scalar per joint, each with a ``.vel`` sibling."""
    observation: dict[str, Any] = {}
    for name, value in zip(SIM_JOINTS, positions, strict=True):
        observation[name] = value
        observation[f"{name}.vel"] = 0.0
    observation["default"] = object()  # the free camera frame rides along
    return observation


class _StubResult:
    def __init__(self, trajectory: list[list[float]]) -> None:
        self.success = True
        self.status = "ok"
        self.trajectory = trajectory


class _StubPlanner:
    """A planner declaring cuRobo's two rosters and recording its start state."""

    def __init__(self, *, plan_space: list[str] = PLAN_SPACE, layout: list[str] = WAYPOINT_LAYOUT) -> None:
        self.kinematics = types.SimpleNamespace(
            tool_frames=["panda_hand"],
            joint_names=plan_space,
            all_articulated_joint_names=layout,
        )
        self.starts: list[Any] = []

    def plan_single(self, start: Any, goal: Any) -> _StubResult:
        self.starts.append(start)
        # One waypoint per articulated joint layout slot, as cuRobo writes them.
        return _StubResult([[0.1 * (i + 1) for i in range(len(self.kinematics.all_articulated_joint_names))]] * 3)


def _curobo(planner: _StubPlanner, *, declare: list[str] | None = SIM_ACTION_KEYS) -> CuroboPolicy:
    policy = CuroboPolicy(motion_gen=planner, action_horizon=8, warmup=False)
    if declare is not None:
        policy.set_robot_state_keys(declare)
    return policy


class TestTheObservationIsReadAsAStateVector:
    """One state, two shapes - :mod:`strands_robots.policies._state_keys` reads both."""

    @pytest.mark.parametrize(
        ("observation", "declared", "expected"),
        [
            pytest.param({"observation.state": [1, 2, 3]}, [], [1.0, 2.0, 3.0], id="flat-vector"),
            pytest.param(
                {"observation.state": [1, 2], "joint1": 9.0, "joint2": 9.0},
                ["joint1", "joint2"],
                [1.0, 2.0],
                id="flat-wins-over-scalars",
            ),
            pytest.param(sim_observation(), [], list(HOME), id="sim-scalars-drop-velocity-siblings"),
            pytest.param(
                sim_observation(),
                ["joint7", "joint1"],
                [HOME[6], HOME[0]],
                id="declared-roster-orders-the-scalars",
            ),
            pytest.param(sim_observation(), SIM_ACTION_KEYS, list(HOME), id="declared-roster-absent-is-inferred"),
            pytest.param({}, [], None, id="no-state-at-all"),
            pytest.param({"default": object()}, ["joint1"], None, id="images-only"),
        ],
    )
    def test_positions_read_from_either_shape(
        self, observation: dict[str, Any], declared: list[str], expected: list[float] | None
    ) -> None:
        assert joint_positions_from_observation(observation, declared) == expected

    def test_keys_are_the_ordering_the_vector_was_read_in(self) -> None:
        assert observation_joint_keys(sim_observation()) == list(SIM_JOINTS)


class TestCuroboPlansFromTheSimulatedRobotsState:
    def test_start_state_is_the_plan_space_of_the_observed_joints(self) -> None:
        """The 9 observed joints reach cuRobo as the 7 it plans over, in plan order."""
        planner = _StubPlanner()
        policy = _curobo(planner)
        policy.get_actions_sync(sim_observation(), "", target_pose=[0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0])
        assert planner.starts == [list(HOME[:7])]

    def test_waypoints_are_keyed_by_the_joints_the_robot_published(self) -> None:
        """A 9-wide plan keyed by a robot declaring 8 action keys - the fabricated
        ``joint_<i>`` labels resolved to no actuator, so nothing was commanded."""
        policy = _curobo(_StubPlanner())
        actions = policy.get_actions_sync(sim_observation(), "", target_pose=[0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0])
        assert list(actions[0]) == list(SIM_JOINTS)

    def test_declared_roster_still_wins_when_it_fits_the_plan(self) -> None:
        """A caller whose declared keys match the plan width keys the row with them."""
        layout = list(PLAN_SPACE)
        planner = _StubPlanner(layout=layout)
        policy = _curobo(planner, declare=[f"j{i}" for i in range(7)])
        actions = policy.get_actions_sync(
            {"observation.state": list(HOME[:7])}, "", target_pose=[0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]
        )
        assert list(actions[0]) == [f"j{i}" for i in range(7)]

    def test_observation_with_no_state_is_refused_naming_both_shapes(self) -> None:
        """cuRobo plans FROM a configuration; ``None`` died inside the vendor library."""
        policy = _curobo(_StubPlanner())
        with pytest.raises(ValueError, match="observation.state"):
            asyncio.run(policy.get_actions({"default": object()}, "", target_pose=[0.5, 0.0, 0.4, 1, 0, 0, 0]))

    def test_a_state_matching_neither_roster_is_refused(self) -> None:
        policy = _curobo(_StubPlanner())
        with pytest.raises(ValueError, match="plans over 7 joints"):
            asyncio.run(
                policy.get_actions(
                    {"observation.state": [0.0] * 5}, "", target_pose=[0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]
                )
            )


class TestMoveIt2ReadsTheSameObservation:
    def test_sim_scalars_are_the_sidecars_start_state(self) -> None:
        """Two providers on one ``Policy`` contract read one observation as one vector."""
        policy = MoveIt2Policy(planning_group="panda_arm", connect_eagerly=False)
        policy.set_robot_state_keys(SIM_ACTION_KEYS)
        assert policy._extract_joint_state(sim_observation()) == list(HOME)

    def test_no_state_leaves_the_sidecar_its_own_estimate(self) -> None:
        policy = MoveIt2Policy(planning_group="panda_arm", connect_eagerly=False)
        assert policy._extract_joint_state({"default": object()}) is None
