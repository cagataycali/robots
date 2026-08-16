# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A model action vector shorter than the robot's actuator list must not fabricate commands.

Both LeRobot providers map a policy's flat action vector onto actuator names BY
INDEX. When the vector is shorter than the actuator list -- a 6-DOF checkpoint
pointed at a 7-actuator robot, an embodiment declaring a gripper the checkpoint
never learned -- the unmatched actuators have no value from the model.

They used to be filled with ``0.0`` and sent anyway. That is not a no-op: every
action space involved here is ABSOLUTE POSITION (a LeRobot ``<motor>.pos``
follower, a MuJoCo position actuator), so ``0.0`` means "travel to zero" and the
unmatched joints move -- a real follower drives them there at servo speed, and a
gripper column padded to ``0.0`` closes. The library's own diagnostic described
the opposite ("zero-filled and will not move"), so the documented intent was
already "these actuators hold".

These tests pin the corrected contract: an actuator the model said nothing about
is omitted from the action dict, so it holds position. ``pad_short_actions=True``
restores the padding for a consumer that needs a fixed-width dict, and the
diagnostic then reports the consequence that is actually in effect.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest
import torch  # real or conftest mock - both work

from strands_robots.policies import align_action_values, create_policy
from strands_robots.policies.lerobot_async import LerobotAsyncPolicy
from strands_robots.policies.lerobot_local.embodiment import EmbodimentMap, diagnose_action_dim
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

SIX_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

# A two-link arm with damped, range-limited joints driven by position actuators.
# Written inline so the test needs no asset download: the point is only that the
# action space is absolute joint position, which is what makes a padded 0.0 a
# command rather than an omission.
ARM_MJCF = """
<mujoco model="two_link">
  <compiler angle="radian"/>
  <default>
    <joint type="hinge" axis="0 0 1" damping="4" range="-1.5 1.5" limited="true"/>
    <geom type="capsule" size="0.02"/>
  </default>
  <worldbody>
    <body name="base" pos="0 0 0.1">
      <joint name="shoulder"/>
      <geom fromto="0 0 0 0.2 0 0"/>
      <body name="link2" pos="0.2 0 0">
        <joint name="elbow"/>
        <geom fromto="0 0 0 0.2 0 0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a_shoulder" joint="shoulder" kp="30" ctrlrange="-1.5 1.5"/>
    <position name="a_elbow" joint="elbow" kp="30" ctrlrange="-1.5 1.5"/>
  </actuator>
</mujoco>
"""


# -- The shared rule ----------------------------------------------------------


class TestAlignActionValues:
    """One rule both providers apply, so their contracts cannot drift apart."""

    def test_short_vector_returns_only_the_keys_the_model_produced(self):
        values, keys = align_action_values([1.0, 2.0], ["a", "b", "c"])
        assert values == [1.0, 2.0]
        assert keys == ["a", "b"]  # "c" is omitted, not zeroed

    def test_short_vector_pads_with_zero_only_when_asked(self):
        values, keys = align_action_values([1.0, 2.0], ["a", "b", "c"], pad_short=True)
        assert values == [1.0, 2.0, 0.0]
        assert keys == ["a", "b", "c"]

    def test_exact_length_maps_every_key(self):
        values, keys = align_action_values([1.0, 2.0, 3.0], ["a", "b", "c"])
        assert values == [1.0, 2.0, 3.0]
        assert keys == ["a", "b", "c"]

    def test_long_vector_drops_the_values_with_no_actuator(self):
        # Padding is irrelevant here: there is no actuator to receive index 3.
        for pad in (False, True):
            values, keys = align_action_values([1.0, 2.0, 3.0, 4.0], ["a", "b", "c"], pad_short=pad)
            assert values == [1.0, 2.0, 3.0]
            assert keys == ["a", "b", "c"]

    def test_empty_vector_commands_nothing(self):
        assert align_action_values([], ["a", "b"]) == ([], [])

    def test_numpy_values_are_coerced_to_float(self):
        values, _ = align_action_values(np.array([1.5, -2.5], dtype=np.float32), ["a", "b"])
        assert values == [1.5, -2.5]
        assert all(type(v) is float for v in values)

    def test_returned_pair_is_always_zippable(self):
        # The caller zips these strict=True after unit conversion; equal length
        # is the invariant that makes that safe for every input shape.
        for n_values in range(0, 6):
            values, keys = align_action_values([0.0] * n_values, ["a", "b", "c"])
            assert len(values) == len(keys)


# -- lerobot_local ------------------------------------------------------------


class TestLocalPolicyShortAction:
    def test_unmatched_actuators_are_absent_from_the_action_dict(self):
        policy = LerobotLocalPolicy()
        policy.set_robot_state_keys(SIX_KEYS)
        action = policy._tensor_to_action_dicts(torch.tensor([12.0, -30.0, 45.0, 5.0]))[0]
        assert list(action) == SIX_KEYS[:4]
        # The gripper is the dangerous one: padded to 0.0 it closes on whatever
        # the arm is holding. Absent, it keeps its current opening.
        assert "gripper.pos" not in action
        assert "wrist_roll.pos" not in action

    def test_pad_short_actions_restores_the_zero_fill(self):
        policy = LerobotLocalPolicy(pad_short_actions=True)
        policy.set_robot_state_keys(SIX_KEYS)
        action = policy._tensor_to_action_dicts(torch.tensor([12.0, -30.0, 45.0, 5.0]))[0]
        assert list(action) == SIX_KEYS
        assert action["wrist_roll.pos"] == 0.0
        assert action["gripper.pos"] == 0.0

    def test_default_is_not_to_pad(self):
        assert LerobotLocalPolicy().pad_short_actions is False

    def test_hardware_pos_key_override_omits_unmatched_actuators_too(self):
        # The '.pos' path is where a fabricated 0.0 reaches a physical bus as an
        # absolute Goal_Position, so it must not pad either.
        policy = LerobotLocalPolicy()
        policy.set_robot_state_keys(SIX_KEYS)
        action = policy._tensor_to_action_dicts(
            torch.tensor([12.0, -30.0]),
            hw_action_keys=["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos"],
        )[0]
        assert action == {"shoulder_pan.pos": 12.0, "shoulder_lift.pos": -30.0}

    def test_unit_conversion_applies_to_the_emitted_columns_only(self):
        # A degrees embodiment converts model degrees -> sim radians positionally.
        # The conversion must run on the aligned prefix so column N keeps its own
        # scaling instead of being shifted by a fabricated tail.
        policy = LerobotLocalPolicy()
        policy.set_robot_state_keys(["j1", "j2", "j3"])
        policy._embodiment = EmbodimentMap(
            name="degrees_short",
            state_keys=["j1", "j2", "j3"],
            action_keys=["j1", "j2", "j3"],
            action_units="degrees",
        )
        action = policy._tensor_to_action_dicts(torch.tensor([90.0, 180.0]))[0]
        assert list(action) == ["j1", "j2"]
        assert action["j1"] == pytest.approx(np.pi / 2, rel=1e-6)
        assert action["j2"] == pytest.approx(np.pi, rel=1e-6)

    def test_exact_and_long_vectors_are_unchanged(self):
        policy = LerobotLocalPolicy()
        policy.set_robot_state_keys(["a", "b", "c"])
        assert policy._tensor_to_action_dicts(torch.tensor([1.0, 2.0, 3.0]))[0] == {
            "a": 1.0,
            "b": 2.0,
            "c": 3.0,
        }
        assert policy._tensor_to_action_dicts(torch.tensor([1.0, 2.0, 3.0, 4.0]))[0] == {
            "a": 1.0,
            "b": 2.0,
            "c": 3.0,
        }


# -- lerobot_async ------------------------------------------------------------


class _TimedAction:
    """Minimal stand-in for lerobot's ``TimedAction`` (only ``.action`` is read)."""

    def __init__(self, values: list[float]) -> None:
        self.action = np.asarray(values, dtype=np.float32)


def _async_policy(**kwargs: Any) -> LerobotAsyncPolicy:
    policy = create_policy(
        "lerobot_async",
        server_address="h:1",
        policy_type="act",
        pretrained_name_or_path="x/y",
        **kwargs,
    )
    assert isinstance(policy, LerobotAsyncPolicy)
    policy.set_robot_state_keys(SIX_KEYS)
    return policy


class TestAsyncPolicyShortChunk:
    def test_short_server_chunk_omits_unmatched_actuators(self):
        policy = _async_policy()
        action = policy._chunk_to_action_dicts([_TimedAction([1.0, 2.0, 3.0])])[0]
        assert list(action) == SIX_KEYS[:3]
        assert "gripper.pos" not in action

    def test_pad_short_actions_restores_the_zero_fill(self):
        policy = _async_policy(pad_short_actions=True)
        action = policy._chunk_to_action_dicts([_TimedAction([1.0, 2.0, 3.0])])[0]
        assert list(action) == SIX_KEYS
        assert action["gripper.pos"] == 0.0

    def test_default_is_not_to_pad(self):
        assert _async_policy().pad_short_actions is False

    def test_pad_short_actions_is_a_declared_kwarg_not_an_ignored_one(self, caplog):
        # The client warns about kwargs it drops; this one must be honored, so it
        # must not show up in that warning.
        with caplog.at_level(logging.WARNING):
            policy = _async_policy(pad_short_actions=True)
        assert policy.pad_short_actions is True
        assert "pad_short_actions" not in caplog.text


def test_both_providers_apply_the_same_rule():
    """The shared helper exists so these two cannot drift apart again."""
    values = [1.0, 2.0, 3.0, 4.0]
    for pad in (False, True):
        local = LerobotLocalPolicy(pad_short_actions=pad)
        local.set_robot_state_keys(SIX_KEYS)
        remote = _async_policy(pad_short_actions=pad)
        assert (
            local._tensor_to_action_dicts(torch.tensor(values))[0]
            == remote._chunk_to_action_dicts([_TimedAction(values)])[0]
        )


# -- The diagnostic must describe what actually happens -----------------------


class TestDimMismatchDiagnostic:
    def test_default_reports_that_the_actuators_hold(self):
        msg = diagnose_action_dim(4, 6, name="so101")
        assert msg is not None
        assert "receive no command" in msg
        assert "hold their current position" in msg
        # The old message claimed a zero-fill that no longer happens.
        assert "zero-filled" not in msg
        assert "so101" in msg
        assert "2" in msg  # names the count of unmatched actuators

    def test_pad_short_reports_the_zero_command_and_its_consequence(self):
        msg = diagnose_action_dim(4, 6, pad_short=True)
        assert msg is not None
        assert "commanded to 0.0" in msg
        assert "pad_short_actions=True" in msg
        assert "travels" in msg

    def test_matching_dims_stay_silent_either_way(self):
        assert diagnose_action_dim(6, 6) is None
        assert diagnose_action_dim(6, 6, pad_short=True) is None

    def test_more_values_than_keys_still_flags_the_drop(self):
        for pad in (False, True):
            msg = diagnose_action_dim(8, 6, pad_short=pad)
            assert msg is not None
            assert "dropped" in msg

    def test_warning_logged_by_the_policy_matches_its_own_behaviour(self, caplog):
        policy = LerobotLocalPolicy()
        policy.set_robot_state_keys(SIX_KEYS)
        with caplog.at_level(logging.WARNING):
            action = policy._tensor_to_action_dicts(torch.zeros(4))
        assert "gripper.pos" not in action[0]
        assert any("action dim 4" in r.message and "hold their current position" in r.message for r in caplog.records)


# -- End to end in sim: absolute-position actuators ---------------------------


@pytest.mark.parametrize("pad_short", [False, True])
def test_omitted_actuator_holds_while_a_padded_one_travels_to_zero(tmp_path, pad_short):
    """The whole point, measured on a real position-controlled joint.

    Both actuators are parked at 0.8 rad. The policy then emits a ONE-value
    action, leaving ``a_elbow`` unmatched. Omitted, the elbow holds 0.8. Padded,
    it is commanded to 0.0 and travels there -- which is what a real follower
    does to an unmatched joint, at servo speed.
    """
    from strands_robots import Simulation

    mjcf = tmp_path / "two_link.xml"
    mjcf.write_text(ARM_MJCF)

    sim = Simulation(backend="mujoco", tool_name="short_action_sim", mesh=False)
    try:
        sim.create_world()
        assert sim.add_robot(name="arm", urdf_path=str(mjcf))["status"] == "success"
        action_keys = sim.robot_action_keys(robot_name="arm")
        assert action_keys == ["a_shoulder", "a_elbow"]

        def elbow() -> float:
            return float(sim.get_observation(robot_name="arm")["elbow"])

        # Park both joints at 0.8 rad.
        sim.send_action({key: 0.8 for key in action_keys}, robot_name="arm", n_substeps=500)
        assert elbow() == pytest.approx(0.8, abs=0.01)

        policy = LerobotLocalPolicy(pad_short_actions=pad_short)
        policy.set_robot_state_keys(action_keys)
        action = policy._tensor_to_action_dicts(torch.tensor([0.2]))[0]
        sim.send_action(action, robot_name="arm", n_substeps=500)

        if pad_short:
            assert "a_elbow" in action
            assert elbow() == pytest.approx(0.0, abs=0.01)  # travelled to zero
        else:
            assert "a_elbow" not in action
            assert elbow() == pytest.approx(0.8, abs=0.01)  # held position
    finally:
        sim.cleanup()
