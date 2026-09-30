"""An embodiment can speak DROID: joint-velocity actions and a gripper fraction.

π0.5-DROID (``lerobot/pi05_droid``) is trained on state = 7 arm joints + the
gripper as the fraction closed (0 open .. 1 closed), and emits 7 arm joint
VELOCITIES + that gripper fraction. strands had no velocity action mode (every
action is a position target) and no gripper conversion outside the SO arms'
``RANGE_0_100``, so running it took an out-of-tree adapter. ``EmbodimentMap``
now declares ``action_mode="velocity"`` + ``action_dt`` (integrated from the
measured joints), ``gripper_fraction`` (``[open, closed]``) and
``gripper_followers`` (a two-finger hand from one gripper dimension), and the
shipped ``panda_droid`` embodiment uses them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch  # real or conftest mock - both work

from strands_robots.policies.lerobot_local.embodiment import (
    EmbodimentMap,
    fraction_to_gripper,
    gripper_to_fraction,
    load_embodiment,
)
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

ARM = [f"joint{i}" for i in range(1, 8)]


def _droid(**kw: Any) -> EmbodimentMap:
    base: dict[str, Any] = dict(
        name="droid_probe",
        state_keys=[*ARM, "finger_joint1"],
        action_keys=[*ARM, "finger_joint1"],
        gripper_index=7,
        gripper_fraction=[0.04, 0.0],
        gripper_followers=["finger_joint2"],
        action_mode="velocity",
        action_dt=0.1,
    )
    base.update(kw)
    return EmbodimentMap(**base)


class TestTheGripperFraction:
    def test_open_is_zero_and_closed_is_one_whichever_way_the_joint_travels(self) -> None:
        assert gripper_to_fraction(0.04, [0.04, 0.0]) == pytest.approx(0.0)
        assert gripper_to_fraction(0.0, [0.04, 0.0]) == pytest.approx(1.0)
        assert gripper_to_fraction(0.01, [0.04, 0.0]) == pytest.approx(0.75)
        assert fraction_to_gripper(0.75, [0.04, 0.0]) == pytest.approx(0.01)

    def test_a_command_outside_zero_to_one_is_clipped_to_the_joint(self) -> None:
        assert fraction_to_gripper(1.7, [0.04, 0.0]) == pytest.approx(0.0)
        assert fraction_to_gripper(-0.3, [0.04, 0.0]) == pytest.approx(0.04)

    def test_the_map_converts_its_gripper_column_both_ways(self) -> None:
        emb = _droid()
        assert emb.sim_state_to_model([0.1] * 7 + [0.01])[7] == pytest.approx(0.75)
        assert emb.model_action_to_sim([0.0] * 7 + [0.25])[7] == pytest.approx(0.03)


class TestVelocityActions:
    def test_a_chunk_integrates_from_the_measured_joints(self) -> None:
        emb = _droid()
        q: list[float | None] = [*(0.1 * i for i in range(7)), 0.04]
        chunk = [[1.0] * 7 + [0.5], [2.0] * 7 + [1.0]]

        targets = emb.velocities_to_targets(chunk, q)

        # action_dt = 0.1: first step +0.1, the second a further +0.2
        measured = np.array(q[:7], dtype=float)
        np.testing.assert_allclose(targets[0][:7], measured + 0.1)
        np.testing.assert_allclose(targets[1][:7], measured + 0.3)
        # the gripper is a position, not integrated
        assert [targets[0][7], targets[1][7]] == [0.5, 1.0]

    def test_a_joint_with_no_measurement_is_refused_not_integrated_from_zero(self) -> None:
        with pytest.raises(ValueError, match="measured position"):
            unmeasured: list[float | None] = [*([0.0] * 6), None, 0.04]
            _droid().velocities_to_targets([[1.0] * 8], unmeasured)

    def test_position_mode_leaves_the_chunk_alone(self) -> None:
        emb = EmbodimentMap(name="p", state_keys=["a"], action_keys=["a"])
        assert emb.velocities_to_targets([[3.0]], [1.0]) == [[3.0]]


class TestTheDeclarationsAreGraded:
    @pytest.mark.parametrize(
        ("kw", "match"),
        [
            ({"action_mode": "torque"}, "action_mode"),
            ({"action_dt": 0.0}, "positive action_dt"),
            ({"action_dt": float("nan")}, "positive action_dt"),
            ({"action_dt": True}, "positive action_dt"),
            ({"gripper_fraction": [0.04]}, r"\[open, closed\]"),
            ({"gripper_fraction": [0.04, 0.04]}, r"\[open, closed\]"),
            ({"gripper_index": -1, "gripper_followers": []}, "needs gripper_index"),
            ({"gripper_followers": [""]}, "gripper_followers"),
            ({"state_units": "degrees"}, "both claim the gripper column"),
        ],
    )
    def test_a_declaration_no_site_can_honour_is_refused(self, kw, match) -> None:
        with pytest.raises(ValueError, match=match):
            _droid(**kw)


class TestThePolicyEmitsDroidActions:
    def test_velocities_become_targets_and_one_gripper_drives_both_fingers(self) -> None:
        policy = LerobotLocalPolicy(actions_per_step=2)
        policy._embodiment = _droid()
        policy.set_robot_state_keys([*ARM, "finger_joint1"])
        observation = {**{j: 0.5 for j in ARM}, "finger_joint1": 0.04, "finger_joint2": 0.04}
        chunk = torch.tensor([[1.0] * 7 + [1.0] + [9.0] * 24, [1.0] * 7 + [0.0] + [9.0] * 24])  # 32-wide, padded

        actions = policy._tensor_to_action_dicts(chunk, observation=observation)

        assert actions[0]["joint1"] == pytest.approx(0.6)
        assert actions[1]["joint1"] == pytest.approx(0.7)
        # fraction 1.0 = closed = 0.0 m on both fingers; 0.0 = open = 0.04 m
        assert actions[0]["finger_joint1"] == pytest.approx(0.0) and actions[0]["finger_joint2"] == pytest.approx(0.0)
        assert actions[1]["finger_joint1"] == pytest.approx(0.04) and actions[1]["finger_joint2"] == pytest.approx(0.04)
        assert set(actions[0]) == {*ARM, "finger_joint1", "finger_joint2"}  # no padding column leaks out


def test_the_shipped_panda_droid_embodiment_loads() -> None:
    emb = load_embodiment("panda_droid")
    assert emb.action_mode == "velocity"
    assert emb.action_dt == pytest.approx(1 / 15)
    assert emb.gripper_fraction == [0.04, 0.0]
    assert emb.gripper_followers == ["finger_joint2"]
    assert emb.state_keys == [*ARM, "finger_joint1"]
    assert emb.action_dim_policy == "truncate"
    assert load_embodiment("franka_droid").name == "panda_droid"


class TestAPaddedActionWidth:
    class _F:
        def __init__(self, n: int) -> None:
            self.shape = (n,)

    def test_truncate_accepts_a_padded_model_and_strict_does_not(self) -> None:
        out = {"action": self._F(32)}
        _droid(action_dim_policy="truncate").validate({}, out)
        with pytest.raises(ValueError, match="action dim is 32"):
            _droid().validate({}, out)

    def test_an_unknown_action_dim_policy_is_refused(self) -> None:
        with pytest.raises(ValueError, match="action_dim_policy"):
            _droid(action_dim_policy="pad")
