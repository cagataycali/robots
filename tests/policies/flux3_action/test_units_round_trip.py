"""``UnitAdapter`` carries MuJoCo ``so101`` radians into the FLUX 3 Action frame and back exactly.

The checkpoint ``black-forest-labs/flux-3-action-so101`` speaks lerobot SO-101
units: five arm joints in degrees, gripper in percent, in the frame of the
community episodes it was tuned on (rest ~ (0, 190, 180, 70, 0) deg, gripper
~0 %). The MuJoCo model reports radians around a different zero (upper arm
vertical, forearm forward) and its ``shoulder_lift`` axis points the other way.
A unit mistake here is silent at runtime - the arm just wanders to a joint
limit - so the conversion is pinned down without a GPU.
"""

from __future__ import annotations

import math

import pytest

from strands_robots.policies.flux3_action.units import (
    SO101_JOINT_LABELS,
    SO101_SIM_GRIPPER_RANGE_RAD,
    SO101_SIM_JOINT_OFFSETS_DEG,
    SO101_SIM_JOINT_SIGNS,
    SO101_SIM_REST_QPOS_RAD,
    UnitAdapter,
)


def test_the_sim_rest_posture_lands_inside_the_checkpoint_window() -> None:
    # q01..q99 of ``state`` in the released checkpoint's dataset_statistics.json.
    q01 = (-92.0, 84.8, 61.4, -24.0, -124.5, 0.5)
    q99 = (54.1, 184.2, 178.9, 80.6, 60.2, 53.6)
    state = UnitAdapter().robot_to_model(SO101_SIM_REST_QPOS_RAD)
    assert [round(v) for v in state] == [0, 184, 175, 70, 0, 5]
    for label, value, lo, hi in zip(SO101_JOINT_LABELS, state, q01, q99, strict=True):
        assert lo <= value <= hi, f"{label}={value} outside [{lo}, {hi}]"


def test_the_model_zero_maps_to_a_vertical_upper_arm_and_a_right_angle_elbow() -> None:
    state = UnitAdapter().robot_to_model([0.0] * 5 + [SO101_SIM_GRIPPER_RANGE_RAD[0]])
    assert state == [0.0, 90.0, 90.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    "joints",
    [
        list(SO101_SIM_REST_QPOS_RAD),
        [0.3, -1.2, 0.9, -0.4, 2.0, 1.0],
        [-1.9, 1.7, -1.7, 1.6, -2.7, -0.17],
    ],
)
def test_round_trip_is_exact_to_float_precision(joints: list[float]) -> None:
    adapter = UnitAdapter()
    back = adapter.model_to_robot(adapter.robot_to_model(joints))
    assert max(abs(a - b) for a, b in zip(joints, back, strict=True)) < 1e-12


def test_gripper_percent_spans_the_mujoco_joint_travel() -> None:
    adapter = UnitAdapter()
    closed, opened = SO101_SIM_GRIPPER_RANGE_RAD
    assert adapter.robot_to_model([0.0] * 5 + [closed])[5] == 0.0
    assert adapter.robot_to_model([0.0] * 5 + [opened])[5] == pytest.approx(100.0)
    assert adapter.model_to_robot([0.0] * 5 + [50.0])[5] == pytest.approx((closed + opened) / 2)


def test_lift_sign_is_negative_and_both_folding_joints_carry_the_90_degree_offset() -> None:
    assert SO101_SIM_JOINT_SIGNS == (1.0, -1.0, 1.0, 1.0, 1.0)
    assert SO101_SIM_JOINT_OFFSETS_DEG == (0.0, 90.0, 90.0, 0.0, 0.0)
    # Folded back by 100 deg in the model is the community rest value 190.
    assert UnitAdapter().robot_to_model([0.0, math.radians(-100.0), 0.0, 0.0, 0.0, 0.0])[1] == pytest.approx(190.0)


def test_hardware_identity_frame_passes_degrees_and_percent_through() -> None:
    adapter = UnitAdapter(
        joint_units="deg", joint_signs=(1.0,) * 5, joint_offsets_deg=(0.0,) * 5, gripper_range=(0.0, 100.0)
    )
    state = [12.5, 150.0, 120.0, 30.0, -10.0, 42.0]
    assert adapter.robot_to_model(state) == state
    assert adapter.model_to_robot(state) == state


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"joint_units": "turns"}, "joint_units"),
        ({"joint_offsets_deg": (0.0, 0.0)}, "5 values"),
        ({"joint_signs": (1.0, 2.0, 1.0, 1.0, 1.0)}, "\\+1 or -1"),
        ({"gripper_range": (1.0, 1.0)}, "distinct"),
    ],
)
def test_malformed_calibration_is_refused_by_name(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        UnitAdapter(**kwargs)


def test_wrong_joint_count_is_refused_naming_the_six_joints() -> None:
    with pytest.raises(ValueError, match="shoulder_pan"):
        UnitAdapter().robot_to_model([0.0] * 5)
    with pytest.raises(ValueError, match="6 action values"):
        UnitAdapter().model_to_robot([0.0] * 7)
