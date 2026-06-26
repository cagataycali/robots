"""Regression tests for the SO-arm degrees<->radians units conversion.

MolmoAct2 SO-100/101 (and other lerobot SO-arm checkpoints) emit joint actions
in the driver's MotorNormMode: arm joints in DEGREES, gripper in RANGE_0_100
(see lerobot/robots/so_follower/so_follower.py). The MuJoCo sim joints are
RADIANS. Without conversion the raw degree values saturate the radian joint
limits and the arm freezes -- this pins the fix.
"""

import math

import numpy as np

from strands_robots.policies.lerobot_local.embodiment import (
    EmbodimentMap,
    _convert_joint_vector,
    load_embodiment,
)

# so101 sim gripper joint range (robotstudio_so101/so101_new_calib.xml joint 6).
GRIPPER_RANGE = [-0.175, 1.745]


def _so_arm_map() -> EmbodimentMap:
    return EmbodimentMap(
        name="test_so",
        state_keys=["1", "2", "3", "4", "5", "6"],
        action_keys=["1", "2", "3", "4", "5", "6"],
        state_units="degrees",
        action_units="degrees",
        gripper_index=5,
        gripper_joint_range=GRIPPER_RANGE,
    )


def test_action_degrees_to_radians_arm():
    emb = _so_arm_map()
    # 90 deg arm joints -> pi/2 rad; gripper handled separately below.
    out = emb.model_action_to_sim([90.0, 90.0, 90.0, 90.0, 90.0, 50.0])
    for v in out[:5]:
        assert math.isclose(v, math.pi / 2, abs_tol=1e-6), out


def test_action_gripper_range_0_100_to_joint():
    emb = _so_arm_map()
    lo, hi = GRIPPER_RANGE
    # 0 -> lo, 100 -> hi, 50 -> midpoint.
    assert math.isclose(emb.model_action_to_sim([0, 0, 0, 0, 0, 0])[5], lo, abs_tol=1e-6)
    assert math.isclose(emb.model_action_to_sim([0, 0, 0, 0, 0, 100])[5], hi, abs_tol=1e-6)
    assert math.isclose(emb.model_action_to_sim([0, 0, 0, 0, 0, 50])[5], (lo + hi) / 2, abs_tol=1e-6)


def test_state_radians_to_degrees_round_trip():
    emb = _so_arm_map()
    sim_state = [0.5, -1.0, 1.2, -0.3, 2.0, 0.4]
    to_model = emb.sim_state_to_model(sim_state)
    back = emb.model_action_to_sim(to_model)
    assert np.allclose(back, sim_state, atol=1e-6), (back, sim_state)


def test_native_units_is_noop():
    emb = EmbodimentMap(name="t", action_units="native", state_units="native")
    vals = [125.0, -270.0, 10.0]
    assert emb.model_action_to_sim(vals) == vals
    assert emb.sim_state_to_model(vals) == vals


def test_degree_action_stays_in_so101_joint_range():
    """A realistic MolmoAct2 degree action (mean joint2 ~125 deg) must land
    inside the so101 radian joint limits after conversion -- the whole point of
    the fix (raw degrees saturate; converted radians fit)."""
    emb = _so_arm_map()
    # so101 joint ranges (rad): 1:+/-1.92 2:+/-1.745 3:[-1.745,1.571] 4:+/-1.658 5:+/-2.793
    deg_action = [40.0, 95.0, 80.0, 50.0, -60.0, 50.0]  # within trained quantiles + joint limits
    rad = emb.model_action_to_sim(deg_action)
    limits = [(-1.92, 1.92), (-1.745, 1.745), (-1.745, 1.571), (-1.658, 1.658), (-2.793, 2.793), tuple(GRIPPER_RANGE)]
    for v, (lo, hi) in zip(rad, limits, strict=True):
        assert lo - 1e-6 <= v <= hi + 1e-6, (v, lo, hi)


def test_so101_embodiment_declares_degrees():
    emb = load_embodiment("so101")
    assert emb.state_units == "degrees"
    assert emb.action_units == "degrees"
    assert emb.gripper_index == 5
    assert emb.gripper_joint_range == [-0.175, 1.745]


def test_so_real_embodiment_stays_native():
    """Real hardware speaks the driver units already -- must NOT double-convert."""
    emb = load_embodiment("so_real")
    assert emb.state_units == "native"
    assert emb.action_units == "native"


def test_convert_helper_does_not_mutate_input():
    src = [10.0, 20.0, 30.0]
    _convert_joint_vector(src, to_model=False)
    assert src == [10.0, 20.0, 30.0]
