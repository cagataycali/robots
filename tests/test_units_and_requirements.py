"""Units belong to the policy's dataset; a policy declares its needs (experiments/SMOLVLA.md in the reference).

The pure parts first, then what ``Robot.run`` does with them.
"""

from __future__ import annotations

import numpy as np
import pytest

from strands_robots import Description, JointState, Observation, Policy, Requirements, Robot, Simulation
from strands_robots.units import Units, declared_units, resolve_units


@pytest.fixture(scope="module")
def so101():
    return Description.load("so101")


def test_lerobot_so10x_units_round_trip_on_so101(so101):
    u = Units.for_robot(so101, "lerobot_so10x")
    lo = np.array([so101.limits[k][0] for k in so101.action_keys])
    hi = np.array([so101.limits[k][1] for k in so101.action_keys])
    # body joints map their limits onto -100..100, the gripper onto 0..100
    np.testing.assert_allclose(u.from_rad(lo), [-100, -100, -100, -100, -100, 0])
    np.testing.assert_allclose(u.from_rad(hi), [100, 100, 100, 100, 100, 100])
    np.testing.assert_allclose(u.from_rad((lo + hi) / 2), [0, 0, 0, 0, 0, 50])
    q = np.linspace(lo, hi, 7)
    np.testing.assert_allclose(u.to_rad(u.from_rad(q)), q, atol=1e-12)


def test_rad_units_are_identity_and_unknown_names_refuse(so101):
    assert Units.for_robot(so101, "rad").is_identity
    with pytest.raises(ValueError, match="unknown units 'degrees'"):
        Units.for_robot(so101, "degrees")


class _Speaks:
    def __init__(self, units=None):
        if units is not None:
            self.units = units


def test_resolve_units_follows_the_declaration_and_rad_or_none_disable_it(so101):
    lerobot = _Speaks("lerobot_so10x")
    assert resolve_units(lerobot, "rad", so101) is None
    assert resolve_units(lerobot, None, so101) is None
    assert resolve_units(lerobot, "auto", so101).name == "lerobot_so10x"
    assert resolve_units(_Speaks(), "auto", so101) is None  # cannot say -> radians
    assert resolve_units(_Speaks(), "lerobot_so10x", so101).name == "lerobot_so10x"
    with pytest.raises(ValueError, match="policy declares units 'furlongs'"):
        declared_units(_Speaks("furlongs"))


def _obs(n_joints: int, **cams: tuple[int, int]) -> Observation:
    names = tuple(f"j{i}" for i in range(n_joints))
    images = {c: np.zeros((h, w, 3), np.uint8) for c, (w, h) in cams.items()}
    return Observation(0.0, JointState.zeros(names), images=images)


_THREE_CAMERAS = Requirements(images={"camera1": (256, 256), "camera2": (256, 256), "camera3": None}, state_dim=6)


@pytest.mark.parametrize(
    ("obs", "refusal"),
    [
        (_obs(6), r"needs camera\(s\) \['camera1', 'camera2', 'camera3'\] but the observation has none"),
        (
            _obs(6, camera1=(256, 256), camera2=(320, 240), camera3=(64, 64)),
            "camera2: has 320x240, needs 256x256",
        ),
        (
            _obs(29, camera1=(256, 256), camera2=(256, 256), camera3=(64, 64)),
            "expects a 6-dim state, the robot reports 29 joints",
        ),
        (_obs(6, camera1=(256, 256), camera2=(256, 256), camera3=(64, 64)), None),
    ],
)
def test_requirements_refuse_naming_what_is_missing(obs, refusal):
    if refusal is None:
        _THREE_CAMERAS.check(obs)
        return
    with pytest.raises(ValueError, match=refusal):
        _THREE_CAMERAS.check(obs)


@pytest.fixture
def arm():
    with Simulation() as sim:
        yield Robot("so101", sim=sim)


class _HoldInLerobotUnits(Policy):
    """A policy that speaks lerobot_so10x: it sees -100..100 and asks for the gripper fully open (100)."""

    units = "lerobot_so10x"

    def __init__(self):
        self.seen = []

    def act(self, obs, *, instruction=None):
        self.seen.append(obs.joints.positions.copy())
        a = obs.joints.positions.copy()
        a[-1] = 100.0
        return a


def test_run_converts_both_directions_when_the_policy_declares_units(arm):
    pol = _HoldInLerobotUnits()
    arm.run(pol, steps=3)
    assert all(abs(s[:-1]).max() <= 100 for s in pol.seen) and 0 <= pol.seen[0][-1] <= 100  # it saw bus units
    arm.run(pol, steps=20)
    assert abs(arm.observe()["gripper"] - arm.description.limits["gripper"][1]) < 0.15  # the arm got radians


class _Needs(Policy):
    def __init__(self, req):
        self.req = req

    def requirements(self):
        return self.req

    def act(self, obs, *, instruction=None):
        return obs.joints.positions


def test_run_refuses_before_inference_naming_the_missing_camera(arm):
    with pytest.raises(ValueError, match=r"needs camera\(s\) \['camera1', 'camera2', 'camera3'\]"):
        arm.run(_Needs(_THREE_CAMERAS), steps=1)
    for i, size in ((1, (256, 256)), (2, (320, 240)), (3, (64, 64))):
        arm.sim.camera(f"camera{i}", size=size, attach=arm)
    with pytest.raises(ValueError, match="camera2: has 320x240, needs 256x256"):
        arm.run(_Needs(_THREE_CAMERAS), steps=1)


def test_run_passes_when_the_requirements_are_met(arm):
    for i in (1, 2, 3):
        arm.sim.camera(f"camera{i}", size=(256, 256), attach=arm)
    assert arm.run(_Needs(_THREE_CAMERAS), steps=2).steps == 2


def test_run_names_a_state_length_mismatch():
    with Simulation() as sim, pytest.raises(ValueError, match="expects a 6-dim state, the robot reports"):
        Robot("g1", sim=sim).run(_Needs(Requirements(state_dim=6)), steps=1)
