"""Newton poses a robot with ``set_joint_positions``, as MuJoCo and Isaac do.

``NewtonSimEngine`` had no ``set_joint_positions``, so a scene script that
posed a robot before rendering or rolling out raised ``AttributeError`` on
Newton while it ran on the other two backends (docs scripts special-cased
Newton with ``send_action`` + ``step`` instead). The verb now writes the joint
coordinates, zeroes their velocities and runs forward kinematics; ``hold=True``
moves the drive targets with the pose, and the write is all-or-nothing.
"""

from __future__ import annotations

import importlib.util

import pytest

_HAS_NEWTON = importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None

pytestmark = pytest.mark.skipif(not _HAS_NEWTON, reason="newton/warp not installed")


@pytest.fixture
def engine():
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    eng = NewtonSimEngine()
    assert eng.create_world()["status"] == "success"
    assert eng.add_robot("so101", data_config="so101")["status"] == "success"
    yield eng
    eng.destroy()


def _q(engine, keys):
    obs = engine.get_observation("so101", skip_images=True)
    return [obs[k] for k in keys]


def test_the_pose_is_written_and_the_bodies_follow(engine) -> None:
    before = engine._state_0.body_q.numpy()[:, :3].copy()

    r = engine.set_joint_positions({"2": 1.0, "3": -0.5}, robot_name="so101")

    assert r["status"] == "success"
    assert _q(engine, ["2", "3"]) == pytest.approx([1.0, -0.5], abs=1e-5)
    assert abs(engine._state_0.body_q.numpy()[:, :3] - before).max() > 0.05  # forward kinematics ran


def test_the_ordered_form_binds_to_the_action_keys(engine) -> None:
    keys = engine.robot_action_keys("so101")
    assert engine.set_joint_positions([0.1] * len(keys))["status"] == "success"
    assert _q(engine, keys) == pytest.approx([0.1] * len(keys), abs=1e-5)


def test_hold_keeps_the_pose_through_a_step_and_the_default_does_not(engine) -> None:
    engine.set_joint_positions({"2": 0.3}, robot_name="so101")
    engine.step(100)
    assert abs(_q(engine, ["2"])[0] - 0.3) > 0.1  # the drive pulled it back

    engine.set_joint_positions({"2": 0.3}, robot_name="so101", hold=True)
    engine.step(100)
    assert _q(engine, ["2"])[0] == pytest.approx(0.3, abs=0.06)


@pytest.mark.parametrize(
    ("positions", "match"),
    [
        ({"2": 99.0}, "outside the joint limits"),
        ({"nope": 0.1}, "not joints of"),
        ({"2": float("nan")}, ""),
        ({}, "empty"),
        (None, "required"),
    ],
)
def test_a_bad_write_writes_nothing(engine, positions, match) -> None:
    before = _q(engine, ["2"])
    r = engine.set_joint_positions(positions, robot_name="so101")
    assert r["status"] == "error"
    assert match in r["content"][0]["text"]
    assert _q(engine, ["2"]) == before


def test_hold_must_be_a_boolean(engine) -> None:
    r = engine.set_joint_positions({"2": 0.1}, robot_name="so101", hold="false")
    assert r["status"] == "error"


def test_the_ground_is_drawn_as_a_floor_not_a_backdrop(engine) -> None:
    """DA-002's other half: Newton's default ground colour sat a few levels off
    the clear colour, so frames showed no floor. It is now a visible checkerboard."""
    import numpy as np

    engine.add_camera("front", position=[0.6, 0.0, 0.35], target=[0.0, 0.0, 0.05], width=160, height=120)
    rgb, _ = engine.get_frame("front")
    floor = rgb[90:].astype(float)
    assert engine._ground_shape is not None
    assert floor.std() > 5.0, "a flat, featureless floor: the checkerboard did not reach the ground shape"
    assert abs(floor.mean() - rgb[:5].astype(float).mean()) > 5.0 or floor.std() > 5.0
    assert np.isfinite(floor).all()
