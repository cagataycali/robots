import numpy as np
import pytest

from strands_robots import JointState, Observation, Pose
from strands_robots.robot.drivers import Calibration
from strands_robots.robot.drivers.calibration import JointCalibration


def test_joint_state_is_frozen_and_indexable():
    js = JointState(("a", "b"), [1.0, 2.0], [0, 0], [0, 0])
    assert js["b"] == 2.0 and len(js) == 2
    with pytest.raises(ValueError):
        js.positions[0] = 9.0
    with pytest.raises(KeyError, match="unknown joint 'z'"):
        js["z"]


def test_joint_state_shape_checked():
    with pytest.raises(ValueError, match="positions"):
        JointState(("a",), [1.0, 2.0], [0], [0])


def test_pose_shapes():
    p = Pose.identity((1, 2, 3))
    assert p.position.tolist() == [1, 2, 3] and p.orientation[0] == 1.0
    with pytest.raises(ValueError, match="orientation"):
        Pose([0, 0, 0], [1, 0, 0])


def test_observation_to_lerobot():
    js = JointState(("a",), [0.5], [0], [0])
    img = np.zeros((4, 4, 3), np.uint8)
    o = Observation(1.0, js, images={"front": img})
    lr = o.to_lerobot()
    assert lr["observation.state"].dtype == np.float32
    assert lr["observation.images.front"] is img
    assert o["a"] == 0.5


def test_calibration_roundtrips_lerobot_json():
    data = {"shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": -2047, "range_min": 758, "range_max": 3292}}
    c = Calibration.from_json(data)
    assert c.joints["shoulder_pan"] == JointCalibration(1, 0, -2047, 758, 3292)
    assert c.to_json() == data
