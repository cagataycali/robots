"""``add_camera(parent_body=...)`` with no pose of its own is refused, not mounted 1.73 m off the wrist.

A mounted camera reads ``position`` and ``target`` in the parent body's LOCAL
frame. Their defaults - ``[1, 1, 1]`` looking at ``[0, 0, 0]`` - are the free
camera's world-frame overview, and in a body frame they put the camera 1.73 m
from the body looking back at it. Measured on the exact call ``list_bodies`` and
the camera-naming guide suggested, ``add_camera(name="wrist",
parent_body="so101/gripper")``: ``status="success"``, ``Camera 'wrist' added at
[1.0, 1.0, 1.0] (mounted on 'so101/gripper')``, and a render of the whole arm
from above with the gripper a few pixels wide. A "wrist" stream recorded under
that name carried the wrong view end to end.

Pinned here: the mount with both omitted is refused on both backends' shared
rule, the refusal names the camera, the body, the frame and (on MuJoCo, when
the robot's end-effector site sits on that body) a concrete starting pose in
that frame which is itself accepted; either coordinate supplied is enough; a
free camera keeps its defaults; a body that is not a gripper gets the generic
hint, not another body's fingertips.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation
from strands_robots.utils import mounted_camera_pose_error


@pytest.fixture
def sim():
    s = Simulation(tool_name="mounted_cam_test", mesh=False)
    s.create_world()
    assert s.add_robot(name="so101", data_config="so101")["status"] == "success"
    s.step(5)
    yield s
    s.cleanup()


def _text(result: dict) -> str:
    return result["content"][0]["text"]


def _vec(text: str, key: str) -> list[float]:
    m = re.search(rf"{key}=\[([^\]]+)\]", text)
    assert m, text
    return [float(x) for x in m.group(1).split(",")]


class TestTheSharedRule:
    def test_a_free_camera_is_never_refused(self):
        assert mounted_camera_pose_error("add_camera", "c", None, None, None) is None
        assert mounted_camera_pose_error("add_camera", "c", "", None, None) is None

    def test_either_coordinate_is_enough(self):
        assert mounted_camera_pose_error("add_camera", "c", "b", [0, 0, 0.1], None) is None
        assert mounted_camera_pose_error("add_camera", "c", "b", None, [0, 0, 1]) is None

    def test_both_omitted_is_refused_with_the_frame_named(self):
        text = mounted_camera_pose_error("add_camera", "wrist", "so101/gripper", None, None)
        assert text is not None
        assert text.startswith("add_camera: camera 'wrist' is mounted on 'so101/gripper'"), text
        assert "LOCAL frame" in text and "both were omitted" in text, text
        assert "1.73 m" in text, text
        # Generic hint when the backend has nothing better.
        assert "approach axis" in text, text


class TestOnMuJoCo:
    def test_the_gripper_mount_with_no_pose_is_refused_with_a_start_that_works(self, sim):
        result = sim.add_camera(name="wrist", parent_body="so101/gripper")
        assert result["status"] == "error", result
        text = _text(result)
        assert "end-effector site 'so101/gripper'" in text, text
        pos, tgt = _vec(text, "position"), _vec(text, "target")
        # The start is beside the approach axis and looks past the fingertips
        # (the site sits at ~[-0.008, 0, -0.098] in the gripper frame).
        assert abs(pos[0] - 0.06) < 0.02 and abs(pos[2]) < 0.06, pos
        assert tgt[2] < -0.2, tgt
        assert "wrist" not in sim.list_cameras()
        # ... and is itself accepted.
        again = sim.add_camera(name="wrist", parent_body="so101/gripper", position=pos, target=tgt)
        assert again["status"] == "success", again
        assert "wrist" in sim.list_cameras()

    def test_a_non_gripper_body_gets_the_generic_hint_not_the_fingertips(self, sim):
        result = sim.add_camera(name="chase", parent_body="so101/base")
        assert result["status"] == "error", result
        text = _text(result)
        assert "end-effector site" not in text, text
        assert "approach axis" in text, text

    def test_an_unknown_body_is_still_the_not_found_refusal(self, sim):
        text = _text(sim.add_camera(name="c", parent_body="so101/nope"))
        assert "not found" in text and "list_bodies" in text, text

    def test_a_free_camera_keeps_its_defaults(self, sim):
        result = sim.add_camera(name="overview")
        assert result["status"] == "success", result
        assert "[1.0, 1.0, 1.0]" in _text(result), result

    def test_list_bodies_names_the_pose_arguments(self, sim):
        text = _text(sim.list_bodies())
        assert "position=..., target=..." in text, text
        assert "that body's frame" in text, text
