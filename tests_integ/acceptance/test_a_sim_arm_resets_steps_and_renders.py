"""Acceptance: sim reset / step / render moves a real SO-101 MJCF and returns a frame.

``Robot("so101", mode="sim")`` loads the shipped SO-101 model into MuJoCo. A
joint command followed by ``step`` must move that joint and change what the
camera sees; ``render`` must hand back a ``480x640`` image; ``reset`` must put
the arm back where it started. Real MuJoCo, real MJCF, no doubles.
"""

from __future__ import annotations

import io
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")

pytest.importorskip("mujoco")
Image = pytest.importorskip("PIL.Image")

ROBOT = "so101"
JOINT = "2"  # shoulder lift
TARGET = 0.5  # rad, inside the joint's range


def test_a_sim_arm_resets_steps_and_renders() -> None:
    from strands_robots import Robot

    robot = Robot(ROBOT, mode="sim")
    try:
        assert robot.reset()["status"] == "success"
        rest = robot.get_observation(ROBOT)
        assert rest["default"].shape == (480, 640, 3)

        assert robot.send_action({JOINT: TARGET}, robot_name=ROBOT)["status"] == "success"
        assert robot.step(200)["status"] == "success"
        moved = robot.get_observation(ROBOT)
        assert abs(moved[JOINT] - TARGET) < 0.05, moved[JOINT]
        changed = np.abs(moved["default"].astype(int) - rest["default"].astype(int)).max(axis=-1)
        assert (changed > 16).mean() > 0.001, "the frame did not show the arm move"

        rendered = robot.render()
        assert rendered["status"] == "success", rendered
        png = next(block["image"]["source"]["bytes"] for block in rendered["content"] if "image" in block)
        assert Image.open(io.BytesIO(png)).size == (640, 480)

        assert robot.reset()["status"] == "success"
        assert robot.get_observation(ROBOT, skip_images=True)[JOINT] == rest[JOINT]
    finally:
        robot.destroy()
