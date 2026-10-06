"""The native Feetech driver refuses to front ``lekiwi`` by default because
its motor map is the SO-arm six, which leaves the three omniwheels of lekiwi's
base unreachable through this bus. ``motor_ids=`` over the arm half is the
documented escape hatch and still constructs. ``so100`` / ``so101`` are the
whole robot through this driver and are unaffected.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")


def _sim_for(name: str):
    from strands_robots import Robot

    return Robot(name)


def test_feetech_driver_refuses_lekiwi_without_motor_ids_and_names_lerobot():
    from strands_robots import Robot

    sim = _sim_for("lekiwi")
    try:
        with pytest.raises(ValueError) as info:
            Robot("lekiwi", mode="real", driver="strands", transport="twin", sim=sim)
    finally:
        try:
            sim.cleanup()
        except Exception:
            pass

    text = str(info.value)
    # The refusal names the driver (not just the bus), the actuator the user
    # cares about (omniwheels), and the driver that would cover the whole robot.
    assert "FeetechDriver" in text
    assert "lekiwi" in text
    assert "omniwheels" in text
    assert "driver='lerobot'" in text
    assert "strands-robots[lerobot]" in text


def test_feetech_driver_accepts_lekiwi_with_motor_ids_over_the_arm_half():
    from strands_robots import Robot

    sim = _sim_for("lekiwi")
    try:
        arm = Robot(
            "lekiwi",
            mode="real",
            driver="strands",
            transport="twin",
            sim=sim,
            motor_ids=(1, 2, 3, 4, 5, 6),
        )
        # Narrowed to the arm half; no wheel keys claimed.
        assert set(arm.MOTORS) == {"shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"}
        try:
            arm.cleanup()
        except Exception:
            pass
    finally:
        try:
            sim.cleanup()
        except Exception:
            pass


@pytest.mark.parametrize("name", ["so100", "so101"])
def test_feetech_driver_still_constructs_cleanly_for_whole_robot_so_arms(name: str):
    from strands_robots import Robot

    sim = _sim_for(name)
    try:
        arm = Robot(name, mode="real", driver="strands", transport="twin", sim=sim)
        assert type(arm).__name__ == "FeetechDriver"
        try:
            arm.cleanup()
        except Exception:
            pass
    finally:
        try:
            sim.cleanup()
        except Exception:
            pass
