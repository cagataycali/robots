"""A hardware-only keyword on a simulated Robot() is refused, naming the mode."""

from unittest.mock import patch

import pytest

from strands_robots.hardware_robot import _FORWARDABLE_KWARGS
from strands_robots.robot import Robot, _reject_hardware_kwargs_in_sim

pytest.importorskip("mujoco")


def test_port_on_a_sim_robot_is_refused_with_the_mode_remedy():
    with pytest.raises(TypeError, match=r"port=.*Add mode='real'") as info:
        Robot("so101", port="/dev/cu.usbmodem5AB01818061")
    assert "mode='sim' is the default" in str(info.value)


def test_every_hardware_keyword_is_named_in_one_refusal():
    with pytest.raises(TypeError) as info:
        Robot("so101", mode="sim", robot_ip="10.0.0.2", kp=[1.0], calibration_dir="/tmp")
    text = str(info.value)
    assert "robot_ip=, kp=, calibration_dir=" in text


def test_auto_mode_that_fell_back_to_sim_says_so():
    with patch("strands_robots.robot._auto_detect_mode", return_value="sim"):
        with pytest.raises(TypeError, match="mode='auto' found no servo bus"):
            Robot("so101", mode="auto", port="/dev/cu.usbmodem5AB01818061")


def test_cross_backend_options_still_pass_through():
    # The tolerance the refusal must NOT break: another backend's options.
    arm = Robot("so101", mode="sim", num_envs=4, device="cpu")
    try:
        assert arm.tool_name == "so101_sim"
    finally:
        arm.destroy()


def test_helper_is_silent_without_hardware_keywords():
    assert _reject_hardware_kwargs_in_sim({"num_envs": 4}, "so101", "sim") is None


def test_refused_set_is_the_hardware_forwardable_set():
    # One source of truth: every name the hardware class forwards is refused
    # here, and nothing else is.
    for key in _FORWARDABLE_KWARGS:
        with pytest.raises(TypeError, match=f"{key}="):
            _reject_hardware_kwargs_in_sim({key: 1}, "so101", "sim")
