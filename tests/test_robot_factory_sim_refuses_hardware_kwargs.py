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


def test_the_refusal_names_the_driver_not_the_physicality():
    # Two names in the set are not descriptions of a physical robot at all:
    # mock= asks for a mocked servo bus and is_simulation= points the lerobot
    # driver at a simulator. Telling that caller their keyword "describes a
    # physical robot" is false, and for is_simulation= it contradicts the
    # remedy printed on the same line. What every name has in common is the
    # hardware driver, which mode="sim" never builds.
    for key in ("mock", "is_simulation"):
        with pytest.raises(TypeError) as info:
            _reject_hardware_kwargs_in_sim({key: True}, "so101", "sim")
        text = str(info.value)
        assert "describe a physical robot" not in text
        assert "configure a hardware driver" in text
        assert "mode='real'" in text


def test_no_refused_keyword_is_a_spawn_keyword_of_a_shipped_sim_backend():
    # The refusal reads the hardware class's forwardable set, so a name added
    # there lands here without review. That is the point - and the risk: a
    # name a sim backend ALSO binds would start refusing a call the backend
    # honours today. Nothing else pins that, so pin it against every backend
    # Robot(mode="sim") can resolve.
    import dataclasses
    import inspect

    from strands_robots.simulation.isaac.config import IsaacConfig
    from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    spawn_keywords = {
        "mujoco": set(inspect.signature(MuJoCoSimEngine.__init__).parameters),
        "newton": set(inspect.signature(NewtonSimEngine.__init__).parameters),
        "isaac": {field.name for field in dataclasses.fields(IsaacConfig)},
    }
    for backend, names in spawn_keywords.items():
        collision = sorted(names & set(_FORWARDABLE_KWARGS))
        assert not collision, (
            f"{backend} binds {collision}, which Robot(mode='sim') now refuses: "
            "a working call would start raising. Carve the name out of the refusal."
        )
    # Non-vacuity: the sets are the real ones, not empty lookups.
    assert "default_timestep" in spawn_keywords["mujoco"]
    assert "num_envs" in spawn_keywords["isaac"]


def test_a_named_tool_and_the_mode_refusal_coexist_on_one_call():
    # Robot(tool_name=) grades the caller's own parameter; this refusal grades
    # the mode the call resolved to. Both sit on the sim path, so the pair has
    # an order and a shared call, and neither guard's own tests exercise the
    # other. A malformed value of a parameter the caller wrote answers first:
    # nothing can be said about which mode that call wanted until the name it
    # asked for is usable. A usable name leaves the mode refusal in charge.
    with pytest.raises(ValueError, match="is not a valid tool name") as bad_name:
        Robot("so101", mode="sim", tool_name="left arm", port="/dev/cu.usbmodem5AB01818061")
    assert "mode='real'" not in str(bad_name.value)

    with pytest.raises(TypeError, match=r"port=.*Add mode='real'"):
        Robot("so101", mode="sim", tool_name="left_arm", port="/dev/cu.usbmodem5AB01818061")

    # And a named simulated tool with no hardware keyword is built, under the
    # name asked for rather than the "<name>_sim" default.
    arm = Robot("so101", mode="sim", tool_name="left_arm")
    try:
        assert arm.tool_name == "left_arm"
    finally:
        arm.destroy()
