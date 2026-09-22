"""End-to-end: the M3 Pro's real-robot agent tool drives its MuJoCo twin.

Network + MuJoCo integration, the sibling of ``test_yahboom_m3pro_sim.py``:
registry lookup -> asset auto-download -> ``Robot("yahboom_m3pro", mode="real",
transport="twin")`` -> the SAME agent verbs the hardware driver exposes
(``home``, ``arm``, ``gripper``, ``move``, ``sensors``) land on the compiled
model in the model's units, and ``get_observation`` reads the model back.

What the network-free twin tests cannot claim is claimed here: that the degree
targets the agent sends really arrive at the MJCF's joints after the servo
travel time, that a body-frame twist really moves ``base_x``/``base_y`` in the
world frame, and that the twin's clamp note fires when the agent asks for more
than the model's velocity actuators allow.
"""

from __future__ import annotations

import asyncio
import math
import os
import sys
from typing import Any

import pytest

os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")


@pytest.fixture
def twin(tmp_path, monkeypatch):
    """The hardware driver on the twin transport, assets in an isolated cache."""
    monkeypatch.setenv("STRANDS_ASSETS_DIR", str(tmp_path))
    import strands_robots as sr
    from strands_robots.registry import reload

    reload()
    driver = sr.Robot("yahboom_m3pro", mode="real", transport="twin")
    assert driver.connect_eagerly() is None
    yield driver
    driver.cleanup()


def _invoke(driver: Any, **request: Any) -> dict[str, Any]:
    async def _run() -> dict[str, Any]:
        tool_use: Any = {"toolUseId": "t1", "name": driver.tool_name, "input": request}
        return [event async for event in driver.stream(tool_use, {})][0]

    return asyncio.run(_run())


def test_the_hardware_driver_builds_on_the_twin_and_reads_the_home_pose(twin) -> None:
    from strands_robots.drivers.yahboom_m3pro import YahboomM3ProDriver

    assert isinstance(twin, YahboomM3ProDriver)
    assert twin.endpoint == "sim://yahboom_m3pro"
    obs = twin.get_observation()
    # The engine was built at the ``home`` keyframe: arm3/arm4 folded at -pi/2.
    assert obs["arm3.pos"] == pytest.approx(-math.pi / 2, abs=0.02)
    assert obs["arm4.pos"] == pytest.approx(-math.pi / 2, abs=0.02)
    assert obs["gripper.pos"] == pytest.approx(0.0, abs=0.02)


def test_agent_degrees_arrive_at_the_models_joints(twin) -> None:
    """45/90/90/90/180/60 -> arm1 -pi/4, arm2..4 0, arm5 pi/2, gripper -1.232 - after 1.5 s of servo travel."""
    result = _invoke(twin, action="arm", joints=[45, 90, 90, 90, 180, 60], time_ms=1500)
    assert result["status"] == "success", result
    obs = twin.get_observation()
    assert obs["arm1.pos"] == pytest.approx(-math.pi / 4, abs=0.03)
    assert obs["arm2.pos"] == pytest.approx(0.0, abs=0.03)
    assert obs["arm5.pos"] == pytest.approx(math.pi / 2, abs=0.03)
    assert obs["gripper.pos"] == pytest.approx(-1.232, abs=0.03)

    assert _invoke(twin, action="gripper", open=False)["status"] == "success"
    assert twin.get_observation()["gripper.pos"] == pytest.approx(-1.54, abs=0.03)


def test_a_held_move_drives_the_base_and_the_watchdog_stops_it(twin) -> None:
    result = _invoke(twin, action="move", linear_x=0.3, duration_s=2.0)
    assert result["status"] == "success", result
    assert result["content"][0]["json"]["stopped"] is True
    obs = twin.get_observation()
    # 0.3 m/s for 2.0 s plus the ~0.3 s watchdog hold, on a velocity servo that tracks x well.
    assert obs["base_x.pos"] == pytest.approx(0.3 * 2.3, abs=0.08)
    assert obs["base_y.pos"] == pytest.approx(0.0, abs=0.01)
    # Stopped: another read a moment later is the same place.
    twin.sim.step(n_steps=250)
    assert twin.get_observation()["base_x.pos"] == pytest.approx(obs["base_x.pos"], abs=0.01)


def test_a_strafe_after_a_turn_is_rotated_into_the_world_frame(twin) -> None:
    """Face +y, strafe left (+y body) -> the world sees -x, not +y."""
    assert _invoke(twin, action="move", angular_z=1.0, duration_s=2.0)["status"] == "success"
    yaw = twin.get_observation()["base_yaw.pos"]
    assert yaw > 0.5, "the base turned counter-clockwise"
    before = twin.get_observation()
    assert _invoke(twin, action="move", linear_y=0.2, duration_s=1.0)["status"] == "success"
    after = twin.get_observation()
    dx, dy = after["base_x.pos"] - before["base_x.pos"], after["base_y.pos"] - before["base_y.pos"]
    heading = math.atan2(dy, dx)
    # Body +y is world yaw + pi/2.
    assert math.cos(heading - (yaw + math.pi / 2)) > 0.95, (heading, yaw)


def test_sensors_read_the_twins_odometry(twin) -> None:
    _invoke(twin, action="move", linear_x=0.2, duration_s=1.0)
    sensors = _invoke(twin, action="sensors")
    assert sensors["status"] == "success", sensors
    odom = sensors["content"][1]["json"]["odom"]
    assert odom["pose"]["pose"]["position"]["x"] == pytest.approx(0.2 * 1.3, abs=0.06)


def test_a_command_past_the_models_ctrlrange_is_reported_not_hidden(twin, caplog) -> None:
    """The robot's envelope is 1.0 m/s; the model's base_x actuator clamps at 0.5 - and says so."""
    from strands_robots.drivers.yahboom_m3pro import CMD_VEL_TOPIC

    graph = twin._twin
    with caplog.at_level("WARNING"):
        reply = graph("publish", topic=CMD_VEL_TOPIC, fields={"linear": {"x": 0.9}, "angular": {}}, count=1, rate=10.0)
    assert reply["status"] == "success"
    assert "clamped" in reply["content"][0]["text"] and "base_x 0.9 -> 0.5" in reply["content"][0]["text"]
    assert any("clamped" in record.message for record in caplog.records)


def test_the_twins_cameras_render_beside_the_driver(twin) -> None:
    render = twin.sim.render(camera_name="yahboom_m3pro/wrist", width=320, height=240)
    assert render["status"] == "success", render
