"""End-to-end: the SO arms' real-robot agent tool drives their MuJoCo twins.

Network + MuJoCo integration, the Feetech sibling of
``test_yahboom_m3pro_twin.py``: registry lookup -> asset auto-download ->
``Robot("so101", mode="real", driver="strands", transport="twin")`` -> the SAME
agent verbs the hardware driver exposes (``sensors``, ``move_to``,
``set_torque``, ``stop``) land on the compiled model in the model's units, and
the mesh's joint read source reads the model back.

What the network-free twin tests cannot claim is claimed here: that the
degree targets the agent sends really arrive at the MJCF's joints within one
read period on both assets - the SO-101 (joints ``1``..``6``, unlimited
``ctrlrange``, kp 17.8) and the SO-100 (``Rotation``..``Jaw``, declared
``ctrlrange``, kp 50) - that gripper 0 / 100 percent land on the jaw's closed
and open stops, and that a released arm really falls under the model's gravity.

``driver="strands"`` is spelled because the SO arms' registry entries declare
no ``hardware.driver``: without it the factory builds the lerobot driver,
which has no twin.
"""

from __future__ import annotations

import asyncio
import math
import os
import sys
from typing import Any

import pytest

os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")

#: One encoder count of a 4096-count servo, in degrees - the read-back bar.
_ONE_COUNT_DEG = 360.0 / 4096
#: What the model adds on top: a position servo settles where its gain balances
#: the joint's friction/gravity (``frictionloss / kp``), so a target is reached
#: to within a degree, not a count. The model's, not the driver's.
_SETTLE_DEG = 1.0


@pytest.fixture(params=["so101", "so100"])
def twin(request, tmp_path, monkeypatch):
    """The hardware driver on the twin transport, assets in an isolated cache."""
    monkeypatch.setenv("STRANDS_ASSETS_DIR", str(tmp_path))
    import strands_robots as sr
    from strands_robots.registry import reload

    reload()
    driver = sr.Robot(request.param, mode="real", driver="strands", transport="twin")
    assert driver.connect_eagerly() is None
    yield driver
    driver.cleanup()


def _invoke(driver: Any, **request: Any) -> dict[str, Any]:
    async def _run() -> dict[str, Any]:
        tool_use: Any = {"toolUseId": "t1", "name": driver.tool_name, "input": request}
        return [event async for event in driver.stream(tool_use, {})][0]

    return asyncio.run(_run())


def _joints(driver: Any) -> dict[str, float]:
    reply = _invoke(driver, action="sensors")
    assert reply["status"] == "success", reply
    return reply["content"][0]["json"]["joint_state"]


def test_the_hardware_driver_builds_on_the_twin_and_reads_the_model(twin) -> None:
    from strands_robots.drivers.feetech import FeetechDriver
    from strands_robots.drivers.feetech.twin import FeetechTwinBus

    assert isinstance(twin, FeetechDriver)
    assert type(twin.bus).__name__ == FeetechTwinBus.__name__  # by name: another module may have reloaded feetech
    assert twin.endpoint == f"sim://{twin.tool_name}"
    assert twin.sim is not None and twin.sim.list_robots() == [twin.tool_name]
    joints = _joints(twin)
    assert set(joints) == {"shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"}
    # The model spawned at qpos 0; with no calibration 0 degrees is the middle
    # of each joint's travel, so a symmetric joint reads ~0 and an asymmetric
    # one (the SO-100's Pitch, -3.32..0.174) reads its offset. Either way the
    # number is the model's, back through the map.
    assert joints["shoulder_pan"] == pytest.approx(0.0, abs=_SETTLE_DEG)


def test_agent_degrees_arrive_at_the_models_joints_within_one_read_period(twin) -> None:
    """One ``move_to`` -> one bus read period of physics -> the joints are there on the next ``sensors``."""
    targets = {"shoulder_pan": 30.0, "wrist_flex": -20.0, "wrist_roll": 45.0}
    reply = _invoke(twin, action="move_to", targets=targets)
    assert reply["status"] == "success", reply
    assert "note" not in reply["content"][0]["json"], "nothing clamped"
    joints = _joints(twin)
    for name, target in targets.items():
        assert joints[name] == pytest.approx(target, abs=_SETTLE_DEG), (name, joints[name])
    # The same reading, straight off the model in radians, agrees with the driver's degrees.
    bus = twin.bus
    obs = twin.sim.get_observation(robot_name=twin.tool_name, skip_images=True)
    for name, target in targets.items():
        binding = bus._bindings[name]
        expected_rad = bus.counts_to_model(name, bus.to_counts(name, target))
        assert obs[binding.joint] == pytest.approx(expected_rad, abs=math.radians(_SETTLE_DEG))


def test_a_reading_is_reproducible_to_one_encoder_count(twin) -> None:
    """Two reads with no write between them differ by no more than the count they are quantised to."""
    _invoke(twin, action="move_to", targets={"elbow_flex": 15.0})
    first = _joints(twin)
    second = _joints(twin)
    for name in first:
        assert abs(first[name] - second[name]) <= _ONE_COUNT_DEG + 1e-9, name


def test_gripper_percent_lands_on_the_jaws_closed_and_open_stops(twin) -> None:
    from strands_robots.registry import get_robot

    bus = twin.bus
    jaw = bus._bindings["gripper"]
    closed_end = jaw.low if (get_robot(twin.tool_name) or {})["gripper"]["closed"] == "low" else jaw.high
    open_end = jaw.high if closed_end == jaw.low else jaw.low

    assert _invoke(twin, action="move_to", targets={"gripper": 0.0})["status"] == "success"
    obs = twin.sim.get_observation(robot_name=twin.tool_name, skip_images=True)
    assert obs[jaw.joint] == pytest.approx(closed_end, abs=0.02)
    assert _joints(twin)["gripper"] == pytest.approx(0.0, abs=1.0)

    assert _invoke(twin, action="move_to", targets={"gripper": 100.0})["status"] == "success"
    obs = twin.sim.get_observation(robot_name=twin.tool_name, skip_images=True)
    assert obs[jaw.joint] == pytest.approx(open_end, abs=0.02)
    assert _joints(twin)["gripper"] == pytest.approx(100.0, abs=1.0)


def test_a_released_arm_falls_under_the_models_gravity_and_an_energized_one_holds(twin) -> None:
    assert _invoke(twin, action="move_to", targets={"shoulder_lift": 0.0, "elbow_flex": 0.0})["status"] == "success"
    held = _joints(twin)
    assert _invoke(twin, action="set_torque", enabled=False)["status"] == "success"
    refused = _invoke(twin, action="move_to", targets={"shoulder_lift": 0.0})
    assert refused["status"] == "error" and "needs torque on" in refused["content"][0]["text"]
    twin.sim.step(n_steps=500)  # one second, limp
    limp = _joints(twin)
    moved = {name: abs(limp[name] - held[name]) for name in held}
    assert max(moved.values()) > 5.0, f"a limp arm moves under gravity: {moved}"

    assert _invoke(twin, action="set_torque", enabled=True)["status"] == "success"
    energized = _joints(twin)
    twin.sim.step(n_steps=500)
    later = _joints(twin)
    for name in later:
        assert later[name] == pytest.approx(energized[name], abs=_SETTLE_DEG), name


def test_the_mesh_joint_read_source_reads_the_model(twin) -> None:
    from strands_robots.bus_access import read_joints

    assert _invoke(twin, action="move_to", targets={"shoulder_pan": -25.0})["status"] == "success"
    joints = read_joints(twin)
    assert joints is not None
    assert joints["shoulder_pan.pos"] == pytest.approx(-25.0, abs=_SETTLE_DEG)


def test_every_declared_verb_answers_success_on_the_twin(twin) -> None:
    verbs = twin.tool_spec["inputSchema"]["json"]["properties"]["action"]["enum"]
    requests: dict[str, dict[str, Any]] = {
        "status": {},
        "sensors": {},
        "move_to": {"targets": {"shoulder_pan": 5.0, "gripper": 50.0}},
        "set_torque": {"enabled": True},
        "stop": {},
    }
    assert set(verbs) == set(requests)
    for verb in verbs:
        reply = _invoke(twin, action=verb, **requests[verb])
        assert reply["status"] == "success", (verb, reply)


def test_the_twins_cameras_render_beside_the_driver(twin) -> None:
    render = twin.sim.render(width=160, height=120)
    assert render["status"] == "success", render
