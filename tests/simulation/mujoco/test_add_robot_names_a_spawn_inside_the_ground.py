"""``add_robot`` names a robot that starts inside the flat ground plane.

Terrain seating lifts a floating base out of a heightfield, but a flat plane is
left alone and a fixed-base arm is never moved, so an asset authored for a
recessed floor (LeKiwi's wheels sit 34.6 mm below its root) started the episode
inside the plane and the contact solver ejected it on the first step, behind a
plain ``success``. The result now names the depth and the ``position=`` that
spawns the robot clear, and following that advice silences it.

Hermetic: inline MJCF in ``tmp_path``, ``mesh=False``, no rendering.
"""

from __future__ import annotations

import logging

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

#: A floating box base whose bottom face sits at ``z = root - 0.05``, and a
#: fixed mount whose hanging link reaches ``z = root - 0.12``. MuJoCo generates
#: no contact for a body welded to the world, so a fixed arm is buried by its
#: moving links, which are what the solver pushes.
_FLOATING = """
<mujoco model="rover"><worldbody>
  <body name="base" pos="0 0 {z}"><freejoint/><geom type="box" size="0.1 0.1 0.05"/>
    <body name="arm" pos="0 0 0.05"><joint name="pan" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.1" size="0.02"/></body></body>
</worldbody><actuator><position joint="pan"/></actuator></mujoco>
"""
_FIXED = """
<mujoco model="arm"><worldbody>
  <body name="mount" pos="0 0 {z}"><geom type="box" size="0.05 0.05 0.05" contype="0" conaffinity="0"/>
    <body name="link"><joint name="pan" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.1" size="0.02"/></body></body>
</worldbody><actuator><position joint="pan"/></actuator></mujoco>
"""


def _add(tmp_path, xml: str, position: list[float] | None = None) -> str:
    model = tmp_path / "robot.xml"
    model.write_text(xml)
    sim = Simulation(tool_name="test_spawn_burial", mesh=False)
    try:
        sim.create_world(gravity=[0, 0, -9.81])
        result = sim.add_robot(name="bot", urdf_path=str(model), position=position)
        assert result["status"] == "success", result
        return result["content"][0]["text"]
    finally:
        sim.cleanup()


@pytest.mark.parametrize(
    ("xml", "position", "named"),
    [
        (_FLOATING.format(z=0.03), None, "starts 20.0 mm inside the ground"),
        (_FIXED.format(z=0.07), [0.2, 0.0, 0.0], "starts 50.0 mm inside the ground"),
        (_FLOATING.format(z=0.03), [0.0, 0.0, 0.02], None),  # the advice, followed
        (_FIXED.format(z=0.2), None, None),  # authored resting on the plane
    ],
    ids=["floating-base-buried", "fixed-base-buried", "lifted-clear", "resting"],
)
def test_add_robot_names_a_burial_and_only_a_burial(tmp_path, xml, position, named) -> None:
    text = _add(tmp_path, xml, position)
    if named is None:
        assert "inside the ground" not in text, text
    else:
        assert named in text, text


def test_the_named_position_spawns_the_robot_clear(tmp_path, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        text = _add(tmp_path, _FIXED.format(z=0.07), [0.2, 0.0, 0.0])
    assert "Pass position=[0.2, 0.0, 0.05]" in text, text
    assert any("50.0 mm inside the ground" in r.getMessage() for r in caplog.records), "Robot() returns no envelope"
    assert "inside the ground" not in _add(tmp_path, _FIXED.format(z=0.07), [0.2, 0.0, 0.05])
