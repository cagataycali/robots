"""``add_robot`` names a robot that starts inside the flat ground plane.

Terrain seating lifts a floating base out of a heightfield, but a flat plane is
left alone and a fixed-base arm is never moved, so an asset authored for a
recessed floor (LeKiwi's wheels sit 34.6 mm below its root) started the episode
inside the plane and the contact solver ejected it on the first step, behind a
plain ``success``. The result now names the depth and the ``position=`` that
spawns the robot clear, and following that advice silences it - on a
``keyframe=`` spawn too, whose free-joint pose is placed inside the
``position=`` frame rather than over it.

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


#: The floating base again, posed by a ``<key>`` that writes the free joint's
#: seven qpos values: the key puts the box bottom 20 mm under the plane.
_KEYED = (
    _FLOATING.replace("<freejoint/>", '<freejoint name="root"/>').replace(
        "</mujoco>", '<keyframe><key name="stand" qpos="0.1 0 0.03 1 0 0 0 0" ctrl="0"/></keyframe></mujoco>'
    )
).format(z=0.5)


def _add(tmp_path, xml: str, position: list[float] | None = None, keyframe: str | None = None) -> str:
    model = tmp_path / "robot.xml"
    model.write_text(xml)
    sim = Simulation(tool_name="test_spawn_burial", mesh=False)
    try:
        sim.create_world(gravity=[0, 0, -9.81])
        result = sim.add_robot(name="bot", urdf_path=str(model), position=position, keyframe=keyframe)
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


def test_a_keyframe_spawn_follows_the_named_position(tmp_path) -> None:
    text = _add(tmp_path, _KEYED, keyframe="stand")
    assert "Pass position=[0.0, 0.0, 0.02]" in text, text
    # Before the key was placed in the position= frame it overwrote it, so this
    # advice reported the same 20 mm and a larger position=, forever.
    assert "inside the ground" not in _add(tmp_path, _KEYED, [0.0, 0.0, 0.02], keyframe="stand")
    model = tmp_path / "robot.xml"
    sim = Simulation(tool_name="test_spawn_burial", mesh=False)
    try:
        sim.create_world(gravity=[0, 0, -9.81])
        sim.add_robot(name="bot", urdf_path=str(model), position=[1.0, 2.0, 0.02], keyframe="stand")
        world = sim._world
        assert world is not None
        for _ in range(2):  # the spawn, then reset() restoring the stored home
            base = world._data.xpos[world._model.body("bot/base").id]
            assert base.tolist() == pytest.approx([1.1, 2.0, 0.05]), base
            sim.reset()
    finally:
        sim.cleanup()
