"""A robot label and an object name cannot be the same string.

A robot's label is not one of its body names (``so101/base``,
``so101/gripper``, ...), so MuJoCo's repeated-name check never saw the
collision ``add_object(name="so101")`` creates - but every by-name reader
did. Measured: ``get_body_state(body_name="so101")`` answered "not found. Did
you mean: so101/base, ..." until the object was added, then answered with
the box's pose; ``add_robot(name="cube")`` over an object ``cube`` likewise
succeeded and left ``cube`` resolving to the box. Both directions are now
refused before anything is registered, naming the readers that would have
been misled and the remedy.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation

ARM_XML = """
<mujoco model="arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 0.1">
      <joint name="pan" type="hinge" axis="0 0 1"/>
      <geom type="cylinder" size="0.05 0.05"/>
    </body>
  </worldbody>
  <actuator>
    <position name="pan_act" joint="pan" kp="50"/>
  </actuator>
</mujoco>
"""


def _text(result) -> str:
    return " ".join(c["text"] for c in result["content"] if "text" in c)


@pytest.fixture
def sim(tmp_path):
    (tmp_path / "arm.xml").write_text(ARM_XML)
    s = Simulation(tool_name="name_collision", mesh=False)
    s.create_world()
    assert s.add_robot(name="arm", urdf_path=str(tmp_path / "arm.xml"))["status"] == "success"
    yield s
    s.cleanup()


def _box(sim: Simulation, name: str):
    return sim.add_object(name=name, shape="box", size=[0.02, 0.02, 0.02], position=[0.3, 0, 0.02])


class TestAnObjectCannotTakeARobotsName:
    def test_refused_before_registration(self, sim):
        result = _box(sim, "arm")
        assert result["status"] == "error"
        text = _text(result)
        assert text.startswith("add_object: 'arm' is the name of a robot in this world")
        assert "get_body_state" in text and "'arm/<body>'" in text and "Pick another name" in text
        assert "No objects" in _text(sim.list_objects())

    def test_the_robot_keeps_answering_for_its_own_name(self, sim):
        _box(sim, "arm")
        result = sim.get_body_state(body_name="arm")
        assert result["status"] == "error"
        assert "Did you mean: arm/base" in _text(result)

    def test_another_name_is_fine(self, sim):
        assert _box(sim, "arm_marker")["status"] == "success"


class TestARobotCannotTakeAnObjectsName:
    def test_refused_before_registration(self, sim, tmp_path):
        assert _box(sim, "cube")["status"] == "success"
        result = sim.add_robot(name="cube", urdf_path=str(tmp_path / "arm.xml"))
        assert result["status"] == "error"
        text = _text(result)
        assert text.startswith("Robot name 'cube' is already an object in this world")
        assert "omit name= to auto-number" in text
        assert "cube" not in sim._world.robots

    def test_the_duplicate_robot_refusal_still_comes_first(self, sim, tmp_path):
        result = sim.add_robot(name="arm", urdf_path=str(tmp_path / "arm.xml"))
        assert result["status"] == "error"
        assert "Robot 'arm' already exists" in _text(result)
