"""attach_bodies says whether the two bodies touch, and by how much they miss.

An agent running the README quickstart ("pick up the red cube") could not
reach the cube, closed the gripper centimetres away, welded the cube to the
jaw with ``attach_bodies`` and reported a successful pick. The object was
riding along 7 cm below the fingers. The tool now measures the closest
surface distance between the parent's subtree and the child, prints it when
they do not touch, and puts ``gap_m`` / ``touching`` in the json payload.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    s = Simulation(tool_name="t", mesh=False)
    s.create_world()
    # add_object sizes are full extents: carrier spans z 0.475..0.525.
    s.add_object("carrier", shape="box", size=[0.05, 0.05, 0.05], position=[0, 0, 0.5])
    yield s
    s.cleanup()


def _parts(result):
    text = next(c["text"] for c in result["content"] if "text" in c)
    payload = next(c["json"] for c in result["content"] if "json" in c)
    return text, payload


@pytest.mark.parametrize("mode", ["weld", "kinematic"])
def test_a_gap_is_named_in_centimetres(sim, mode):
    sim.add_object("cube", shape="box", size=[0.02, 0.02, 0.02], position=[0, 0, 0.57])
    result = sim.attach_bodies("carrier", "cube", mode=mode)
    assert result["status"] == "success", "attaching at a distance stays allowed - mounting is a real use"
    text, payload = _parts(result)
    assert "NOT touching" in text
    assert "3.5 cm" in text
    assert "not a pick" in text
    assert payload["touching"] is False
    assert payload["gap_m"] == pytest.approx(0.035, abs=1e-3)
    assert payload["mode"] == mode


def test_touching_bodies_say_so(sim):
    # cube bottom at 0.525 = carrier top.
    sim.add_object("cube", shape="box", size=[0.02, 0.02, 0.02], position=[0, 0, 0.535])
    text, payload = _parts(sim.attach_bodies("carrier", "cube", mode="weld"))
    assert "The bodies are touching." in text
    assert "NOT touching" not in text
    assert payload["touching"] is True
    assert payload["gap_m"] <= 1e-3


def test_robot_parent_measures_from_its_whole_subtree(sim):
    """A gripper's pads live on links below the body the caller names."""
    sim.add_robot("so101")
    sim.add_object("red_cube", shape="box", size=[0.04, 0.04, 0.04], position=[0.15, 0, 0.02])
    text, payload = _parts(sim.attach_bodies("so101/gripper", "red_cube", mode="kinematic"))
    assert payload["touching"] is False
    assert payload["gap_m"] > 0.1, "the arm at rest is far from a cube on the table"
    assert "floats rigidly" in text
