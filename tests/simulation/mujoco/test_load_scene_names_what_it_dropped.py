"""``load_scene`` names the robots, objects and cameras its world swap discarded, and how to put them back.

``load_scene`` replaces ``self._world`` with a fresh ``SimWorld``: every
registered robot, object and camera is gone and the loaded file is the whole
scene. Through the agent tool the caller is a ``Robot("so101", mode="sim")``
facade named after the very arm this drops, and the result said only
"Scene loaded from table.xml / Bodies: 3" - the loss surfaced two calls later
as "No robots registered in the simulation". ``add_robot`` mutates the loaded
spec in place, so the arm can go back INTO the loaded scene; the result now
says so, spelled with the data_config the robot was registered under.

Pinned: dropped robot named with its add_robot call; dropped user camera and
object named; the seeded free camera "default" is not reported as dropped;
a fresh world (nothing registered) keeps the historical text; json block
carries the three lists; the recovery actually works.
"""

from __future__ import annotations

import textwrap

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation, _load_scene_dropped_line

_SCENE = textwrap.dedent(
    """
    <mujoco model="table">
      <worldbody>
        <light pos="0 0 2"/>
        <geom name="floor" type="plane" size="2 2 0.1"/>
        <body name="table" pos="0.3 0 0.2"><geom type="box" size="0.3 0.3 0.02"/></body>
      </worldbody>
    </mujoco>
    """
)


@pytest.fixture
def scene_path(tmp_path):
    p = tmp_path / "table.xml"
    p.write_text(_SCENE)
    return str(p)


@pytest.fixture
def sim():
    s = Simulation(tool_name="load_scene_dropped", mesh=False)
    s.create_world()
    yield s
    s.cleanup()


def _text(result: dict) -> str:
    return "\n".join(c["text"] for c in result["content"] if isinstance(c, dict) and "text" in c)


def _json(result: dict) -> dict:
    return next(c["json"] for c in result["content"] if isinstance(c, dict) and "json" in c)


def test_dropped_robot_is_named_with_its_add_robot_call(sim, scene_path):
    assert sim.add_robot(name="so101", data_config="so101")["status"] == "success"
    r = sim.load_scene(scene_path)
    assert r["status"] == "success", r
    text = _text(r)
    assert "REPLACED the live world: dropped robot(s) ['so101']" in text, text
    assert "add_robot(name='so101', data_config='so101') puts the arm back INTO the loaded scene" in text, text
    assert "No robots registered" in text
    assert _json(r)["dropped_robots"] == ["so101"]
    assert sim.list_robots() == []


def test_dropped_object_and_camera_are_named_but_the_seeded_free_camera_is_not(sim, scene_path):
    assert sim.add_robot(name="so101", data_config="so101")["status"] == "success"
    assert (
        sim.add_object(name="cube", shape="box", position=[0.3, 0, 0.3], size=[0.02, 0.02, 0.02])["status"] == "success"
    )
    assert (
        sim.add_camera(name="wrist", parent_body="so101/gripper", position=[0.05, 0, -0.03], target=[0, 0, -0.3])[
            "status"
        ]
        == "success"
    )
    assert "default" in sim._world.cameras  # the seeded free camera
    r = sim.load_scene(scene_path)
    text = _text(r)
    assert "object(s) ['cube']" in text and "camera(s) ['wrist']" in text, text
    assert "'default'" not in text, text
    j = _json(r)
    assert j["dropped_objects"] == ["cube"] and j["dropped_cameras"] == ["wrist"]
    assert "add_object re-adds objects" in text and "add_camera re-adds cameras" in text


def test_fresh_world_keeps_the_historical_text(sim, scene_path):
    r = sim.load_scene(scene_path)
    text = _text(r)
    assert "Scene loaded from table.xml" in text and "REPLACED" not in text and "dropped" not in text, text
    j = _json(r)
    assert j == {"dropped_robots": [], "dropped_objects": [], "dropped_cameras": []}


def test_the_named_recovery_works(sim, scene_path):
    sim.add_robot(name="so101", data_config="so101")
    sim.load_scene(scene_path)
    assert sim.add_robot(name="so101", data_config="so101")["status"] == "success"
    st = sim.get_robot_state("so101")
    assert st["status"] == "success", st
    # The loaded scene is still there: the table body survived the re-add.
    assert sim._world._model.body("table") is not None


def test_line_without_a_known_data_config_leaves_a_placeholder():
    line = _load_scene_dropped_line(["arm"], [], [], {"arm": None})
    assert "add_robot(name='arm', data_config=...)" in line
    assert _load_scene_dropped_line([], [], []) == ""
