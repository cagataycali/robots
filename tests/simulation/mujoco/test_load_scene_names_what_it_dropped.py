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
    assert j == {
        "carried_robots": [],
        "dropped_robots": [],
        "dropped_objects": [],
        "dropped_cameras": [],
        "still_in_loaded_file": [],
    }


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


def _export_then_load(sim, tmp_path):
    out = str(tmp_path / "exported.xml")
    assert sim.export_xml(output_path=out)["status"] == "success"
    return sim.load_scene(out)


def test_export_xml_load_scene_round_trip_keeps_the_robot_registered(sim, tmp_path):
    assert sim.add_robot(name="so101", data_config="so101")["status"] == "success"
    r = _export_then_load(sim, tmp_path)
    assert r["status"] == "success", r
    text = _text(r)
    assert "Robot(s) ['so101'] found in the loaded file" in text and "kept registered" in text, text
    assert "dropped robot(s)" not in text
    j = _json(r)
    assert j["carried_robots"] == ["so101"] and j["dropped_robots"] == []
    assert "so101" in sim._world.robots
    robot = sim._world.robots["so101"]
    assert len(robot.joint_ids) == 6 and len(robot.actuator_ids) == 6, (robot.joint_ids, robot.actuator_ids)
    st = sim.get_robot_state("so101")
    assert st["status"] == "success", st
    assert sim._world._backend_state.get("scene_loaded") is True


def test_round_trip_then_add_robot_under_the_same_name_is_refused_as_already_registered(sim, tmp_path):
    sim.add_robot(name="so101", data_config="so101")
    _export_then_load(sim, tmp_path)
    r = sim.add_robot(name="so101", data_config="so101")
    assert r["status"] == "error"
    assert "already exists" in _text(r), _text(r)
    # The carried registration survived the refused add.
    assert "so101" in sim._world.robots and sim.get_robot_state("so101")["status"] == "success"


def test_loading_an_exported_scene_into_a_fresh_sim_then_add_robot_names_the_collision(sim, tmp_path):
    # A file exported WITH the robot, loaded where nothing is registered: the
    # subtree is in the model but unknown to the registry, so add_robot under
    # that name collides in MuJoCo. The reason travels (it used to be the bare
    # "Failed to inject robot 'so101' into scene.") with a hint.
    sim.add_robot(name="so101", data_config="so101")
    out = str(tmp_path / "exported.xml")
    assert sim.export_xml(output_path=out)["status"] == "success"
    fresh = Simulation(tool_name="load_scene_fresh", mesh=False)
    try:
        fresh.create_world()
        assert fresh.load_scene(out)["status"] == "success"
        assert fresh.list_robots() == []
        r = fresh.add_robot(name="so101", data_config="so101")
        assert r["status"] == "error"
        text = _text(r)
        assert "repeated name 'so101/" in text, text
        assert "A subtree named 'so101/' is already in the scene" in text, text
        assert "Failed to inject robot" not in text
        assert "so101" not in fresh._world.robots
    finally:
        fresh.cleanup()


def test_a_carried_robots_mesh_bridge_follows_the_new_world(sim, tmp_path):
    # ``_attach_robot_to_mesh`` points SimRobot._world at the live SimWorld so
    # the child Mesh's ``_read_state`` can read joint positions. Carrying the
    # robot over has to re-point it: left on the world load_scene discarded,
    # the child peer publishes state from a dead model. Off-mesh robots keep
    # None, the documented value for a standalone robot.
    sim.add_robot(name="so101", data_config="so101")
    robot = sim._world.robots["so101"]
    assert robot._world is None  # off-mesh: nothing bridged it
    robot._world = sim._world  # what _attach_robot_to_mesh does
    discarded = sim._world
    _export_then_load(sim, tmp_path)
    assert sim._world is not discarded
    assert sim._world.robots["so101"]._world is sim._world


def test_a_carried_off_mesh_robot_keeps_no_world_backref(sim, tmp_path):
    sim.add_robot(name="so101", data_config="so101")
    _export_then_load(sim, tmp_path)
    assert sim._world.robots["so101"]._world is None


def test_an_object_the_loaded_file_still_carries_is_not_given_an_add_object_it_would_refuse(sim, tmp_path):
    # export_xml -> load_scene: the cube's body is IN the loaded file, so it is
    # untracked, not absent. "add_object re-adds objects" was a dead end here -
    # MuJoCo refuses the re-add with "repeated name 'cube' in body", which is
    # the same shape of dead end this file exists to remove for the robot.
    sim.add_robot(name="so101", data_config="so101")
    sim.add_object(name="cube", shape="box", position=[0.25, 0, 0.05], size=[0.02, 0.02, 0.02])
    r = _export_then_load(sim, tmp_path)
    text = _text(r)
    assert "dropped object(s) ['cube']" in text, text
    assert "add_object re-adds objects" not in text, text
    assert "already carries object(s) ['cube'] under the same name" in text, text
    assert "repeated name" in text, text
    assert _json(r)["still_in_loaded_file"] == ["cube"]
    # The advice is honest: following the old one is what MuJoCo refuses.
    refused = _text(sim.add_object(name="cube", shape="box", position=[0.25, 0, 0.05], size=[0.02, 0.02, 0.02]))
    assert "repeated name 'cube'" in refused, refused


def test_the_no_robots_registered_warning_is_only_given_when_a_robot_was_dropped(sim, tmp_path, scene_path):
    # Carried robot: robot-scoped actions keep working, so claiming they refuse
    # contradicts the same envelope's own first line.
    sim.add_robot(name="so101", data_config="so101")
    sim.add_object(name="cube", shape="box", position=[0.25, 0, 0.05], size=[0.02, 0.02, 0.02])
    carried = _text(_export_then_load(sim, tmp_path))
    assert "kept registered" in carried
    assert "No robots registered" not in carried, carried
    assert sim.get_robot_state("so101")["status"] == "success"
    # Dropped robot: the warning is true, so it is given.
    sim2 = Simulation(tool_name="load_scene_warn", mesh=False)
    try:
        sim2.create_world()
        sim2.add_robot(name="so101", data_config="so101")
        assert "No robots registered" in _text(sim2.load_scene(scene_path))
    finally:
        sim2.cleanup()


def test_a_scene_without_the_robot_still_drops_it(sim, tmp_path, scene_path):
    sim.add_robot(name="so101", data_config="so101")
    r = sim.load_scene(scene_path)
    j = _json(r)
    assert j["carried_robots"] == [] and j["dropped_robots"] == ["so101"]
