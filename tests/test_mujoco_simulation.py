"""Integration tests for the MuJoCo Simulation class.

Tests the full Simulation public API through behavioral end-to-end scenarios
— create worlds, add robots/objects/cameras, step physics, render, record,
randomize, dispatch actions, and clean up.

Every test exercises real user-visible behavior. No isinstance checks or
attribute-existence tests.

Run: MUJOCO_GL=osmesa python -m pytest tests/test_mujoco_simulation.py -v
"""

import json
import os
import shutil
import tempfile

import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.backend import _can_render  # noqa: E402

requires_gl = pytest.mark.skipif(
    not _can_render(),
    reason="No OpenGL context available (headless without EGL/OSMesa)",
)

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# ── Test robot XML ──

ROBOT_XML = """
<mujoco model="test_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <camera name="front" pos="1.5 0 1" xyaxes="0 1 0 -0.5 0 1"/>
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.05 0.05" rgba="0.3 0.3 0.8 1"/>
      <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <body name="link1" pos="0 0 0.1">
        <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.2" rgba="0.8 0.3 0.3 1"/>
        <joint name="shoulder_lift" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
        <body name="link2" pos="0 0 0.2">
          <geom type="capsule" size="0.025" fromto="0 0 0 0 0 0.15" rgba="0.3 0.8 0.3 1"/>
          <joint name="elbow" type="hinge" axis="0 1 0" range="-2.0 2.0"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_pan_act" joint="shoulder_pan" kp="50"/>
    <position name="shoulder_lift_act" joint="shoulder_lift" kp="50"/>
    <position name="elbow_act" joint="elbow" kp="50"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def sim():
    """Create a fresh Simulation instance."""
    s = Simulation(tool_name="test_sim", mesh=False)
    yield s
    s.cleanup()


@pytest.fixture
def sim_with_world(sim):
    """Simulation with a world already created."""
    result = sim.create_world(gravity=[0, 0, -9.81])
    assert result["status"] == "success"
    return sim


@pytest.fixture
def robot_xml_path():
    """Write test robot XML to a temp file."""
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test_arm.xml")
    with open(path, "w") as f:
        f.write(ROBOT_XML)
    yield path
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sim_with_robot(sim_with_world, robot_xml_path):
    """Simulation with world + robot loaded."""
    result = sim_with_world.add_robot("arm1", urdf_path=robot_xml_path)
    assert result["status"] == "success"
    return sim_with_world


# ── World Management ──


class TestWorldLifecycle:
    """Test create_world → get_state → reset → destroy lifecycle."""

    def test_create_world_defaults(self, sim):
        result = sim.create_world()
        assert result["status"] == "success"
        assert "Simulation world created" in result["content"][0]["text"]
        assert sim._world is not None
        assert sim._world.gravity == [0.0, 0.0, -9.81]

    def test_create_world_custom_gravity(self, sim):
        result = sim.create_world(gravity=[0, 0, -5.0])
        assert result["status"] == "success"
        assert sim._world.gravity == [0.0, 0.0, -5.0]

    def test_create_world_scalar_gravity(self, sim):
        result = sim.create_world(gravity=-3.0)
        assert result["status"] == "success"
        assert sim._world.gravity == [0.0, 0.0, -3.0]

    def test_create_world_custom_timestep(self, sim):
        result = sim.create_world(timestep=0.001)
        assert result["status"] == "success"
        assert sim._world.timestep == 0.001

    def test_create_world_no_ground_plane(self, sim):
        result = sim.create_world(ground_plane=False)
        assert result["status"] == "success"

    def test_create_world_duplicate_fails(self, sim_with_world):
        result = sim_with_world.create_world()
        assert result["status"] == "error"
        assert "already exists" in result["content"][0]["text"]

    def test_get_state(self, sim_with_world):
        result = sim_with_world.get_state()
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "Simulation State" in text
        assert "t=" in text

    def test_reset(self, sim_with_world):
        # Step forward
        sim_with_world.step(n_steps=100)
        assert sim_with_world._world.sim_time > 0

        # Reset
        result = sim_with_world.reset()
        assert result["status"] == "success"
        assert sim_with_world._world.sim_time == 0.0
        assert sim_with_world._world.step_count == 0

    def test_destroy(self, sim_with_world):
        result = sim_with_world.destroy()
        assert result["status"] == "success"
        assert sim_with_world._world is None

    def test_destroy_no_world(self, sim):
        result = sim.destroy()
        assert result["status"] == "success"

    def test_step_advances_state(self, sim_with_world):
        result = sim_with_world.step(n_steps=50)
        assert result["status"] == "success"
        assert sim_with_world._world.step_count == 50
        assert sim_with_world._world.sim_time > 0

    def test_set_gravity(self, sim_with_world):
        result = sim_with_world.set_gravity([0, 0, -5.0])
        assert result["status"] == "success"
        assert sim_with_world._world.gravity == [0, 0, -5.0]

    def test_set_gravity_scalar(self, sim_with_world):
        result = sim_with_world.set_gravity(-3.0)
        assert result["status"] == "success"
        assert sim_with_world._world.gravity == [0.0, 0.0, -3.0]

    def test_set_timestep(self, sim_with_world):
        result = sim_with_world.set_timestep(0.001)
        assert result["status"] == "success"
        assert sim_with_world._world.timestep == 0.001

    def test_load_scene_from_file(self, sim, robot_xml_path):
        result = sim.load_scene(robot_xml_path)
        assert result["status"] == "success"
        assert "Scene loaded" in result["content"][0]["text"]
        assert sim._world._model.njnt > 0

    def test_load_scene_nonexistent(self, sim):
        result = sim.load_scene("/nonexistent/path.xml")
        assert result["status"] == "error"


# ── Object Management ──


class TestObjectManagement:
    """Test add_object → list_objects → move_object → remove_object."""

    def test_add_object_box(self, sim_with_world):
        result = sim_with_world.add_object("red_cube", shape="box", position=[0.3, 0, 0.1], color=[1, 0, 0, 1])
        assert result["status"] == "success"
        assert "red_cube" in sim_with_world._world.objects

    def test_add_object_sphere(self, sim_with_world):
        result = sim_with_world.add_object("ball", shape="sphere", mass=0.2)
        assert result["status"] == "success"

    def test_add_object_cylinder(self, sim_with_world):
        result = sim_with_world.add_object("can", shape="cylinder", is_static=True)
        assert result["status"] == "success"

    def test_add_duplicate_object_fails(self, sim_with_world):
        sim_with_world.add_object("obj1", shape="box")
        result = sim_with_world.add_object("obj1", shape="sphere")
        assert result["status"] == "error"
        assert "exists" in result["content"][0]["text"]

    def test_add_object_no_world(self, sim):
        result = sim.add_object("obj", shape="box")
        assert result["status"] == "error"

    def test_list_objects_empty(self, sim_with_world):
        result = sim_with_world.list_objects()
        assert result["status"] == "success"
        assert "No objects" in result["content"][0]["text"]

    def test_list_objects_populated(self, sim_with_world):
        sim_with_world.add_object("a", shape="box")
        sim_with_world.add_object("b", shape="sphere")
        result = sim_with_world.list_objects()
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "a" in text
        assert "b" in text

    def test_move_object(self, sim_with_world):
        sim_with_world.add_object("cube", shape="box", position=[0, 0, 0.1])
        result = sim_with_world.move_object("cube", position=[1.0, 0, 0.1])
        assert result["status"] == "success"
        assert sim_with_world._world.objects["cube"].position == [1.0, 0, 0.1]

    def test_move_nonexistent_object(self, sim_with_world):
        result = sim_with_world.move_object("ghost", position=[0, 0, 0])
        assert result["status"] == "error"

    def test_remove_object(self, sim_with_world):
        sim_with_world.add_object("tmp", shape="box")
        assert "tmp" in sim_with_world._world.objects
        result = sim_with_world.remove_object("tmp")
        assert result["status"] == "success"
        assert "tmp" not in sim_with_world._world.objects

    def test_remove_nonexistent_object(self, sim_with_world):
        result = sim_with_world.remove_object("ghost")
        assert result["status"] == "error"


# ── Robot Management ──


class TestRobotManagement:
    """Test add_robot → list_robots → get_robot_state → remove_robot."""

    def test_add_robot(self, sim_with_world, robot_xml_path):
        result = sim_with_world.add_robot("arm1", urdf_path=robot_xml_path)
        assert result["status"] == "success"
        assert "arm1" in sim_with_world._world.robots
        robot = sim_with_world._world.robots["arm1"]
        assert len(robot.joint_names) == 3
        assert len(robot.actuator_ids) > 0

    def test_add_robot_no_world(self, sim, robot_xml_path):
        result = sim.add_robot("arm1", urdf_path=robot_xml_path)
        assert result["status"] == "error"

    def test_add_duplicate_robot(self, sim_with_robot, robot_xml_path):
        result = sim_with_robot.add_robot("arm1", urdf_path=robot_xml_path)
        assert result["status"] == "error"

    def test_add_robot_nonexistent_file(self, sim_with_world):
        result = sim_with_world.add_robot("arm", urdf_path="/nonexistent.xml")
        assert result["status"] == "error"

    def test_add_robot_no_path(self, sim_with_world):
        # Neither urdf_path nor data_config, and name doesn't resolve
        result = sim_with_world.add_robot("nonexistent_model_xyz")
        assert result["status"] == "error"

    def test_list_robots_empty(self, sim_with_world):
        result = sim_with_world.list_robots()
        assert result["status"] == "success"
        assert "No robots" in result["content"][0]["text"]

    def test_list_robots_populated(self, sim_with_robot):
        result = sim_with_robot.list_robots()
        assert result["status"] == "success"
        assert "arm1" in result["content"][0]["text"]

    def test_get_robot_state(self, sim_with_robot):
        result = sim_with_robot.get_robot_state("arm1")
        assert result["status"] == "success"
        # Should contain joint position data
        text = result["content"][0]["text"]
        assert "shoulder_pan" in text

    def test_get_robot_state_invalid(self, sim_with_robot):
        result = sim_with_robot.get_robot_state("nonexistent")
        assert result["status"] == "error"

    def test_remove_robot(self, sim_with_robot):
        result = sim_with_robot.remove_robot("arm1")
        assert result["status"] == "success"
        assert "arm1" not in sim_with_robot._world.robots

    def test_remove_nonexistent_robot(self, sim_with_world):
        result = sim_with_world.remove_robot("ghost")
        assert result["status"] == "error"

    def test_robot_compatible_observation(self, sim_with_robot):
        """Robot ABC compatible get_observation should return joint data."""
        obs = sim_with_robot.get_observation(robot_name="arm1")
        assert isinstance(obs, dict)
        # Should have joint positions
        assert len(obs) > 0

    def test_robot_compatible_send_action(self, sim_with_robot):
        """Robot ABC compatible send_action should not crash."""
        sim_with_robot.send_action(
            {"shoulder_pan_act": 0.5, "shoulder_lift_act": 0.1, "elbow_act": -0.2},
            robot_name="arm1",
        )
        # Verify physics advanced
        assert sim_with_robot._world.sim_time > 0


# ── Camera Management ──


class TestCameraManagement:
    def test_add_camera(self, sim_with_world):
        result = sim_with_world.add_camera("overhead", position=[0, 0, 3], target=[0, 0, 0])
        assert result["status"] == "success"
        assert "overhead" in sim_with_world._world.cameras

    def test_add_camera_no_world(self, sim):
        result = sim.add_camera("cam")
        assert result["status"] == "error"

    def test_remove_camera(self, sim_with_world):
        sim_with_world.add_camera("tmp_cam")
        result = sim_with_world.remove_camera("tmp_cam")
        assert result["status"] == "success"
        assert "tmp_cam" not in sim_with_world._world.cameras

    def test_remove_nonexistent_camera(self, sim_with_world):
        result = sim_with_world.remove_camera("ghost")
        assert result["status"] == "error"


# ── Scene Injection (XML round-trip) ──


class TestSceneInjection:
    """Test that objects/cameras injected into a robot scene persist."""

    def test_add_object_to_robot_scene(self, sim_with_robot):
        """Adding an object to a scene with robots uses XML injection."""
        old_nbody = sim_with_robot._world._model.nbody
        result = sim_with_robot.add_object("cube", shape="box", position=[0.3, 0, 0.05])
        assert result["status"] == "success"
        # The model should have more bodies after injection
        assert sim_with_robot._world._model.nbody > old_nbody

    def test_remove_object_from_robot_scene(self, sim_with_robot):
        sim_with_robot.add_object("cube", shape="box", position=[0.3, 0, 0.05])
        nbody_with_cube = sim_with_robot._world._model.nbody
        sim_with_robot.remove_object("cube")
        # After ejection, body count should decrease
        assert sim_with_robot._world._model.nbody < nbody_with_cube

    def test_add_camera_to_robot_scene(self, sim_with_robot):
        """Cameras injected into robot scene via XML round-trip."""
        result = sim_with_robot.add_camera("top", position=[0, 0, 2])
        assert result["status"] == "success"
        assert "top" in sim_with_robot._world.cameras

    def test_robot_joints_survive_object_injection(self, sim_with_robot):
        """Verify robot joint IDs are re-discovered after scene recompile."""
        robot = sim_with_robot._world.robots["arm1"]
        original_joints = list(robot.joint_names)

        sim_with_robot.add_object("box1", shape="box", position=[0.5, 0, 0.1])

        # Joints should still be valid
        assert robot.joint_names == original_joints
        assert len(robot.joint_ids) == len(original_joints)
        assert len(robot.actuator_ids) > 0


# ── Rendering ──


@requires_gl
class TestRendering:
    def test_render_default_camera(self, sim_with_world):
        result = sim_with_world.render(camera_name="default")
        assert result["status"] == "success"
        assert any("image" in c for c in result["content"])

    def test_render_custom_size(self, sim_with_world):
        result = sim_with_world.render(width=320, height=240)
        assert result["status"] == "success"

    def test_render_depth(self, sim_with_world):
        result = sim_with_world.render_depth()
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "Depth" in text

    def test_render_no_world(self, sim):
        result = sim.render()
        assert result["status"] == "error"

    def test_get_contacts(self, sim_with_world):
        # Add an object that will contact the ground
        sim_with_world.add_object("ball", shape="sphere", position=[0, 0, 0.5])
        sim_with_world.step(n_steps=500)
        result = sim_with_world.get_contacts()
        assert result["status"] == "success"


# ── Randomization ──


class TestRandomization:
    def test_randomize_colors(self, sim_with_world):
        sim_with_world.add_object("cube", shape="box")
        result = sim_with_world.randomize(randomize_colors=True, seed=42)
        assert result["status"] == "success"
        assert "Colors" in result["content"][0]["text"]

    def test_randomize_lighting(self, sim_with_world):
        result = sim_with_world.randomize(randomize_lighting=True, seed=42)
        assert result["status"] == "success"

    def test_randomize_physics(self, sim_with_world):
        sim_with_world.add_object("cube", shape="box")
        result = sim_with_world.randomize(randomize_physics=True, seed=42)
        assert result["status"] == "success"
        assert "Physics" in result["content"][0]["text"]

    def test_randomize_positions(self, sim_with_world):
        sim_with_world.add_object("cube", shape="box", position=[0, 0, 0.1])
        result = sim_with_world.randomize(randomize_positions=True, seed=42)
        assert result["status"] == "success"

    def test_randomize_no_world(self, sim):
        result = sim.randomize()
        assert result["status"] == "error"


# ── Introspection ──


class TestIntrospection:
    def test_get_features_with_robot(self, sim_with_robot):
        result = sim_with_robot.get_features()
        assert result["status"] == "success"
        data = json.loads(result["content"][1]["text"])
        features = data["features"]
        assert features["n_joints"] > 0
        assert features["n_actuators"] > 0
        assert "arm1" in features["robots"]

    def test_get_features_no_world(self, sim):
        result = sim.get_features()
        assert result["status"] == "error"


# ── URDF Registry ──


class TestURDFRegistry:
    def test_list_urdfs(self, sim):
        result = sim.list_urdfs_action()
        assert result["status"] == "success"

    def test_register_urdf(self, sim, robot_xml_path):
        result = sim.register_urdf_action("test_arm", robot_xml_path)
        assert result["status"] == "success"
        assert "test_arm" in result["content"][0]["text"]


# ── Policy Execution ──


class TestPolicyExecution:
    """Test run_policy and eval_policy through the Simulation class."""

    def test_run_policy_mock(self, sim_with_robot):
        result = sim_with_robot.run_policy(
            "arm1",
            policy_provider="mock",
            instruction="wave",
            duration=0.1,
            fast_mode=True,
        )
        assert result["status"] == "success"
        assert "Policy complete" in result["content"][0]["text"]
        assert sim_with_robot._world.sim_time > 0

    def test_run_policy_no_world(self, sim):
        result = sim.run_policy("arm1", policy_provider="mock")
        assert result["status"] == "error"

    def test_run_policy_invalid_robot(self, sim_with_world):
        result = sim_with_world.run_policy("nonexistent", policy_provider="mock")
        assert result["status"] == "error"

    def test_eval_policy_mock(self, sim_with_robot):
        result = sim_with_robot.eval_policy(
            robot_name="arm1",
            policy_provider="mock",
            instruction="reach",
            n_episodes=2,
            max_steps=10,
        )
        assert result["status"] == "success"
        # eval_policy returns json in the second content item
        json_content = result["content"][1]
        data = json_content.get("json") or json.loads(json_content.get("text", "{}"))
        assert data["n_episodes"] == 2
        assert "success_rate" in data

    def test_eval_policy_no_world(self, sim):
        result = sim.eval_policy()
        assert result["status"] == "error"

    def test_start_policy_and_stop(self, sim_with_robot):
        result = sim_with_robot.start_policy(
            "arm1",
            policy_provider="mock",
            duration=0.2,
            fast_mode=True,
        )
        assert result["status"] == "success"
        assert "started" in result["content"][0]["text"]

        # Stop it
        result = sim_with_robot._stop_policy("arm1")
        assert result["status"] == "success"

    def test_start_policy_no_world(self, sim):
        result = sim.start_policy("arm1")
        assert result["status"] == "error"

    def test_start_policy_invalid_robot(self, sim_with_world):
        result = sim_with_world.start_policy("ghost")
        assert result["status"] == "error"


# ── Action Dispatch ──


class TestActionDispatch:
    """Test _dispatch_action routes correctly via tool_spec actions."""

    def test_dispatch_create_world(self, sim):
        result = sim._dispatch_action("create_world", {"action": "create_world"})
        assert result["status"] == "success"

    def test_dispatch_get_state(self, sim_with_world):
        result = sim_with_world._dispatch_action("get_state", {"action": "get_state"})
        assert result["status"] == "success"

    def test_dispatch_step(self, sim_with_world):
        result = sim_with_world._dispatch_action("step", {"action": "step", "n_steps": 10})
        assert result["status"] == "success"

    def test_dispatch_add_object(self, sim_with_world):
        result = sim_with_world._dispatch_action(
            "add_object",
            {"action": "add_object", "name": "box1", "shape": "box", "position": [0, 0, 0.1]},
        )
        assert result["status"] == "success"

    def test_dispatch_unknown_action(self, sim):
        result = sim._dispatch_action("nonexistent", {"action": "nonexistent"})
        assert result["status"] == "error"
        assert "Unknown action" in result["content"][0]["text"]

    def test_dispatch_private_action_blocked(self, sim):
        """Actions starting with _ are blocked (security)."""
        result = sim._dispatch_action("_compile_world", {"action": "_compile_world"})
        assert result["status"] == "error"

    def test_dispatch_list_urdfs_alias(self, sim):
        result = sim._dispatch_action("list_urdfs", {"action": "list_urdfs"})
        assert result["status"] == "success"

    def test_dispatch_set_gravity(self, sim_with_world):
        result = sim_with_world._dispatch_action("set_gravity", {"action": "set_gravity", "gravity": [0, 0, -5.0]})
        assert result["status"] == "success"


# ── Context Manager ──


class TestContextManager:
    def test_context_manager_cleanup(self):
        with Simulation(tool_name="ctx_test", mesh=False) as sim:
            sim.create_world()
            assert sim._world is not None
        # After exit, world should be cleaned up
        assert sim._world is None


# ── Tool Spec ──


class TestToolSpec:
    def test_tool_name(self, sim):
        assert sim.tool_name == "test_sim"

    def test_tool_type(self, sim):
        assert sim.tool_type == "simulation"

    def test_tool_spec_schema(self, sim):
        spec = sim.tool_spec
        assert spec["name"] == "test_sim"
        assert "inputSchema" in spec
        assert "json" in spec["inputSchema"]
        schema = spec["inputSchema"]["json"]
        assert "properties" in schema
        assert "action" in schema["properties"]


# ── Viewer (headless safe) ──


class TestViewer:
    def test_open_viewer_no_world(self, sim):
        result = sim.open_viewer()
        assert result["status"] == "error"

    def test_close_viewer_noop(self, sim):
        result = sim.close_viewer()
        assert result["status"] == "success"


# ── Error Paths ──


class TestErrorPaths:
    """Test that error conditions return proper error dicts, not exceptions."""

    def test_get_state_no_world(self, sim):
        result = sim.get_state()
        assert result["status"] == "error"

    def test_step_no_world(self, sim):
        result = sim.step()
        assert result["status"] == "error"

    def test_reset_no_world(self, sim):
        result = sim.reset()
        assert result["status"] == "error"

    def test_add_object_no_world(self, sim):
        result = sim.add_object("x", shape="box")
        assert result["status"] == "error"

    def test_move_object_no_world(self, sim):
        result = sim.move_object("x", position=[0, 0, 0])
        assert result["status"] == "error"

    def test_list_objects_no_world(self, sim):
        result = sim.list_objects()
        assert result["status"] == "error"

    def test_list_robots_no_world(self, sim):
        result = sim.list_robots()
        assert result["status"] == "error"

    def test_render_no_world(self, sim):
        result = sim.render()
        assert result["status"] == "error"

    def test_render_depth_no_world(self, sim):
        result = sim.render_depth()
        assert result["status"] == "error"

    def test_get_contacts_no_world(self, sim):
        result = sim.get_contacts()
        assert result["status"] == "error"

    def test_get_features_no_world(self, sim):
        result = sim.get_features()
        assert result["status"] == "error"

    def test_set_gravity_no_world(self, sim):
        result = sim.set_gravity([0, 0, -5])
        assert result["status"] == "error"

    def test_set_timestep_no_world(self, sim):
        result = sim.set_timestep(0.001)
        assert result["status"] == "error"

    def test_get_robot_state_no_world(self, sim):
        result = sim.get_robot_state("x")
        assert result["status"] == "error"

    def test_randomize_no_world(self, sim):
        result = sim.randomize()
        assert result["status"] == "error"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
