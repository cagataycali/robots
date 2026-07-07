"""Discovery and state-introspection parity for the Newton backend.

Exercises the methods that bring NewtonSimEngine to MuJoCo parity for scene
discovery and per-joint state queries: get_robot_state, list_robots_info,
list_bodies, list_objects, move_object, get_features, list_urdfs,
register_urdf, plus the discovery surface in describe(). Gated on Newton +
Warp being importable (real GPU model build).
"""

from __future__ import annotations

import importlib.util

import pytest

_HAS_NEWTON = importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None

pytestmark = pytest.mark.skipif(not _HAS_NEWTON, reason="newton/warp not installed")


@pytest.fixture
def engine():
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    sim = NewtonSimEngine(solver="mujoco")
    sim.create_world()
    yield sim
    sim.destroy()


@pytest.fixture
def engine_so100(engine):
    engine.add_robot("so100")
    return engine


class TestGetRobotState:
    def test_returns_position_and_velocity_per_joint(self, engine_so100):
        result = engine_so100.get_robot_state()
        assert result["status"] == "success"
        state = result["content"][1]["json"]["state"]
        # All six arm joints present with the position/velocity contract.
        assert set(state) == set(engine_so100.robot_joint_names("so100"))
        for vals in state.values():
            assert set(vals) == {"position", "velocity"}
            assert isinstance(vals["position"], float)
            assert isinstance(vals["velocity"], float)

    def test_velocity_is_nonzero_after_actuation(self, engine_so100):
        # Drive a joint hard, then confirm a non-zero velocity is reported -
        # i.e. velocities are read from joint_qd, not left at zero.
        engine_so100.send_action({"Rotation": 0.8}, robot_name="so100", n_substeps=5)
        state = engine_so100.get_robot_state()["content"][1]["json"]["state"]
        assert abs(state["Rotation"]["velocity"]) > 1e-3

    def test_no_world_errors(self):
        from strands_robots.simulation.newton.simulation import NewtonSimEngine

        sim = NewtonSimEngine(solver="mujoco")
        assert sim.get_robot_state()["status"] == "error"

    def test_unknown_robot_errors(self, engine_so100):
        assert engine_so100.get_robot_state("ghost")["status"] == "error"


# A minimal floating-base model: a free-jointed root body carrying two hinge
# children. The root's free joint spans 7 coordinates (xyz + quaternion) and 6
# DOFs, so ``j1``/``j2`` live at coordinate indices 7/8 and DOF indices 6/7 -
# NOT the ordinal 1/2 a naive per-joint offset would assign. Kept inline (no
# downloaded asset) so the test runs anywhere Newton is importable.
_FLOATER_MJCF = """<mujoco model="floater">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <freejoint name="root"/>
      <geom type="box" size="0.1 0.1 0.05" mass="1"/>
      <body name="link1" pos="0.1 0 0">
        <joint name="j1" type="hinge" axis="0 0 1"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.2"/>
        <body name="link2" pos="0.2 0 0">
          <joint name="j2" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.2"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>"""


@pytest.fixture
def engine_floater(engine, tmp_path):
    """Real Newton build of the inline floating-base model above."""
    path = tmp_path / "floater.xml"
    path.write_text(_FLOATER_MJCF)
    assert engine.add_robot("floater", urdf_path=str(path))["status"] == "success"
    return engine


class TestFloatingBaseJointIndices:
    """A free-jointed root must not shift every child joint's state reading.

    Regression for the Newton index maps: ``_joint_coord_index`` /
    ``_joint_dof_index`` were built from a per-joint ordinal offset (one
    coordinate and one DOF per joint), which is wrong once a robot has a
    multi-coordinate joint. A floating base (free joint: 7 coordinates, 6 DOFs)
    made every child joint read the base's coordinates instead of its own, so
    get_robot_state / get_observation reported garbage for a humanoid's leg and
    arm joints (and the policy observation path saw the same garbage).
    """

    def test_index_maps_use_authoritative_joint_starts(self, engine_floater):
        model = engine_floater._model
        q_start = model.joint_q_start.numpy()
        qd_start = model.joint_qd_start.numpy()
        names = engine_floater._world.robots["floater"].joint_names
        assert names == ["root", "j1", "j2"]
        for i, jname in enumerate(names):
            assert engine_floater._joint_coord_index[("floater", jname)] == int(q_start[i])
            assert engine_floater._joint_dof_index[("floater", jname)] == int(qd_start[i])
        # The hinges live past the free joint's 7 coordinates / 6 DOFs, not at
        # the ordinal 1/2 that the buggy mapping produced.
        assert engine_floater._joint_coord_index[("floater", "j1")] == 7
        assert engine_floater._joint_dof_index[("floater", "j1")] == 6
        assert engine_floater._joint_coord_index[("floater", "j2")] == 8
        assert engine_floater._joint_dof_index[("floater", "j2")] == 7

    def _set_position_sentinels(self, engine_floater):
        """Write joint_q[i] = 100 + i so each position value encodes its coord index."""
        q = engine_floater._state_0.joint_q.numpy().copy()
        for i in range(len(q)):
            q[i] = 100.0 + i
        engine_floater._state_0.joint_q.assign(q)
        return engine_floater._model.joint_q_start.numpy()

    def test_get_robot_state_reads_hinge_coordinates_past_free_joint(self, engine_floater):
        q_start = self._set_position_sentinels(engine_floater)
        state = engine_floater.get_robot_state("floater")["content"][1]["json"]["state"]
        # Each hinge must report the sentinel at ITS own coordinate index, not
        # the base coordinate a per-joint offset would read.
        assert state["j1"]["position"] == pytest.approx(100.0 + int(q_start[1]))
        assert state["j2"]["position"] == pytest.approx(100.0 + int(q_start[2]))

    def test_get_observation_reads_hinge_coordinates_past_free_joint(self, engine_floater):
        q_start = self._set_position_sentinels(engine_floater)
        obs = engine_floater.get_observation("floater", skip_images=True)
        assert obs["j1"] == pytest.approx(100.0 + int(q_start[1]))
        assert obs["j2"] == pytest.approx(100.0 + int(q_start[2]))

    def test_fixed_base_arm_index_maps_are_the_identity(self, engine_so100):
        # A robot with no multi-coordinate joint (an all-revolute arm) must be
        # unchanged: coordinate/DOF index == ordinal position.
        names = engine_so100._world.robots["so100"].joint_names
        coords = [engine_so100._joint_coord_index[("so100", j)] for j in names]
        dofs = [engine_so100._joint_dof_index[("so100", j)] for j in names]
        assert coords == list(range(len(names)))
        assert dofs == list(range(len(names)))


class TestListBodies:
    def test_scoped_lists_only_robot_bodies_with_gripper(self, engine_so100):
        result = engine_so100.list_bodies("so100")
        assert result["status"] == "success"
        payload = result["content"][1]["json"]
        assert payload["bodies"]
        assert all(b.startswith("so_arm100") for b in payload["bodies"])
        # Gripper auto-detection resolves a jaw/gripper mount body.
        assert payload["gripper_body"] is not None
        assert "jaw" in payload["gripper_body"].lower()

    def test_global_includes_object_bodies(self, engine_so100):
        engine_so100.add_object("cube", shape="box", position=[0.3, 0.0, 0.05], mass=0.2)
        scoped = engine_so100.list_bodies("so100")["content"][1]["json"]["bodies"]
        every = engine_so100.list_bodies()["content"][1]["json"]["bodies"]
        # The free-floating cube body is in the global list but not the robot scope.
        assert len(every) > len(scoped)

    def test_unknown_robot_errors(self, engine_so100):
        assert engine_so100.list_bodies("ghost")["status"] == "error"


class TestObjectsListingAndMove:
    def test_list_objects_reports_shape_and_pose(self, engine_so100):
        engine_so100.add_object("cube", shape="box", position=[0.3, 0.0, 0.05], mass=0.2)
        text = engine_so100.list_objects()["content"][0]["text"]
        assert "cube" in text and "box" in text

    def test_list_objects_empty(self, engine_so100):
        assert "No objects" in engine_so100.list_objects()["content"][0]["text"]

    def test_move_object_updates_pose_and_rebuilds(self, engine_so100):
        engine_so100.add_object("cube", shape="box", position=[0.3, 0.0, 0.05], mass=0.2)
        result = engine_so100.move_object("cube", position=[0.1, 0.1, 0.05])
        assert result["status"] == "success"
        assert engine_so100._world.objects["cube"].position == [0.1, 0.1, 0.05]
        # Model still steppable after the rebuild triggered by the move.
        assert engine_so100.step(1)["status"] == "success"

    def test_move_unknown_object_errors(self, engine_so100):
        assert engine_so100.move_object("ghost", position=[0, 0, 0])["status"] == "error"


class TestRobotsInfoAndFeatures:
    def test_list_robots_info_names_asset(self, engine_so100):
        text = engine_so100.list_robots_info()["content"][0]["text"]
        assert "so100" in text and "so_arm100.xml" in text

    def test_list_robots_info_empty(self, engine):
        assert "No robots" in engine.list_robots_info()["content"][0]["text"]

    def test_get_features_schema(self, engine_so100):
        features = engine_so100.get_features("so100")["content"][1]["json"]["features"]
        for key in ("n_bodies", "n_joints", "n_dofs", "timestep", "joint_names", "robots"):
            assert key in features
        assert features["robots"]["so100"]["n_joints"] == 6
        assert "so100" in features["robots"]

    def test_get_features_unknown_robot_errors(self, engine_so100):
        assert engine_so100.get_features("ghost")["status"] == "error"


class TestRegistryPassthrough:
    def test_list_urdfs_returns_registry_table(self, engine):
        text = engine.list_urdfs()["content"][0]["text"]
        assert "Category" in text

    def test_register_urdf_missing_file_errors(self, engine):
        assert engine.register_urdf("xx", "/no/such/path.xml")["status"] == "error"

    def test_register_urdf_empty_path_errors(self, engine):
        assert engine.register_urdf("xx", "")["status"] == "error"


class TestDescribeSurface:
    def test_describe_exposes_new_methods(self, engine_so100):
        described = engine_so100.describe()
        methods = described["methods"]
        for name in (
            "get_robot_state",
            "list_bodies",
            "move_object",
            "get_features",
            "list_urdfs",
            "add_object",
            "remove_object",
        ):
            assert name in methods
        # add_object advertises its real distinguishing parameter so a
        # caller can place a manipulable object without reading source.
        assert "shape" in methods["add_object"]
        assert described["cameras"] == ["default"]
        assert described["bodies"]
        assert described["world_created"] is True
