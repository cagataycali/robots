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
        for name in ("get_robot_state", "list_bodies", "move_object", "get_features", "list_urdfs"):
            assert name in methods
        assert described["cameras"] == ["default"]
        assert described["bodies"]
        assert described["world_created"] is True
