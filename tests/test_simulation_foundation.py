"""Tests for simulation foundation — models, ABC, factory, model_registry.

These tests verify the lightweight simulation abstractions without
requiring MuJoCo or any heavy dependencies.
"""

import pytest

from strands_robots.simulation.base import SimulationBackend
from strands_robots.simulation.factory import (
    create_simulation,
    list_backends,
    register_backend,
)
from strands_robots.simulation.models import (
    SimCamera,
    SimObject,
    SimRobot,
    SimStatus,
    SimWorld,
    TrajectoryStep,
)

# ── Dataclass Tests ──────────────────────────────────────────────


class TestSimModels:
    """Test simulation dataclass construction and defaults."""

    def test_sim_robot_defaults(self):
        robot = SimRobot(name="test", urdf_path="/fake/path.urdf")
        assert robot.name == "test"
        assert robot.position == [0.0, 0.0, 0.0]
        assert robot.orientation == [1.0, 0.0, 0.0, 0.0]
        assert robot.joint_ids == []
        assert robot.joint_names == []
        assert robot.actuator_ids == []
        assert robot.body_id == -1
        assert robot.policy_running is False

    def test_sim_robot_custom_position(self):
        robot = SimRobot(name="arm", urdf_path="/p", position=[1.0, 2.0, 3.0])
        assert robot.position == [1.0, 2.0, 3.0]

    def test_sim_object_defaults(self):
        obj = SimObject(name="cube", shape="box")
        assert obj.name == "cube"
        assert obj.shape == "box"
        assert obj.size == [0.05, 0.05, 0.05]
        assert obj.color == [0.5, 0.5, 0.5, 1.0]
        assert obj.mass == 0.1
        assert obj.is_static is False
        assert obj.mesh_path is None

    def test_sim_object_preserves_originals(self):
        obj = SimObject(name="ball", shape="sphere", position=[1, 2, 3], color=[1, 0, 0, 1])
        assert obj._original_position == [1, 2, 3]
        assert obj._original_color == [1, 0, 0, 1]

    def test_sim_camera_defaults(self):
        cam = SimCamera(name="default")
        assert cam.fov == 60.0
        assert cam.width == 640
        assert cam.height == 480
        assert cam.camera_id == -1

    def test_sim_world_defaults(self):
        world = SimWorld()
        assert world.timestep == 0.002
        assert world.gravity == [0.0, 0.0, -9.81]
        assert world.ground_plane is True
        assert world.status == SimStatus.IDLE
        assert world.sim_time == 0.0
        assert world.step_count == 0
        assert world.robots == {}
        assert world.objects == {}
        assert world.cameras == {}

    def test_sim_status_enum(self):
        assert SimStatus.IDLE.value == "idle"
        assert SimStatus.RUNNING.value == "running"
        assert SimStatus.PAUSED.value == "paused"
        assert SimStatus.COMPLETED.value == "completed"
        assert SimStatus.ERROR.value == "error"

    def test_trajectory_step(self):
        step = TrajectoryStep(
            timestamp=1.0,
            sim_time=0.5,
            robot_name="arm",
            observation={"state": [1, 2, 3]},
            action={"joint_0": 0.5},
            instruction="pick up cube",
        )
        assert step.robot_name == "arm"
        assert step.instruction == "pick up cube"

    def test_trajectory_step_default_instruction(self):
        step = TrajectoryStep(timestamp=0.0, sim_time=0.0, robot_name="r", observation={}, action={})
        assert step.instruction == ""

    def test_sim_world_add_robot(self):
        world = SimWorld()
        robot = SimRobot(name="so100", urdf_path="/p")
        world.robots["so100"] = robot
        assert "so100" in world.robots
        assert world.robots["so100"].name == "so100"


# ── ABC Tests ────────────────────────────────────────────────────


class TestSimulationBackend:
    """Test the abstract base class."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            SimulationBackend()

    def test_has_required_abstract_methods(self):
        abstract_methods = SimulationBackend.__abstractmethods__
        expected = {
            "create_world",
            "destroy",
            "reset",
            "step",
            "get_state",
            "add_robot",
            "remove_robot",
            "add_object",
            "remove_object",
            "get_observation",
            "send_action",
            "render",
        }
        assert expected == abstract_methods

    def test_default_optional_methods(self):
        """Optional methods raise NotImplementedError."""

        # Create a minimal concrete subclass
        class Dummy(SimulationBackend):
            def create_world(self, **kw):
                return {}

            def destroy(self):
                return {}

            def reset(self):
                return {}

            def step(self, n_steps=1):
                return {}

            def get_state(self):
                return {}

            def add_robot(self, name, **kw):
                return {}

            def remove_robot(self, name):
                return {}

            def add_object(self, name, **kw):
                return {}

            def remove_object(self, name):
                return {}

            def get_observation(self, **kw):
                return {}

            def send_action(self, action, **kw):
                return None

            def render(self, **kw):
                return {}

        d = Dummy()
        # Optional methods should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            d.load_scene("x")
        with pytest.raises(NotImplementedError):
            d.run_policy("x")
        with pytest.raises(NotImplementedError):
            d.randomize()
        with pytest.raises(NotImplementedError):
            d.get_contacts()

    def test_context_manager(self):
        """ABC supports context manager protocol."""

        class Dummy(SimulationBackend):
            cleaned = False

            def create_world(self, **kw):
                return {}

            def destroy(self):
                return {}

            def reset(self):
                return {}

            def step(self, n_steps=1):
                return {}

            def get_state(self):
                return {}

            def add_robot(self, name, **kw):
                return {}

            def remove_robot(self, name):
                return {}

            def add_object(self, name, **kw):
                return {}

            def remove_object(self, name):
                return {}

            def get_observation(self, **kw):
                return {}

            def send_action(self, action, **kw):
                return None

            def render(self, **kw):
                return {}

            def cleanup(self):
                Dummy.cleaned = True

        with Dummy() as _d:
            pass
        assert Dummy.cleaned is True


# ── Factory Tests ────────────────────────────────────────────────


class TestSimulationFactory:
    """Test backend registration and creation."""

    def test_list_backends_includes_mujoco(self):
        backends = list_backends()
        assert "mujoco" in backends

    def test_list_backends_returns_list(self):
        assert isinstance(list_backends(), list)

    def test_register_custom_backend(self):
        """Can register a custom backend class and create an instance."""

        class FakeBackend(SimulationBackend):
            def create_world(self, **kw):
                return {}

            def destroy(self):
                return {}

            def reset(self):
                return {}

            def step(self, n_steps=1):
                return {}

            def get_state(self):
                return {}

            def add_robot(self, name, **kw):
                return {}

            def remove_robot(self, name):
                return {}

            def add_object(self, name, **kw):
                return {}

            def remove_object(self, name):
                return {}

            def get_observation(self, **kw):
                return {}

            def send_action(self, action, **kw):
                return None

            def render(self, **kw):
                return {}

        register_backend("fake_test", lambda: FakeBackend, force=True)
        assert "fake_test" in list_backends()
        sim = create_simulation("fake_test")
        assert isinstance(sim, FakeBackend)

    def test_register_backend_rejects_duplicate(self):
        """Registering an existing name without force raises ValueError."""

        class Dummy(SimulationBackend):
            def create_world(self, **kw): return {}
            def destroy(self): return {}
            def reset(self): return {}
            def step(self, n_steps=1): return {}
            def get_state(self): return {}
            def add_robot(self, name, **kw): return {}
            def remove_robot(self, name): return {}
            def add_object(self, name, **kw): return {}
            def remove_object(self, name): return {}
            def get_observation(self, **kw): return {}
            def send_action(self, action, **kw): return None
            def render(self, **kw): return {}

        register_backend("dup_test", lambda: Dummy, force=True)
        with pytest.raises(ValueError, match="already registered"):
            register_backend("dup_test", lambda: Dummy)

    def test_register_backend_rejects_builtin_alias(self):
        """Registering an alias that conflicts with built-in aliases raises."""

        class Dummy(SimulationBackend):
            def create_world(self, **kw): return {}
            def destroy(self): return {}
            def reset(self): return {}
            def step(self, n_steps=1): return {}
            def get_state(self): return {}
            def add_robot(self, name, **kw): return {}
            def remove_robot(self, name): return {}
            def add_object(self, name, **kw): return {}
            def remove_object(self, name): return {}
            def get_observation(self, **kw): return {}
            def send_action(self, action, **kw): return None
            def render(self, **kw): return {}

        with pytest.raises(ValueError, match="conflicts with built-in"):
            register_backend("custom_phys", lambda: Dummy, aliases=["mj"])


# ── Model Registry Tests ─────────────────────────────────────────


class TestModelRegistry:
    """Test URDF/MJCF model resolution."""

    def test_list_available_models(self):
        from strands_robots.simulation.model_registry import list_available_models

        models = list_available_models()
        assert isinstance(models, str)
        # Should contain robot names in the formatted table
        assert "so100" in models
        assert len(models) > 100

    def test_resolve_known_model(self):
        from strands_robots.simulation.model_registry import resolve_model

        # resolve_model should return a path or None for known robots
        result = resolve_model("so100")
        # It may return None if robot_descriptions doesn't have it,
        # but it shouldn't raise
        assert result is None or isinstance(result, str)

    def test_register_and_resolve_urdf(self, tmp_path):
        from strands_robots.simulation.model_registry import register_urdf, resolve_urdf

        # Create a real temp file so resolve_urdf can find it
        urdf_file = tmp_path / "robot.urdf"
        urdf_file.write_text("<robot/>")
        register_urdf("test_robot_xyz", str(urdf_file))
        result = resolve_urdf("test_robot_xyz")
        assert result == str(urdf_file)

    def test_resolve_unknown_returns_none(self):
        from strands_robots.simulation.model_registry import resolve_urdf

        result = resolve_urdf("nonexistent_robot_12345")
        assert result is None

    def test_list_registered_urdfs(self):
        from strands_robots.simulation.model_registry import list_registered_urdfs, register_urdf

        register_urdf("list_test_bot", "/fake/list.urdf")
        urdfs = list_registered_urdfs()
        assert isinstance(urdfs, dict)
        assert "list_test_bot" in urdfs
