"""Tests for simulation foundation — models, ABC, factory, model_registry.

These tests verify the lightweight simulation abstractions without
requiring MuJoCo or any heavy dependencies.
"""

import pytest

from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.factory import (
    create_simulation,
    list_backends,
    register_backend,
)
from strands_robots.simulation.models import (
    SimObject,
    SimRobot,
    SimStatus,
    SimWorld,
    TrajectoryStep,
)


# ── ABC Tests ────────────────────────────────────────────────────


class TestSimEngine:
    """Test the abstract base class contract."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            SimEngine()

    def test_has_required_abstract_methods(self):
        abstract_methods = SimEngine.__abstractmethods__
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

    def test_optional_methods_raise_not_implemented(self):
        """Optional methods on a concrete subclass raise NotImplementedError."""

        class Dummy(SimEngine):
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
        with pytest.raises(NotImplementedError):
            d.load_scene("x")
        with pytest.raises(NotImplementedError):
            d.run_policy("x")
        with pytest.raises(NotImplementedError):
            d.randomize()
        with pytest.raises(NotImplementedError):
            d.get_contacts()

    def test_context_manager_calls_cleanup(self):
        """ABC supports context manager protocol and calls cleanup on exit."""

        class Dummy(SimEngine):
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
    """Test backend registration and creation — full round-trip."""

    def test_list_backends_includes_mujoco(self):
        backends = list_backends()
        assert "mujoco" in backends

    def test_register_create_and_use_backend(self):
        """Register a custom backend, create it via factory, verify instance."""

        class FakeBackend(SimEngine):
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

    def test_register_rejects_duplicate(self):
        """Registering an existing name without force raises ValueError."""

        class Dummy(SimEngine):
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

        register_backend("dup_test", lambda: Dummy, force=True)
        with pytest.raises(ValueError, match="already registered"):
            register_backend("dup_test", lambda: Dummy)

    def test_register_rejects_builtin_alias(self):
        """Cannot hijack built-in aliases like 'mj'."""

        class Dummy(SimEngine):
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

        with pytest.raises(ValueError, match="conflicts with built-in"):
            register_backend("custom_phys", lambda: Dummy, aliases=["mj"])


# ── Model Registry Tests ─────────────────────────────────────────


class TestModelRegistry:
    """Test URDF/MJCF model resolution."""

    def test_list_available_models_returns_robot_table(self):
        from strands_robots.simulation.model_registry import list_available_models

        models = list_available_models()
        assert isinstance(models, str)
        assert "so100" in models
        assert len(models) > 100

    def test_register_and_resolve_urdf(self, tmp_path):
        """Register a URDF, resolve it back — full round-trip."""
        from strands_robots.simulation.model_registry import register_urdf, resolve_urdf

        urdf_file = tmp_path / "robot.urdf"
        urdf_file.write_text("<robot/>")
        register_urdf("test_robot_xyz", str(urdf_file))
        result = resolve_urdf("test_robot_xyz")
        assert result == str(urdf_file)

    def test_list_registered_urdfs(self):
        from strands_robots.simulation.model_registry import list_registered_urdfs, register_urdf

        register_urdf("list_test_bot", "/fake/list.urdf")
        urdfs = list_registered_urdfs()
        assert isinstance(urdfs, dict)
        assert "list_test_bot" in urdfs


# ── Dataclass Behavioral Tests ───────────────────────────────────


class TestSimModelsUsage:
    """Test that simulation models behave correctly in real usage patterns."""

    def test_sim_world_tracks_robots(self):
        """SimWorld can add robots and objects — simulates real world setup."""
        world = SimWorld()
        robot = SimRobot(name="so100", urdf_path="/p")
        world.robots["so100"] = robot
        assert "so100" in world.robots
        assert world.status == SimStatus.IDLE

    def test_sim_object_preserves_originals_for_randomization(self):
        """SimObject stores original position/color for domain randomization reset."""
        obj = SimObject(name="ball", shape="sphere", position=[1, 2, 3], color=[1, 0, 0, 1])
        assert obj._original_position == [1, 2, 3]
        assert obj._original_color == [1, 0, 0, 1]

    def test_trajectory_step_records_episode_data(self):
        """TrajectoryStep captures full observation-action pair for dataset recording."""
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
        assert step.observation["state"] == [1, 2, 3]
