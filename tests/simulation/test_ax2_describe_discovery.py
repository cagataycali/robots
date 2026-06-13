"""AX-2 regression test: Simulation.describe() + registry.get_robot_info.

These tests verify the discovery surface added in AX-2 so agents can learn
an engine's contract in a single call without probe-and-fail.

Pre-fix state: all three assertions below FAIL:
  - sim.describe() does not exist -> AttributeError
  - registry.get_robot_info does not exist -> ImportError
  - describe()["robots"] cannot equal list_robots() if describe() is absent
"""

from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Part 1: Simulation.describe() on the ABC (no MuJoCo needed)
# ---------------------------------------------------------------------------


def _make_minimal_engine():
    """Create a minimal SimEngine with one robot for describe() testing."""
    from strands_robots.simulation.base import SimEngine

    class MinimalEngine(SimEngine):
        """Smallest concrete engine to test describe()."""

        def __init__(self):
            self._robots = []

        def create_world(self, **kw) -> dict[str, Any]:
            return {}

        def destroy(self) -> dict[str, Any]:
            return {}

        def reset(self) -> dict[str, Any]:
            return {}

        def step(self, n_steps=1) -> dict[str, Any]:
            return {}

        def get_state(self) -> dict[str, Any]:
            return {}

        def add_robot(self, name, **kw) -> dict[str, Any]:
            self._robots.append(name)
            return {}

        def remove_robot(self, name) -> dict[str, Any]:
            self._robots.remove(name)
            return {}

        def list_robots(self) -> list[str]:
            return list(self._robots)

        def robot_joint_names(self, robot_name) -> list[str]:
            return ["joint_0", "joint_1"]

        def add_object(self, name, **kw) -> dict[str, Any]:
            return {}

        def remove_object(self, name) -> dict[str, Any]:
            return {}

        def get_observation(self, robot_name=None, **kw) -> dict[str, Any]:
            return {}

        def send_action(self, action, robot_name=None, n_substeps=1) -> None:
            pass

        def render(self, camera_name="default", **kw) -> dict[str, Any]:
            return {}

    return MinimalEngine()


class TestDescribeABC:
    """Tests for SimEngine.describe() on the abstract base class."""

    def test_describe_exists(self):
        """describe() must exist on SimEngine instances."""
        engine = _make_minimal_engine()
        result = engine.describe()
        assert isinstance(result, dict)

    def test_describe_robots_equals_list_robots(self):
        """describe()['robots'] must equal list_robots()."""
        engine = _make_minimal_engine()
        engine.add_robot("test_bot")
        desc = engine.describe()
        assert desc["robots"] == engine.list_robots()
        assert desc["robots"] == ["test_bot"]

    def test_describe_robots_empty_when_no_robots(self):
        """describe()['robots'] returns empty list when no robots loaded."""
        engine = _make_minimal_engine()
        desc = engine.describe()
        assert desc["robots"] == []

    def test_describe_has_get_robot_state_key(self):
        """describe()['methods'] must contain 'get_robot_state'."""
        engine = _make_minimal_engine()
        desc = engine.describe()
        assert "methods" in desc
        assert "get_robot_state" in desc["methods"]

    def test_describe_has_cameras_key(self):
        """describe() must contain 'cameras' key."""
        engine = _make_minimal_engine()
        desc = engine.describe()
        assert "cameras" in desc
        assert isinstance(desc["cameras"], list)

    def test_describe_has_note(self):
        """describe() must contain a 'note' key explaining robot_name default."""
        engine = _make_minimal_engine()
        desc = engine.describe()
        assert "note" in desc
        assert "robot_name" in desc["note"]

    def test_describe_methods_includes_core_set(self):
        """describe()['methods'] must cover the core agent-callable methods."""
        engine = _make_minimal_engine()
        desc = engine.describe()
        expected_methods = {
            "get_robot_state",
            "get_observation",
            "send_action",
            "run_policy",
            "list_robots",
            "render",
        }
        assert expected_methods.issubset(set(desc["methods"].keys()))


# ---------------------------------------------------------------------------
# Part 2: MuJoCo backend describe() override (needs mujoco)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not pytest.importorskip("mujoco", reason="MuJoCo not installed"),
    reason="MuJoCo not available",
)
class TestDescribeMuJoCo:
    """Tests for MuJoCoSimEngine.describe() with a live sim world."""

    def test_describe_with_world_and_robot(self):
        """MuJoCo describe() returns cameras + world_created after setup."""
        import os

        os.environ.setdefault("MUJOCO_GL", "egl")
        from strands_robots.simulation import Simulation

        sim = Simulation()
        try:
            sim.create_world()
            sim.add_robot("so100", data_config="so100")
            desc = sim.describe()

            assert desc["robots"] == sim.list_robots()
            assert "so100" in desc["robots"]
            assert desc["world_created"] is True
            assert isinstance(desc["cameras"], list)
            assert "get_robot_state" in desc["methods"]
        finally:
            sim.destroy()

    def test_describe_no_world(self):
        """MuJoCo describe() works even before create_world (empty robots)."""
        import os

        os.environ.setdefault("MUJOCO_GL", "egl")
        from strands_robots.simulation import Simulation

        sim = Simulation()
        desc = sim.describe()
        assert desc["robots"] == []
        assert desc["world_created"] is False


# ---------------------------------------------------------------------------
# Part 3: registry.get_robot_info alias
# ---------------------------------------------------------------------------


class TestRegistryGetRobotInfo:
    """Tests that registry.get_robot_info is importable and functional."""

    def test_get_robot_info_importable(self):
        """from strands_robots.registry import get_robot_info must work."""
        from strands_robots.registry import get_robot_info

        assert callable(get_robot_info)

    def test_get_robot_info_same_as_get_robot(self):
        """get_robot_info must return the same object as get_robot."""
        from strands_robots.registry import get_robot, get_robot_info

        # They should be the same function
        assert get_robot_info is get_robot

    def test_get_robot_info_returns_data(self):
        """get_robot_info('so100') must return a non-None dict."""
        from strands_robots.registry import get_robot_info

        info = get_robot_info("so100")
        assert info is not None
        assert isinstance(info, dict)
