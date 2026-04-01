"""Tests for strands_robots.factory — Robot(), list_robots()."""

import os

import pytest

from strands_robots.factory import Robot, _auto_detect_mode, list_robots
from strands_robots.registry import (
    get_robot,
    list_aliases,
    resolve_name,
)
from strands_robots.registry import list_robots as registry_list_robots


class TestResolveNames:
    def test_canonical(self):
        assert resolve_name("so100") == "so100"

    def test_alias(self):
        assert resolve_name("franka") == "panda"
        assert resolve_name("g1") == "unitree_g1"
        assert resolve_name("h1") == "unitree_h1"

    def test_case_insensitive(self):
        assert resolve_name("SO100") == "so100"
        assert resolve_name("Panda") == "panda"

    def test_hyphen_to_underscore(self):
        assert resolve_name("reachy-mini") == "reachy_mini"


class TestListRobots:
    def test_list_all(self):
        robots = list_robots("all")
        assert len(robots) > 0
        names = [r["name"] for r in robots]
        assert "so100" in names
        assert "panda" in names

    def test_list_sim(self):
        robots = list_robots("sim")
        for r in robots:
            assert r["has_sim"] is True

    def test_list_real(self):
        robots = list_robots("real")
        for r in robots:
            assert r["has_real"] is True

    def test_list_both(self):
        robots = list_robots("both")
        for r in robots:
            assert r["has_sim"] is True
            assert r["has_real"] is True

    def test_robot_has_fields(self):
        robots = list_robots()
        for r in robots:
            assert "name" in r
            assert "description" in r
            assert "has_sim" in r
            assert "has_real" in r


class TestRobotRegistry:
    def test_so100_exists(self):
        info = get_robot("so100")
        assert info is not None
        assert "asset" in info
        assert info["asset"]["dir"] == "trs_so_arm100"

    def test_all_aliases_point_to_valid_robots(self):
        aliases = list_aliases()
        for alias, canonical in aliases.items():
            info = get_robot(canonical)
            assert info is not None, f"Alias '{alias}' points to unknown robot '{canonical}'"

    def test_robot_count(self):
        """Ensure we have a reasonable number of robots."""
        robots = registry_list_robots()
        assert len(robots) >= 30

    def test_all_robots_have_description(self):
        robots = registry_list_robots()
        for r in robots:
            assert "description" in r, f"Robot '{r['name']}' missing description"
            assert len(r["description"]) > 0


class TestAutoDetectMode:
    def test_defaults_to_sim(self):
        """No hardware plugged in → sim."""
        assert _auto_detect_mode("so100") == "sim"

    def test_env_override_real(self):
        os.environ["STRANDS_ROBOT_MODE"] = "real"
        try:
            assert _auto_detect_mode("so100") == "real"
        finally:
            del os.environ["STRANDS_ROBOT_MODE"]

    def test_env_override_sim(self):
        os.environ["STRANDS_ROBOT_MODE"] = "sim"
        try:
            assert _auto_detect_mode("so100") == "sim"
        finally:
            del os.environ["STRANDS_ROBOT_MODE"]

    def test_env_override_case_insensitive(self):
        os.environ["STRANDS_ROBOT_MODE"] = "REAL"
        try:
            # .lower() normalizes to "real" — should match
            mode = _auto_detect_mode("so100")
            assert mode == "real"  # .lower() normalizes REAL → real
        finally:
            del os.environ["STRANDS_ROBOT_MODE"]


class TestRobotFactory:
    def test_robot_is_callable(self):
        """Robot is a factory function, not a class."""
        import inspect

        assert callable(Robot)
        assert not inspect.isclass(Robot)

    def test_unknown_backend_raises(self):
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            Robot("so100", mode="sim", backend="isaac")

    def test_newton_not_implemented(self):
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            Robot("so100", mode="sim", backend="newton")

    def test_sim_with_urdf_path(self):
        """Robot() with explicit urdf_path should work (if file exists)."""
        # We don't have a real URDF here, but verify the param is accepted
        with pytest.raises(RuntimeError):
            Robot("test_bot", mode="sim", urdf_path="/nonexistent/robot.xml")

    def test_sim_happy_path_mujoco(self):
        """Happy-path: create a MuJoCo sim, step physics, destroy."""
        mujoco = pytest.importorskip("mujoco")
        sim = Robot("so100", mode="sim", backend="mujoco")
        try:
            # Verify it's a working simulation instance
            assert sim._world is not None
            assert sim._world._model is not None
            assert sim._world._data is not None
            # Step physics once to verify the engine works
            mujoco.mj_step(sim._world._model, sim._world._data)
            assert sim._world._data.time > 0
        finally:
            sim.destroy()

    def test_import_from_top_level(self):
        """Robot and list_robots importable from strands_robots."""
        from strands_robots import Robot as R
        from strands_robots import list_robots as lr

        assert R is Robot
        assert lr is list_robots
