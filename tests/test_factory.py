"""Tests for strands_robots.factory — Robot(), list_robots()."""

from strands_robots.factory import list_robots
from strands_robots.registry import (
    get_robot,
    list_aliases,
    resolve_name,
)
from strands_robots.registry import (
    list_robots as registry_list_robots,
)


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
