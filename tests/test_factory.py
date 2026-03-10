"""Tests for strands_robots.factory — Robot(), list_robots()."""

import pytest

from strands_robots.factory import (
    Robot,
    _ALIASES,
    _UNIFIED_ROBOTS,
    _resolve_name,
    list_robots,
)


class TestResolveNames:
    def test_canonical(self):
        assert _resolve_name("so100") == "so100"

    def test_alias(self):
        assert _resolve_name("franka") == "panda"
        assert _resolve_name("g1") == "unitree_g1"
        assert _resolve_name("h1") == "unitree_h1"

    def test_case_insensitive(self):
        assert _resolve_name("SO100") == "so100"
        assert _resolve_name("Panda") == "panda"

    def test_hyphen_to_underscore(self):
        assert _resolve_name("reachy-mini") == "reachy_mini"


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


class TestUnifiedRobots:
    def test_so100_exists(self):
        assert "so100" in _UNIFIED_ROBOTS
        assert _UNIFIED_ROBOTS["so100"]["sim"] == "so100"

    def test_all_aliases_point_to_valid_robots(self):
        for alias, canonical in _ALIASES.items():
            assert canonical in _UNIFIED_ROBOTS, (
                f"Alias '{alias}' points to unknown robot '{canonical}'"
            )

    def test_robot_count(self):
        """Ensure we have a reasonable number of robots."""
        assert len(_UNIFIED_ROBOTS) >= 30

    def test_all_robots_have_description(self):
        for name, info in _UNIFIED_ROBOTS.items():
            assert "description" in info, f"Robot '{name}' missing description"
            assert len(info["description"]) > 0
