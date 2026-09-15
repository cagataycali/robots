"""``add_camera`` / ``remove_camera`` accept ``camera_name`` for ``name``.

Observed: an agent rendered from ``camera_name="wrist"`` (``render`` spells it
so) and then sent ``remove_camera {"camera_name": "wrist"}`` - refused with
``Unknown parameter 'camera_name'. Valid: ['name']``. Two spellings for the
same fact across the camera actions cost a step. The dispatcher already
extends this courtesy to ``name``/``robot_name``; this is its camera twin,
scoped to actions whose name says ``camera`` and whose method has ``name`` but
no ``camera_name`` of its own. ``render``'s own ``camera_name`` is untouched
and ``camera_name`` on a non-camera action stays unknown.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    sim = Simulation()
    sim.create_world()
    yield sim
    sim.destroy()


def _text(result: dict) -> str:
    return result["content"][0].get("text", "")


class TestCameraNameAliasTarget:
    def test_camera_action_with_name_maps_camera_name_to_name(self):
        assert Simulation._camera_name_alias_target("remove_camera", {"name"}) == "name"
        assert Simulation._camera_name_alias_target("add_camera", {"name", "position", "fov"}) == "name"

    def test_a_method_with_its_own_camera_name_is_left_alone(self):
        assert Simulation._camera_name_alias_target("render_camera", {"camera_name", "width"}) is None

    def test_a_non_camera_action_gets_no_alias(self):
        assert Simulation._camera_name_alias_target("add_object", {"name", "type"}) is None
        assert Simulation._camera_name_alias_target("remove_robot", {"name"}) is None


class TestDispatch:
    def test_remove_camera_accepts_camera_name(self, sim):
        assert sim._dispatch_action("add_camera", {"name": "wrist", "position": [1, 0, 1]})["status"] == "success"
        result = sim._dispatch_action("remove_camera", {"camera_name": "wrist"})
        assert result["status"] == "success", _text(result)
        assert "wrist" in _text(result)
        assert "wrist" not in sim._dispatch_action("list_cameras", {})["content"][0]["text"]

    def test_add_camera_accepts_camera_name(self, sim):
        result = sim._dispatch_action("add_camera", {"camera_name": "wrist", "position": [1, 0, 1]})
        assert result["status"] == "success", _text(result)
        assert "wrist" in sim._dispatch_action("list_cameras", {})["content"][0]["text"]

    def test_the_canonical_spelling_still_works(self, sim):
        assert sim._dispatch_action("add_camera", {"name": "c1", "position": [1, 0, 1]})["status"] == "success"
        assert sim._dispatch_action("remove_camera", {"name": "c1"})["status"] == "success"

    def test_an_unknown_camera_is_still_refused_by_the_method(self, sim):
        result = sim._dispatch_action("remove_camera", {"camera_name": "nope"})
        assert result["status"] == "error"
        assert "not found" in _text(result)

    def test_camera_name_on_a_non_camera_action_stays_unknown(self, sim):
        result = sim._dispatch_action("add_object", {"camera_name": "x", "type": "box"})
        assert result["status"] == "error"
        assert "Unknown parameter 'camera_name'" in _text(result)

    def test_render_does_not_gain_a_name_alias(self, sim):
        result = sim._dispatch_action("render", {"name": "default"})
        assert result["status"] == "error"
        assert "Unknown parameter 'name'" in _text(result)
