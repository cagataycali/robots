"""``add_camera`` / ``remove_camera`` accept ``camera_name`` for ``name``.

Observed: an agent rendered from ``camera_name="wrist"`` (``render`` spells it
so) and then sent ``remove_camera {"camera_name": "wrist"}`` - refused with
``Unknown parameter 'camera_name'. Valid: ['name']``. Two spellings for the
same fact across the camera actions cost a step. The dispatcher already
extends this courtesy to ``name``/``robot_name``; this is its camera twin,
scoped to actions whose name says ``camera`` and whose method has ``name`` but
no ``camera_name`` of its own. ``render``'s own ``camera_name`` is untouched
and ``camera_name`` on a non-camera action stays unknown.

The alias is only offered where ``name`` IS the camera. A camera action that
already names cameras through another parameter spells a different fact in its
``name``: ``start_cameras_recording`` selects with ``cameras`` and its ``name``
is the output filename tag, so binding a camera there recorded every camera in
the scene under that tag and reported success - while the refusal it would
replace names ``cameras`` in its ``Valid:`` list.
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
        # render and its siblings, which have no ``name`` at all...
        assert Simulation._camera_name_alias_target("render_camera", {"camera_name", "width"}) is None
        # ...and a method declaring both: it spells the camera as
        # ``camera_name``, so its ``name`` is some other fact.
        assert Simulation._camera_name_alias_target("render_camera", {"camera_name", "name", "width"}) is None

    def test_a_non_camera_action_gets_no_alias(self):
        assert Simulation._camera_name_alias_target("add_object", {"name", "type"}) is None
        assert Simulation._camera_name_alias_target("remove_robot", {"name"}) is None

    def test_a_method_that_names_cameras_elsewhere_keeps_its_own_name(self):
        # start_cameras_recording selects through ``cameras``; its ``name`` is
        # the output filename tag, so ``camera_name`` must not reach it.
        params = {"cameras", "output_dir", "fps", "width", "height", "name", "max_frames_per_camera"}
        assert Simulation._camera_name_alias_target("start_cameras_recording", params) is None

    def test_only_the_actions_whose_name_is_the_camera_offer_the_alias(self, sim):
        # Enumerated from the shipped action list rather than asserted per
        # action, so a new camera action whose ``name`` means something else
        # cannot quietly inherit the alias.
        import inspect
        import json
        from pathlib import Path

        import strands_robots.simulation.mujoco as mj_module

        spec = json.loads((Path(mj_module.__file__).parent / "tool_spec.json").read_text())
        actions = spec["properties"]["action"]["enum"]
        aliased = {
            action
            for action in actions
            if "camera" in action
            and callable(method := getattr(sim, action, None))
            and Simulation._camera_name_alias_target(
                action, {p for p in inspect.signature(method).parameters if p != "self"}
            )
        }
        assert aliased == {"add_camera", "remove_camera"}


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

    def test_start_cameras_recording_still_refuses_camera_name(self, sim):
        result = sim._dispatch_action("start_cameras_recording", {"camera_name": "wrist"})
        assert result["status"] == "error"
        text = _text(result)
        assert "Unknown parameter 'camera_name'" in text
        # The refusal is what teaches the parameter that does select cameras.
        assert "cameras" in text

    def test_render_does_not_gain_a_name_alias(self, sim):
        result = sim._dispatch_action("render", {"name": "default"})
        assert result["status"] == "error"
        assert "Unknown parameter 'name'" in _text(result)
