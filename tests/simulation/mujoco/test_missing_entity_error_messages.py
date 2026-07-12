"""Regression tests: the ``move_object`` / ``remove_object`` / ``remove_camera``
facade paths return an *actionable* error for an unknown entity instead of a
dead-end ``"<Kind> 'X' not found."``.

Before this change these three paths returned a bare ``"Object 'X' not found."``
or ``"Camera 'X' not found."`` with no list of what *is* available and no
close-match suggestion - forcing an agent driving the API blind into a discovery
round-trip on every typo. The camera *render*/*record* paths already listed
``Available: [...]`` and ``add_robot`` (#1299) already offered a difflib
close-match; these tests pin that the same actionable shape now covers the
remove/move-by-name paths: the message names the entity, offers a close match,
lists the available names, and points at the discovery action
(``list_objects`` / ``list_cameras_info``).

The messages keep the ``"<Kind> 'X' not found."`` prefix so the consistent
error shape (T15 in ``test_agenttool_contract``) is preserved. GL-free
(``mesh=False``, no rendering) so it runs in CI without a GPU.
"""

import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    s = Simulation(tool_name="test_missing_entity_msgs_sim", mesh=False)
    s.create_world(gravity=[0, 0, -9.81])
    s.add_object("cube", shape="box", size=[0.03, 0.03, 0.03], position=[0.15, -0.12, 0.03], is_static=False)
    s.add_camera(name="front_cam", position=[0.5, 0.0, 0.35], target=[0.15, 0.0, 0.05])
    yield s
    s.cleanup()


def _err_text(result):
    assert result["status"] == "error", result
    return result["content"][0]["text"]


def test_move_object_unknown_is_actionable(sim):
    text = _err_text(sim.move_object("crube", position=[0.2, -0.1, 0.03]))
    assert "Object 'crube' not found" in text  # preserved prefix (T15 shape)
    assert "Did you mean: cube" in text  # close-match
    assert "cube" in text  # names the available object
    assert "list_objects" in text  # discovery surface


def test_remove_object_unknown_is_actionable(sim):
    text = _err_text(sim.remove_object("cubee"))
    assert "Object 'cubee' not found" in text
    assert "Did you mean: cube" in text
    assert "list_objects" in text


def test_remove_camera_unknown_lists_available_and_suggests(sim):
    text = _err_text(sim.remove_camera("frnt_cam"))
    assert "Camera 'frnt_cam' not found" in text
    assert "Did you mean: front_cam" in text
    assert "front_cam" in text  # names the available camera
    assert "list_cameras_info" in text


def test_missing_object_in_empty_scene_points_to_add_object():
    s = Simulation(tool_name="test_missing_entity_empty_sim", mesh=False)
    s.create_world(gravity=[0, 0, -9.81])
    try:
        text = _err_text(s.move_object("cube", position=[0, 0, 0.1]))
        assert "Object 'cube' not found" in text
        assert "add_object" in text  # no objects -> point at how to add one
    finally:
        s.cleanup()


def test_valid_move_and_remove_unaffected(sim):
    # No-regression guard: the happy paths still succeed and are not intercepted
    # by the new missing-entity messaging.
    assert sim.move_object("cube", position=[0.2, -0.1, 0.03])["status"] == "success"
    assert sim.remove_camera("front_cam")["status"] == "success"
    assert sim.remove_object("cube")["status"] == "success"
