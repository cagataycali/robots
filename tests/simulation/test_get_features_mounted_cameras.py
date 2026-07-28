"""Regression tests: ``get_features`` reports a robot's body-mounted cameras.

``get_features(robot_name=X)`` scoped its camera list by NAME PREFIX. That finds a
camera declared inside the robot's own MJCF (MuJoCo namespaces it ``arm2/wrist``),
but a camera added at runtime via ``add_camera(parent_body="a/hand")`` keeps its
bare name while being physically bolted to that robot - so it was invisible:

    add_robot("a"); add_robot("b")
    add_camera("wa", parent_body="a/hand")
    add_camera("wb", parent_body="b/hand")

    get_features(robot_name="a")["camera_names"]  -> []          wrong
    get_observation("a")                          -> wa, wb, ... frames exist

``get_features`` is what the dataset recorder reads to declare its columns, so a
robot with a working wrist camera advertised a schema with no camera at all.

``model.cam_bodyid`` is the unambiguous ownership signal (measured: wa -> a/hand,
wb -> b/hand), so mounted cameras are now unioned in by walking each camera's body
up ``body_parentid`` to see whether the chain passes through the robot.

Note ``get_observation`` deliberately returns EVERY camera in the model - a policy
picks what it needs and ``start_recording(cameras=...)`` scopes the dataset via
``_drop_unrecorded_cameras`` - so that side is left as-is.
"""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


def _features(sim, robot_name=None):
    result = sim.get_features(robot_name=robot_name) if robot_name else sim.get_features()
    assert result["status"] == "success", result
    return [b["json"] for b in result["content"] if "json" in b][0]["features"]


@pytest.fixture
def sim():
    s = Simulation(tool_name="get_features_mounted_cameras", mesh=False)
    s.create_world()
    yield s
    s.destroy()


def _mount(sim, name: str, body: str):
    return sim.add_camera(name=name, position=[0.05, 0, 0.05], target=[0, 0, -0.1], parent_body=body)


def test_single_robot_mounted_camera_is_reported(sim) -> None:
    """The core defect: camera_names was empty for a robot with a wrist cam."""
    assert sim.add_robot(name="a", data_config="panda")["status"] == "success"
    assert _mount(sim, "wrist", "a/hand")["status"] == "success"

    features = _features(sim, "a")
    assert "wrist" in features["camera_names"]
    assert features["n_cameras"] == len(features["camera_names"])


def test_each_robot_gets_only_its_own_mounted_camera(sim) -> None:
    assert sim.add_robot(name="a", data_config="panda")["status"] == "success"
    assert sim.add_robot(name="b", data_config="panda", position=[2, 0, 0])["status"] == "success"
    assert _mount(sim, "wa", "a/hand")["status"] == "success"
    assert _mount(sim, "wb", "b/hand")["status"] == "success"

    assert _features(sim, "a")["camera_names"] == ["wa"]
    assert _features(sim, "b")["camera_names"] == ["wb"]


def test_a_world_fixed_camera_is_not_claimed_by_any_robot(sim) -> None:
    """World cameras sit on body 0 and belong to no robot."""
    assert sim.add_robot(name="a", data_config="panda")["status"] == "success"
    assert sim.add_camera(name="world_cam", position=[1, 1, 1], target=[0, 0, 0])["status"] == "success"

    names = _features(sim, "a")["camera_names"]
    assert "world_cam" not in names
    assert "default" not in names


def test_the_schema_matches_what_is_renderable(sim) -> None:
    """Every advertised camera must actually render."""
    assert sim.add_robot(name="a", data_config="panda")["status"] == "success"
    assert _mount(sim, "wrist", "a/hand")["status"] == "success"

    for name in _features(sim, "a")["camera_names"]:
        assert sim.render(camera_name=name, width=64, height=48)["status"] == "success", name


def test_a_camera_deeper_in_the_subtree_is_claimed(sim) -> None:
    """Ownership is the whole body subtree, not just the top-level link."""
    assert sim.add_robot(name="a", data_config="panda")["status"] == "success"
    assert _mount(sim, "finger_cam", "a/left_finger")["status"] == "success"
    assert "finger_cam" in _features(sim, "a")["camera_names"]


def test_mounted_camera_survives_the_full_rebuild(sim) -> None:
    assert sim.add_robot(name="a", data_config="panda")["status"] == "success"
    assert _mount(sim, "wrist", "a/hand")["status"] == "success"
    assert sim.add_robot(name="b", data_config="panda", position=[2, 0, 0])["status"] == "success"
    assert sim.remove_robot(name="b")["status"] == "success"
    assert "wrist" in _features(sim, "a")["camera_names"]


def test_unscoped_features_still_list_every_camera(sim) -> None:
    assert sim.add_robot(name="a", data_config="panda")["status"] == "success"
    assert _mount(sim, "wrist", "a/hand")["status"] == "success"
    assert sim.add_camera(name="world_cam", position=[1, 1, 1], target=[0, 0, 0])["status"] == "success"

    names = _features(sim)["camera_names"]
    for expected in ("default", "wrist", "world_cam"):
        assert expected in names, expected


def test_no_duplicate_when_a_camera_matches_both_rules(sim) -> None:
    """A namespaced camera mounted on its own robot must appear once."""
    assert sim.add_robot(name="a", data_config="panda")["status"] == "success"
    assert _mount(sim, "wrist", "a/hand")["status"] == "success"
    names = _features(sim, "a")["camera_names"]
    assert len(names) == len(set(names))


def test_a_robot_with_no_camera_reports_none(sim) -> None:
    """Guard against the fix over-claiming."""
    assert sim.add_robot(name="a", data_config="panda")["status"] == "success"
    assert sim.add_robot(name="b", data_config="panda", position=[2, 0, 0])["status"] == "success"
    assert _mount(sim, "wa", "a/hand")["status"] == "success"
    assert _features(sim, "b")["camera_names"] == []
