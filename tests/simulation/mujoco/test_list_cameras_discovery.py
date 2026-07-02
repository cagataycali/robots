"""``list_cameras`` is a first-class camera discovery surface on the MuJoCo backend.

The Newton backend exposes a public ``list_cameras()`` and advertises it in
``describe()`` (and always lists the built-in ``"default"`` free view), but the
default MuJoCo backend had no such method: ``sim.list_cameras()`` raised
``AttributeError``, and ``describe()["cameras"]`` was built from a raw
``model.ncam`` loop whose contents depended on whether the loaded MJCF happened
to bake a camera literally named ``"default"`` -- so the two backends reported
different camera sets for the same query, and ``render("default")`` (the default
argument!) targeted a view that discovery could omit.

These tests pin the parity contract: ``list_cameras()`` exists, always begins
with ``"default"`` (deduplicated), includes model + user cameras, equals
``describe()["cameras"]``, and is reachable as an agent action alongside
``list_robots`` / ``list_objects`` / ``list_bodies``. They are GL-free (no
rendering), so they run in CI without an OpenGL context.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from strands_robots import create_simulation


def _sim_with_robot():
    sim = create_simulation(backend="mujoco")
    sim.create_world(ground_plane=True)
    sim.add_robot("so101")
    return sim


def test_list_cameras_is_public_and_lists_default():
    """The default backend exposes list_cameras() starting with 'default'."""
    sim = _sim_with_robot()
    assert hasattr(sim, "list_cameras"), "MuJoCo backend must expose list_cameras()"
    cams = sim.list_cameras()
    assert isinstance(cams, list)
    assert cams[0] == "default"
    # 'default' is the free-view token render() accepts; it must be listed once
    assert cams.count("default") == 1


def test_list_cameras_includes_user_camera_deduped():
    """A camera added via add_camera appears exactly once; 'default' stays unique."""
    sim = _sim_with_robot()
    sim.add_camera(
        "wristcam",
        position=[0.05, 0.0, 0.15],
        target=[0.3, 0.0, 0.05],
        width=256,
        height=256,
        parent_body="so101/gripper",
    )
    cams = sim.list_cameras()
    assert "wristcam" in cams
    assert cams.count("default") == 1
    assert cams.count("wristcam") == 1


def test_describe_cameras_equals_list_cameras():
    """describe()['cameras'] is the single source of truth: it equals list_cameras()."""
    sim = _sim_with_robot()
    sim.add_camera(
        "topcam",
        position=[0.2, 0.0, 0.7],
        target=[0.2, 0.0, 0.0],
        width=128,
        height=128,
    )
    d = sim.describe()
    assert d["cameras"] == sim.list_cameras()
    assert "default" in d["cameras"]
    assert "topcam" in d["cameras"]
    assert "list_cameras" in d["methods"]


def test_list_cameras_no_world_matches_newton():
    """With no world the default view is still advertised (Newton-consistent)."""
    sim = create_simulation(backend="mujoco")
    assert sim.list_cameras() == ["default"]
    assert sim.describe()["cameras"] == ["default"]


def test_list_cameras_agent_action_dispatches():
    """The agent action list_cameras returns a tool-result dict (not AttributeError)."""
    sim = _sim_with_robot()
    res = sim._dispatch_action("list_cameras", {"action": "list_cameras"})
    assert res["status"] == "success", res
    text = res["content"][0]["text"]
    assert "default" in text


def test_list_cameras_in_tool_spec_enum():
    """The LLM-facing tool schema advertises list_cameras alongside the other lists."""
    # Load the actual shipped tool_spec.json from the package.
    import strands_robots.simulation.mujoco as mjpkg

    ts = Path(mjpkg.__file__).parent / "tool_spec.json"
    enum = json.load(open(ts))["properties"]["action"]["enum"]
    assert "list_cameras" in enum
    # parity with the sibling discovery actions
    for sibling in ("list_robots", "list_objects", "list_bodies"):
        assert sibling in enum
