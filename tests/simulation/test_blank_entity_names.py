"""Regression tests: a blank object / camera name is refused, not silently accepted.

``add_object(name="")`` and ``add_camera(name="")`` reported success and compiled an
UNNAMED MJCF entity. ``mj_id2name`` returns ``None`` for those, so nothing can
address them again, while the Python-side registry keeps insisting they exist:

    add_object(name="")        -> success
    list_objects()            -> "  - : box at [0.4, 0, 0.3], 0.1kg"
    get_body_state("")        -> error, "Body '' not found."
    move_object("")           -> success        (registry hit, model miss)
    body_above_z(body="")     -> False forever  (a benchmark that can never pass)
    add_object(name="") again -> error, "Object '' exists."

so the scene the agent sees and the scene MuJoCo simulates disagree, permanently.

The camera case is worse: ``""`` is one of the reserved FREE-camera tokens, so
``render(camera_name="")`` silently returns the free view instead of the camera just
created - a completely different viewpoint, reported as success.

``add_robot`` was already safe: it falls back to the ``data_config`` name.
"""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

_BLANK = ["", " ", "\t", "\n", "   ", None, 123]


@pytest.fixture
def sim():
    s = Simulation(tool_name="blank_entity_names", mesh=False)
    s.create_world()
    yield s
    s.destroy()


@pytest.mark.parametrize("bad", _BLANK)
def test_add_object_refuses_a_blank_name(sim, bad) -> None:
    result = sim.add_object(name=bad, shape="box", size=[0.04] * 3, position=[0.4, 0, 0.3], mass=0.1)
    assert result["status"] == "error"
    assert "non-empty string" in result["content"][0]["text"]


@pytest.mark.parametrize("bad", _BLANK)
def test_add_camera_refuses_a_blank_name(sim, bad) -> None:
    result = sim.add_camera(name=bad, position=[1, 1, 1], target=[0, 0, 0])
    assert result["status"] == "error"
    assert "non-empty string" in result["content"][0]["text"]


def test_a_refused_name_leaves_no_unnamed_body_behind(sim) -> None:
    """The direct invariant: no body in the model may be nameless."""
    sim.add_object(name="", shape="box", size=[0.04] * 3, position=[0.4, 0, 0.3], mass=0.1)
    model = sim.mj_model
    for body_id in range(int(model.nbody)):
        assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id), f"body {body_id} is unnamed"
    assert "" not in sim._world.objects


def test_a_refused_camera_leaves_no_unnamed_camera_behind(sim) -> None:
    sim.add_camera(name="", position=[1, 1, 1], target=[0, 0, 0])
    model = sim.mj_model
    for cam_id in range(int(model.ncam)):
        assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id), f"camera {cam_id} is unnamed"
    assert "" not in sim._world.cameras


def test_the_registry_and_the_model_agree(sim) -> None:
    """Every tracked object must resolve in the compiled model."""
    sim.add_object(name="", shape="box", size=[0.04] * 3, position=[0.4, 0, 0.3], mass=0.1)
    assert (
        sim.add_object(name="cube", shape="box", size=[0.04] * 3, position=[0.4, 0, 0.3], mass=0.1)["status"]
        == "success"
    )
    model = sim.mj_model
    for name in sim._world.objects:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0, name


def test_a_valid_name_still_works(sim) -> None:
    """Guard against the fix degenerating into 'reject everything'."""
    assert (
        sim.add_object(name="cube", shape="box", size=[0.04] * 3, position=[0.4, 0, 0.3], mass=0.1)["status"]
        == "success"
    )
    assert sim.add_camera(name="cam", position=[1, 1, 1], target=[0, 0, 0])["status"] == "success"
    assert sim.get_body_state(body_name="cube")["status"] == "success"
    assert sim.render(camera_name="cam", width=64, height=48)["status"] == "success"


def test_an_internal_space_is_still_allowed(sim) -> None:
    """The guard is on BLANK names, not on unusual ones - do not over-tighten."""
    assert (
        sim.add_object(name="my cube", shape="box", size=[0.04] * 3, position=[0.4, 0, 0.3], mass=0.1)["status"]
        == "success"
    )
    assert sim.get_body_state(body_name="my cube")["status"] == "success"


def test_add_robot_with_a_blank_name_still_falls_back(sim) -> None:
    """It was already safe; pin that so the fix does not have to change it."""
    result = sim.add_robot(name="", data_config="panda")
    assert result["status"] == "success"
    assert "panda" in sim._world.robots
    assert "" not in sim._world.robots
