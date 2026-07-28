"""Regression tests: a body-mounted camera survives the full scene rebuild.

``SpecBuilder.build`` deliberately does NOT attach robots - the eject path attaches
them afterwards, one fresh ``MjSpec`` per robot. But ``build`` added every camera,
including one MOUNTED on a robot body via ``parent_body`` (a wrist cam), whose
parent therefore did not exist yet. ``add_camera`` raised, and the exception
escaped the tool contract entirely:

    add_robot("a")
    add_camera("wrist", parent_body="a/hand")
    add_robot("b")
    remove_robot("b")
      -> ValueError: add_camera: parent_body 'a/hand' not found in scene.

An uncaught ``ValueError`` out of ``remove_robot``, so any scene with a wrist
camera could not remove a robot at all. Only the full-rebuild path was affected -
``add_object`` / ``add_robot`` / ``reset`` / ``patch_scene_mjcf`` all recompile the
live spec, where the parent body already exists.

Mounted cameras are now deferred to ``SpecBuilder.add_deferred_cameras``, called
after every robot has been re-attached.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
Image = pytest.importorskip("PIL.Image")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    s = Simulation(tool_name="mounted_camera_rebuild", mesh=False)
    s.create_world()
    assert s.add_robot(name="a", data_config="panda")["status"] == "success"
    assert (
        s.add_camera(name="wrist", position=[0.05, 0, 0.05], target=[0, 0, -0.1], parent_body="a/hand")["status"]
        == "success"
    )
    yield s
    s.destroy()


def _cam(sim, name: str) -> int:
    return mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, name)


def _frame_mean(sim, camera_name: str) -> float:
    result = sim.render(camera_name=camera_name, width=160, height=120)
    assert result["status"] == "success", result
    for block in result["content"]:
        if "image" in block:
            data = block["image"]["source"]["bytes"]
            return float(np.array(Image.open(io.BytesIO(data)).convert("RGB")).mean())
    raise AssertionError("render returned no image block")


def test_remove_robot_does_not_raise(sim) -> None:
    """The core defect: an uncaught ValueError out of the tool."""
    assert sim.add_robot(name="b", data_config="panda", position=[2, 0, 0])["status"] == "success"
    assert sim.remove_robot(name="b")["status"] == "success"


def test_mounted_camera_survives_the_rebuild(sim) -> None:
    assert sim.add_robot(name="b", data_config="panda", position=[2, 0, 0])["status"] == "success"
    assert sim.remove_robot(name="b")["status"] == "success"

    assert _cam(sim, "wrist") >= 0, "wrist camera lost in the rebuild"
    assert "wrist" in sim._world.cameras


def test_mounted_camera_is_still_bound_to_its_parent(sim) -> None:
    """Body ids shift across a rebuild, so the binding must be re-resolved."""
    assert sim.add_robot(name="b", data_config="panda", position=[2, 0, 0])["status"] == "success"
    assert sim.remove_robot(name="b")["status"] == "success"

    model = sim.mj_model
    cam = _cam(sim, "wrist")
    hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "a/hand")
    assert hand >= 0
    assert int(model.cam_bodyid[cam]) == hand


def test_mounted_camera_still_follows_the_arm_after_the_rebuild(sim) -> None:
    """A wrist cam that no longer tracks the gripper is silently useless."""
    assert sim.add_robot(name="b", data_config="panda", position=[2, 0, 0])["status"] == "success"
    assert sim.remove_robot(name="b")["status"] == "success"

    cam = _cam(sim, "wrist")
    before = np.asarray(sim.mj_data.cam_xpos[cam]).copy()
    sim.mj_data.ctrl[:7] = [0.4, 0.3, 0.0, -1.7, 0.0, 1.5, 0.5]
    assert sim.step(n_steps=600)["status"] == "success"

    after = np.asarray(sim.mj_data.cam_xpos[_cam(sim, "wrist")])
    assert float(np.abs(after - before).max()) > 0.01, "mounted camera stopped tracking its parent"


def test_mounted_camera_still_renders_after_the_rebuild(sim) -> None:
    before = _frame_mean(sim, "wrist")
    assert sim.add_robot(name="b", data_config="panda", position=[2, 0, 0])["status"] == "success"
    assert sim.remove_robot(name="b")["status"] == "success"
    assert _frame_mean(sim, "wrist") == pytest.approx(before, rel=0.05)


def test_world_fixed_cameras_are_unaffected(sim) -> None:
    """The non-deferred path must keep working exactly as before."""
    assert sim.add_camera(name="fixed", position=[1, 1, 1], target=[0, 0, 0])["status"] == "success"
    assert sim.add_robot(name="b", data_config="panda", position=[2, 0, 0])["status"] == "success"
    assert sim.remove_robot(name="b")["status"] == "success"

    for name in ("default", "fixed", "wrist"):
        assert _cam(sim, name) >= 0, name
    model = sim.mj_model
    assert int(model.cam_bodyid[_cam(sim, "fixed")]) == 0, "world camera must stay on the worldbody"


def test_camera_mounted_on_the_removed_robot_is_dropped_not_fatal(sim) -> None:
    """Its mount point is gone, so it cannot be reinstated - but must not raise."""
    assert sim.add_robot(name="b", data_config="panda", position=[2, 0, 0])["status"] == "success"
    assert (
        sim.add_camera(name="bcam", position=[0.05, 0, 0.05], target=[0, 0, -0.1], parent_body="b/hand")["status"]
        == "success"
    )
    assert sim.remove_robot(name="b")["status"] == "success"
    # The surviving robot's camera is intact; the orphaned one is simply absent.
    assert _cam(sim, "wrist") >= 0
    assert _cam(sim, "bcam") < 0
    # And the world is still usable.
    assert sim.step(n_steps=10)["status"] == "success"


def test_mounted_camera_survives_two_consecutive_rebuilds(sim) -> None:
    for i in range(2):
        name = f"tmp{i}"
        assert sim.add_robot(name=name, data_config="panda", position=[2 + i, 0, 0])["status"] == "success"
        assert sim.remove_robot(name=name)["status"] == "success"
        assert _cam(sim, "wrist") >= 0, f"lost after rebuild {i}"
