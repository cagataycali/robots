"""Regression tests: ``render`` stays correct after EVERY scene mutation.

``spec.recompile`` preserves qpos/qvel but leaves ``mjData``'s derived world
transforms zeroed, and ``render`` consumes ``xpos`` / ``cam_xpos`` / ``cam_xmat``
/ ``light_xpos`` directly without a forward pass of its own. Every recompile site
therefore has to run ``mj_forward`` - and each carries a comment saying so,
because a missing one produces a black frame.

``remove_camera`` was the one site that did not:

    render() frame mean   122.26 -> 25.49   after remove_camera
    cam_xpos              [0, 0, 0]
    cam_xmat              all zero
    light_xpos            [0,0,0] x3        (lights still in the model)

The model was byte-identical either side of the call (same ncam, same lights) -
only ``mjData`` was stale, so nothing in the model dump hinted at it and the tool
reported success. An agent that removed a camera lost its view of the scene.

The forward pass now lives in ``scene_ops._install_model`` alongside the
generation bump, so a new swap site cannot reintroduce the omission.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
Image = pytest.importorskip("PIL.Image")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# A lit scene renders far brighter than this; a stale-mjData frame measured 25.5.
_MIN_FRAME_MEAN = 60.0


def _frame_mean(sim, camera_name: str = "default") -> float:
    result = sim.render(camera_name=camera_name, width=200, height=150)
    assert result["status"] == "success", result
    for block in result["content"]:
        if "image" in block:
            data = block["image"]["source"]["bytes"]
            return float(np.array(Image.open(io.BytesIO(data)).convert("RGB")).mean())
    raise AssertionError("render returned no image block")


@pytest.fixture
def sim():
    s = Simulation(tool_name="render_survives_mutations", mesh=False)
    s.create_world()
    assert s.add_robot(name="a", data_config="panda")["status"] == "success"
    yield s
    s.destroy()


def test_remove_camera_does_not_darken_the_frame(sim) -> None:
    """The core defect."""
    assert sim.add_camera(name="cm", position=[1, 1, 1], target=[0, 0, 0])["status"] == "success"
    before = _frame_mean(sim)
    assert sim.remove_camera(name="cm")["status"] == "success"
    after = _frame_mean(sim)
    assert after == pytest.approx(before, rel=0.05), f"frame mean {before:.2f} -> {after:.2f} after remove_camera"


def test_remove_camera_leaves_derived_transforms_populated(sim) -> None:
    """The direct invariant, independent of any pixel threshold."""
    assert sim.add_camera(name="cm", position=[1, 1, 1], target=[0, 0, 0])["status"] == "success"
    assert sim.remove_camera(name="cm")["status"] == "success"

    model, data = sim.mj_model, sim.mj_data
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "default")
    assert cam >= 0
    assert np.any(np.asarray(data.cam_xpos[cam]) != 0.0), "cam_xpos still zeroed"
    assert np.any(np.asarray(data.cam_xmat[cam]) != 0.0), "cam_xmat still zeroed"
    for i in range(model.nlight):
        assert np.any(np.asarray(data.light_xpos[i]) != 0.0), f"light {i} xpos still zeroed"


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda s: s.add_object(name="o", shape="box", size=[0.05] * 3, position=[0.4, 0, 0.3], mass=0.2),
            id="add_object",
        ),
        pytest.param(lambda s: s.add_robot(name="b", data_config="panda", position=[1.5, 0, 0]), id="add_robot"),
        pytest.param(lambda s: s.add_camera(name="cm2", position=[1, 1, 1], target=[0, 0, 0]), id="add_camera"),
    ],
)
def test_every_mutation_keeps_the_frame_lit(sim, mutate) -> None:
    """The property the other recompile sites already had; pin it for all of them."""
    assert _frame_mean(sim) > _MIN_FRAME_MEAN
    assert mutate(sim)["status"] == "success"
    assert _frame_mean(sim) > _MIN_FRAME_MEAN


def test_full_rebuild_keeps_the_frame_lit(sim) -> None:
    assert sim.add_robot(name="b", data_config="panda", position=[1.5, 0, 0])["status"] == "success"
    assert sim.remove_robot(name="b")["status"] == "success"
    assert _frame_mean(sim) > _MIN_FRAME_MEAN


def test_patch_scene_mjcf_keeps_the_frame_lit(sim) -> None:
    assert (
        sim.add_object(name="o", shape="box", size=[0.05] * 3, position=[0.4, 0, 0.3], mass=0.2)["status"] == "success"
    )
    assert sim.patch_scene_mjcf(ops=[{"op": "set_body_pos", "name": "o", "pos": [0.5, 0, 0.4]}])["status"] == "success"
    assert _frame_mean(sim) > _MIN_FRAME_MEAN


def test_replace_scene_mjcf_keeps_the_frame_lit(sim) -> None:
    xml = (
        '<mujoco><worldbody><light pos="0 0 3"/>'
        '<geom name="ground" type="plane" size="5 5 .1"/>'
        '<body name="z" pos="0 0 .3"><freejoint/>'
        '<geom type="sphere" size=".1" rgba="0 1 0 1"/></body></worldbody></mujoco>'
    )
    assert sim.replace_scene_mjcf(xml=xml)["status"] == "success"
    assert _frame_mean(sim) > _MIN_FRAME_MEAN


def test_remove_object_keeps_the_frame_lit(sim) -> None:
    assert (
        sim.add_object(name="o", shape="box", size=[0.05] * 3, position=[0.4, 0, 0.3], mass=0.2)["status"] == "success"
    )
    assert sim.remove_object(name="o")["status"] == "success"
    assert _frame_mean(sim) > _MIN_FRAME_MEAN
