# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``get_world_point`` end to end on MuJoCo: pixels in, metric world xyz out.

The backend-agnostic math and error contract live in
``tests/simulation/test_get_world_point.py``; this file pins the parts that
need a real renderer and the real agent surface - that unprojecting a rendered
depth buffer recovers the pose the scene was built with, and that the action is
reachable through the tool dispatcher and advertised by ``describe()``.
"""

import numpy as np
import pytest

pytest.importorskip("mujoco")

from tests.simulation.mujoco._gl_probe import requires_gl

# Static crate: known pose, known extents, so the top face is exactly at
# CRATE_POS[2] + SIZE[2] / 2 and every unprojected point has a ground truth.
CRATE_POS = [0.2, 0.1, 0.15]
CRATE_SIZE = [0.2, 0.2, 0.3]
TOP_FACE_Z = CRATE_POS[2] + CRATE_SIZE[2] / 2
CAM_POS = [0.9, -0.9, 0.7]
WIDTH, HEIGHT = 320, 240


def _make_sim():
    from strands_robots.simulation import Simulation

    sim = Simulation(backend="mujoco", mesh=False)
    sim.create_world()
    sim.add_object(
        name="crate",
        shape="box",
        position=CRATE_POS,
        size=CRATE_SIZE,
        color=[0.9, 0.2, 0.2, 1.0],
        is_static=True,
    )
    sim.add_camera("front", position=CAM_POS, target=CRATE_POS, width=WIDTH, height=HEIGHT)
    sim.step(n_steps=3)
    return sim


def _project(sim, world_xyz: list[float]) -> list[int]:
    """Project a world point to its pixel using the camera's own parameters.

    Independent of the code under test: this is the forward pinhole projection,
    used only to CHOOSE which pixels to sample. The assertion oracle is the
    depth buffer plus the pose the scene was built with, so a wrong convention
    inside ``get_world_point`` cannot cancel out here - it would sample a pixel
    somewhere else in the image and recover a different point.
    """
    cam = sim.get_camera_params("front", width=WIDTH, height=HEIGHT)
    rotation = cam.T_world_cam[:3, :3]
    translation = cam.T_world_cam[:3, 3]
    p_cam = rotation.T @ (np.asarray(world_xyz, dtype=np.float64) - translation)
    forward = -p_cam[2]
    u = cam.K[0, 2] + cam.K[0, 0] * p_cam[0] / forward
    v = cam.K[1, 2] - cam.K[1, 1] * p_cam[1] / forward
    return [int(round(u)), int(round(v))]


@requires_gl
def test_world_point_recovers_the_pose_the_scene_was_built_with() -> None:
    """Five pixels on the crate's top face unproject back onto that face."""
    sim = _make_sim()
    try:
        center = _project(sim, [CRATE_POS[0], CRATE_POS[1], TOP_FACE_Z])
        pixels = [center] + [[center[0] + du, center[1] + dv] for du, dv in ((4, 0), (-4, 0), (0, 3), (0, -3))]
        # A corner pixel sees empty sky beyond the far clip.
        pixels.append([0, 0])

        result = sim.get_world_point("front", pixels=pixels)

        assert result["status"] == "success", result["content"][0]["text"]
        block = next(b["json"] for b in result["content"] if "json" in b)
        assert block["n_requested"] == 6
        assert block["n_valid"] == 5
        assert block["dropped"] == [5], "the sky pixel must be dropped, not unprojected"
        assert block["points"][5] is None

        point = block["point"]
        # Depth-buffer precision on a table-top scene is millimetre-scale.
        assert abs(point[2] - TOP_FACE_Z) < 0.01, f"z {point[2]} vs top face {TOP_FACE_Z}"
        assert abs(point[0] - CRATE_POS[0]) < 0.02
        assert abs(point[1] - CRATE_POS[1]) < 0.02
        # Agrees with the privileged pose read for the same body.
        truth = next(b["json"] for b in sim.get_body_state("crate")["content"] if "json" in b)
        assert abs(point[0] - truth["position"][0]) < 0.02
        assert abs(point[1] - truth["position"][1]) < 0.02
    finally:
        sim.destroy()


@requires_gl
def test_grounding_is_reachable_and_advertised_through_the_agent_surface() -> None:
    """The action dispatches, is in the tool spec enum, and describe() lists it."""
    import json
    from pathlib import Path

    import strands_robots.simulation.mujoco as mj_mod

    spec = json.loads((Path(mj_mod.__file__).parent / "tool_spec.json").read_text())
    assert "get_world_point" in spec["properties"]["action"]["enum"]
    assert "pixels" in spec["properties"]

    sim = _make_sim()
    try:
        assert "get_world_point" in sim.describe()["methods"]

        center = _project(sim, [CRATE_POS[0], CRATE_POS[1], TOP_FACE_Z])
        dispatched = sim._dispatch_action("get_world_point", {"camera_name": "front", "pixels": [center]})
        assert dispatched["status"] == "success"

        # A malformed pixel comes back through the envelope, never as a raise.
        bad = sim._dispatch_action("get_world_point", {"camera_name": "front", "pixels": [[1.5, 2]]})
        assert bad["status"] == "error"
        assert "whole image coordinate" in bad["content"][0]["text"]
    finally:
        sim.destroy()
