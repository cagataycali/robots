"""reset() must leave forwarded (render-ready) derived kinematics.

``mujoco.mj_resetData`` restores ``qpos``/``qvel``/``ctrl`` but zeroes every
derived Cartesian quantity (``xpos``, ``site_xpos``, ``geom_xpos``,
``cam_xpos``, ...) until the next ``mj_step``/``mj_forward``. The eval loop
(:meth:`Simulation.eval_policy`) calls ``get_observation()`` immediately after
``reset()`` and before the first ``send_action``, so without a forward the
policy's first inference of every episode would receive a degenerate camera
frame with all geometry collapsed at the origin and a zeroed Cartesian state.

These tests pin that ``reset()`` forwards, so:
  * derived body positions are valid immediately after ``reset()`` (no manual
    ``mj_forward`` required), and
  * a camera observation captured right after ``reset()`` is identical to one
    captured after an explicit ``mj_forward`` (i.e. it is not degenerate).
"""

import numpy as np
import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.backend import _can_render  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

requires_gl = pytest.mark.skipif(
    not _can_render(),
    reason="No OpenGL context available (headless without EGL/OSMesa)",
)


@pytest.fixture
def sim():
    s = Simulation(tool_name="test_reset_forward", mesh=False)
    yield s
    s.cleanup()


def _body_xpos(sim: Simulation, name: str) -> np.ndarray:
    assert sim._world is not None, "world not created"
    model, data = sim._world._model, sim._world._data
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
    assert bid >= 0, f"body {name!r} not found"
    return np.asarray(data.xpos[bid]).copy()


def test_reset_forwards_derived_body_positions(sim):
    """After reset(), body Cartesian positions are populated WITHOUT a manual
    forward. Pre-fix (mj_resetData only) they read [0, 0, 0]."""
    assert sim.create_world(gravity=[0, 0, -9.81])["status"] == "success"
    assert sim.add_robot("so101")["status"] == "success"
    # A free object spawned well away from the origin is an unambiguous probe:
    # if reset() forwards, its body xpos lands at the declared spawn; if not,
    # it collapses to the origin.
    assert (
        sim.add_object(
            name="cube",
            shape="box",
            position=[0.3, 0.0, 0.15],
            size=[0.05, 0.05, 0.05],
            color=[1, 0, 0, 1],
            mass=0.05,
        )["status"]
        == "success"
    )

    sim.step(n_steps=50)
    assert sim.reset()["status"] == "success"

    # Read directly from mj_data with NO intervening mj_forward/mj_step.
    cube_xpos = _body_xpos(sim, "cube")
    assert np.linalg.norm(cube_xpos) > 0.05, (
        f"cube body collapsed to origin after reset(): {cube_xpos} - reset() did not forward derived kinematics"
    )
    # x/y are exactly the declared spawn (a free body's rest z settles later,
    # but the reset pose is the un-simulated spawn pose).
    np.testing.assert_allclose(cube_xpos[:2], [0.3, 0.0], atol=1e-6)

    # Sanity: values match an explicit forward (reset() did the same work).
    mj.mj_forward(sim._world._model, sim._world._data)
    np.testing.assert_allclose(_body_xpos(sim, "cube"), cube_xpos, atol=1e-9)


@requires_gl
def test_reset_leaves_render_ready_observation(sim):
    """A camera observation captured right after reset() is identical to one
    captured after an explicit mj_forward - i.e. not a collapsed-geometry
    frame. Pre-fix the two differ substantially (geometry at the origin)."""
    assert sim.create_world(gravity=[0, 0, -9.81])["status"] == "success"
    assert sim.add_robot("so101")["status"] == "success"
    assert (
        sim.add_object(
            name="cube",
            shape="box",
            position=[0.3, 0.0, 0.05],
            size=[0.05, 0.05, 0.05],
            color=[1, 0, 0, 1],
            mass=0.05,
        )["status"]
        == "success"
    )
    assert (
        sim.add_camera(
            name="camera1",
            position=[0.55, 0.0, 0.35],
            target=[0.2, 0.0, 0.05],
            fov=58.0,
            width=128,
            height=128,
        )["status"]
        == "success"
    )

    sim.step(n_steps=50)
    assert sim.reset()["status"] == "success"

    # First observation the policy would see - captured with NO manual forward.
    obs_after_reset = sim.get_observation(robot_name="so101")["camera1"].copy()
    # Now force a forward and re-capture: this is the "correct" frame.
    mj.mj_forward(sim._world._model, sim._world._data)
    obs_after_forward = sim.get_observation(robot_name="so101")["camera1"].copy()

    assert obs_after_reset.shape == obs_after_forward.shape
    np.testing.assert_array_equal(
        obs_after_reset,
        obs_after_forward,
        err_msg="post-reset observation differs from forwarded frame - "
        "reset() left a degenerate (collapsed-geometry) render",
    )
    # And the frame is not a single flat colour (real geometry is visible).
    assert len(np.unique(obs_after_reset.reshape(-1, 3), axis=0)) > 50
