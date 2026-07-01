"""randomize() must leave forwarded (render-ready) derived state.

Domain randomization mutates ``model`` arrays whose rendered/simulated effect
flows through ``data``:

  * ``randomize_lighting`` writes ``model.light_pos``, but the renderer reads
    light positions from ``data.light_xpos`` (populated by ``mj_forward`` from
    ``model.light_pos`` and the parent body kinematics). Jittering
    ``light_pos`` without a forward leaves ``light_xpos`` stale, so the light
    move is a silent visual no-op until some later ``mj_step`` - even though
    ``randomize()`` reports success ("Lighting: N lights randomized").
  * ``randomize_positions`` writes object ``qpos``; the resulting body
    ``xpos`` is only valid after a forward.

These tests pin that ``randomize()`` forwards after mutating, matching the
mutate-then-forward contract already used by ``reset()``, ``load_scene()`` and
``move_object()``.
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
    s = Simulation(tool_name="test_randomize_forward", mesh=False)
    assert s.create_world(gravity=[0, 0, -9.81])["status"] == "success"
    assert s.add_robot("so101")["status"] == "success"
    yield s
    s.cleanup()


def test_randomize_lighting_forwards_light_xpos(sim):
    """After randomize(randomize_lighting=True), data.light_xpos (what the
    renderer consumes) tracks the jittered model.light_pos WITHOUT a manual
    forward. Pre-fix the two diverge - the light move never reaches the render.
    """
    assert sim._world is not None
    model, data = sim._world._model, sim._world._data
    if model.nlight == 0:
        pytest.skip("scene has no lights")

    pos_before = np.asarray(model.light_pos).copy()
    result = sim.randomize(randomize_colors=False, randomize_lighting=True, seed=7)
    assert result["status"] == "success"

    # The jitter actually moved the lights in the model...
    assert not np.allclose(model.light_pos, pos_before), "lighting jitter did not move any light"
    # ...and the derived (rendered) light position was recomputed to match.
    assert np.allclose(model.light_pos, data.light_xpos), (
        "data.light_xpos is stale after randomize(); the light-position jitter "
        "is a silent visual no-op until the next mj_step"
    )


def test_randomize_positions_forwards_body_xpos(sim):
    """Consolidating the forward must not regress randomize_positions: a moved
    dynamic object's body xpos reflects the qpos noise immediately."""
    assert sim._world is not None
    assert (
        sim.add_object(
            name="cube",
            shape="box",
            position=[0.3, 0.0, 0.15],
            size=[0.05, 0.05, 0.05],
        )["status"]
        == "success"
    )
    # add_object recompiles the model; re-fetch model/data (the pre-add handles
    # are now stale and would not contain the new body).
    model, data = sim._world._model, sim._world._data
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "cube")
    assert bid >= 0
    xpos_before = np.asarray(data.xpos[bid]).copy()

    result = sim.randomize(
        randomize_colors=False,
        randomize_lighting=False,
        randomize_positions=True,
        position_noise=0.05,
        seed=3,
    )
    assert result["status"] == "success"
    # Body Cartesian position reflects the qpos perturbation (forward ran).
    assert not np.allclose(data.xpos[bid], xpos_before)


def test_randomize_no_flags_is_true_noop(sim):
    """A no-flag call changes nothing and does not run a spurious forward."""
    assert sim._world is not None
    data = sim._world._data
    light_xpos_before = np.asarray(data.light_xpos).copy()
    result = sim.randomize(
        randomize_colors=False,
        randomize_lighting=False,
        randomize_physics=False,
        randomize_positions=False,
    )
    assert result["status"] == "success"
    assert np.array_equal(data.light_xpos, light_xpos_before)


@requires_gl
def test_randomize_lighting_changes_render(sim):
    """Positive end-to-end smoke: randomize_lighting visibly relights the scene.

    This exercises the render path but is NOT the fail-before proof for the
    stale-``light_xpos`` bug: ``randomize_lighting`` also resamples
    ``light_diffuse``, which the renderer reads straight from ``model`` (no
    forward needed), so the frame changes pre-fix from the diffuse component
    alone. The clean regression is
    ``test_randomize_lighting_forwards_light_xpos`` above, which isolates the
    light-*position* component that only reaches the renderer after a forward.
    """
    before = sim.render(camera_name="default", width=256, height=256)
    assert before["status"] == "success"
    import imageio.v3 as iio

    img_before = iio.imread(before["content"][1]["image"]["source"]["bytes"]).astype(np.int32)

    assert sim.randomize(randomize_colors=False, randomize_lighting=True, seed=7)["status"] == "success"

    after = sim.render(camera_name="default", width=256, height=256)
    img_after = iio.imread(after["content"][1]["image"]["source"]["bytes"]).astype(np.int32)

    changed = int((np.abs(img_before - img_after).max(axis=2) > 8).sum())
    assert changed > 0, "randomize_lighting produced no rendered change (silent visual no-op)"
