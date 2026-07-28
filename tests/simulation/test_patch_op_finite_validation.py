"""Regression tests: a patch op with a non-finite pose is refused, not applied.

MuJoCo's compiler does NOT reject a ``nan``/``inf`` body pose, so an LLM-supplied
one went straight into the model while ``patch_scene_mjcf`` reported success. The
corruption then spread through the whole world on the first step:

    patch add_body pos=[nan, 0, 0.3]    status="success"
    model.body_pos                      [nan, 0, 0.3]
    data.xpos                           non-finite
    step(n_steps=20)                    status="success"
    data.qpos / data.qvel               non-finite
    get_observation("a")["joint1"]       nan

with MuJoCo only muttering ``Nan, Inf or huge value in QPOS`` to stderr. Every call
in that chain reported success, so an agent had no way to know the world was dead.

An all-zero quaternion is the same defect one step removed: it passes a finiteness
check but is not a rotation, and MuJoCo normalises it into ``nan``.

Every numeric field of every op now goes through ``_finite_vec`` / ``_finite_quat``
BEFORE the spec is touched, so the batch is rejected atomically - matching the
up-front validation ``send_action`` and ``set_joint_positions`` already do.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

_NON_FINITE = [float("nan"), float("inf"), -float("inf")]


@pytest.fixture
def sim():
    s = Simulation(tool_name="patch_op_finite_validation", mesh=False)
    s.create_world()
    assert s.add_robot(name="a", data_config="panda")["status"] == "success"
    assert s.add_object(name="c", shape="box", size=[0.05] * 3, position=[0.4, 0, 0.3], mass=0.2)["status"] == "success"
    yield s
    s.destroy()


def _model_is_finite(sim) -> bool:
    model, data = sim.mj_model, sim.mj_data
    return bool(
        np.all(np.isfinite(model.body_pos))
        and np.all(np.isfinite(model.body_quat))
        and np.all(np.isfinite(data.qpos))
        and np.all(np.isfinite(data.xpos))
    )


@pytest.mark.parametrize("bad", _NON_FINITE)
def test_add_body_rejects_a_non_finite_pos(sim, bad) -> None:
    result = sim.patch_scene_mjcf(ops=[{"op": "add_body", "parent": "world", "name": "b", "pos": [bad, 0, 0.3]}])
    assert result["status"] == "error"
    assert "not finite" in result["content"][0]["text"]
    assert _model_is_finite(sim)


@pytest.mark.parametrize("bad", _NON_FINITE)
def test_add_body_rejects_a_non_finite_quat(sim, bad) -> None:
    ops = [{"op": "add_body", "parent": "world", "name": "b", "pos": [0, 0, 0.3], "quat": [bad, 0, 0, 0]}]
    assert sim.patch_scene_mjcf(ops=ops)["status"] == "error"
    assert _model_is_finite(sim)


def test_add_body_rejects_a_zero_norm_quat(sim) -> None:
    """Finite but not a rotation; MuJoCo normalises it into nan."""
    ops = [{"op": "add_body", "parent": "world", "name": "b", "pos": [0, 0, 0.3], "quat": [0, 0, 0, 0]}]
    result = sim.patch_scene_mjcf(ops=ops)
    assert result["status"] == "error"
    assert "norm" in result["content"][0]["text"]
    assert _model_is_finite(sim)


@pytest.mark.parametrize("bad", _NON_FINITE)
def test_set_body_pos_rejects_a_non_finite_pos(sim, bad) -> None:
    assert sim.patch_scene_mjcf(ops=[{"op": "set_body_pos", "name": "c", "pos": [bad, 0, 0.3]}])["status"] == "error"
    assert _model_is_finite(sim)


def test_set_body_quat_rejects_a_non_finite_quat(sim) -> None:
    ops = [{"op": "set_body_quat", "name": "c", "quat": [float("nan"), 0, 0, 0]}]
    assert sim.patch_scene_mjcf(ops=ops)["status"] == "error"
    assert _model_is_finite(sim)


def test_add_geom_and_add_site_reject_a_non_finite_pos(sim) -> None:
    ops = [
        {"op": "add_body", "parent": "world", "name": "b", "pos": [1, 0, 0.3]},
        {"op": "add_geom", "body": "b", "type": "sphere", "size": [0.05], "pos": [float("inf"), 0, 0]},
    ]
    assert sim.patch_scene_mjcf(ops=ops)["status"] == "error"
    assert (
        sim.patch_scene_mjcf(ops=[{"op": "add_site", "body": "c", "name": "s", "pos": [float("nan"), 0, 0]}])["status"]
        == "error"
    )
    assert _model_is_finite(sim)


def test_a_wrong_length_vector_is_rejected(sim) -> None:
    result = sim.patch_scene_mjcf(ops=[{"op": "set_body_pos", "name": "c", "pos": [0.5, 0.2]}])
    assert result["status"] == "error"
    assert "components" in result["content"][0]["text"]


def test_a_non_numeric_component_is_rejected(sim) -> None:
    result = sim.patch_scene_mjcf(ops=[{"op": "set_body_pos", "name": "c", "pos": ["x", 0, 0.3]}])
    assert result["status"] == "error"
    assert "not a number" in result["content"][0]["text"]


def test_the_world_stays_integrable_after_a_rejected_patch(sim) -> None:
    """The user-visible symptom: nan reaching qpos/qvel and every observation."""
    for bad in _NON_FINITE:
        sim.patch_scene_mjcf(ops=[{"op": "add_body", "parent": "world", "name": "b", "pos": [bad, 0, 0.3]}])
    assert sim.step(n_steps=50)["status"] == "success"
    assert _model_is_finite(sim)
    observation = sim.get_observation("a", skip_images=True)
    assert all(np.isfinite(float(v)) for v in observation.values() if isinstance(v, (int, float)))


def test_a_rejected_batch_applies_nothing(sim) -> None:
    """All-or-nothing: the valid first op must not survive the bad second one."""
    before = int(sim.mj_model.nbody)
    ops = [
        {"op": "add_body", "parent": "world", "name": "good", "pos": [1, 0, 0.3]},
        {"op": "add_body", "parent": "world", "name": "bad", "pos": [float("nan"), 0, 0.3]},
    ]
    assert sim.patch_scene_mjcf(ops=ops)["status"] == "error"
    assert int(sim.mj_model.nbody) == before
    assert mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, "good") < 0


def test_valid_poses_still_apply(sim) -> None:
    """Guard against the fix degenerating into 'reject everything'."""
    ops = [
        {"op": "add_body", "parent": "world", "name": "b", "pos": [1.0, 0.5, 0.3], "quat": [0.7071, 0, 0.7071, 0]},
        {"op": "add_geom", "body": "b", "type": "sphere", "size": [0.05], "pos": [0.0, 0.0, 0.01]},
        {"op": "add_site", "body": "c", "name": "s", "pos": [0.0, 0.0, 0.05]},
        {"op": "set_body_pos", "name": "c", "pos": [0.6, 0.1, 0.4]},
    ]
    assert sim.patch_scene_mjcf(ops=ops)["status"] == "success"
    model = sim.mj_model
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "b")
    assert body >= 0
    assert [float(v) for v in model.body_pos[body]] == pytest.approx([1.0, 0.5, 0.3])
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "s") >= 0
    assert _model_is_finite(sim)
