"""The shipped Go2 fall line is grounded in the robot's folded rest height.

A quadruped that gives up does not topple: its legs fold and the trunk stays
LEVEL and high. Measured on the shipped Unitree Go2 asset, a zero-torque fold
comes to rest at 0.203 m only ~3 deg off level, so ``base_tipped`` never fires
and a ``base_below_z`` line under that rest height is UNREACHABLE - a collapsed
robot is then scored as a healthy full-horizon episode. The three shipped Go2
specs carried a biped-style "roughly half the standing height" line (0.18 m)
derived from a stance figure (0.32 m) that is not the asset's: its own ``home``
keyframe stands at 0.27 m.

Pinned here against the REAL asset (not a hand-set base pose): the compiled
failure clause of all three Go2 specs stays quiet at the spawn stance and at the
keyframe stance, fires on the fold, and the fold never reaches the old line;
and each spec's ``base_height`` reward target equals the height the asset's own
keyframe declares, so the number stays grounded in the model rather than in a
comment.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

import mujoco  # noqa: E402

from strands_robots.simulation.benchmark import get_benchmark  # noqa: E402
from strands_robots.simulation.builtin_benchmarks import (  # noqa: E402
    builtin_benchmark_specs,
    register_builtin_benchmarks,
)
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

GO2_SPECS = ("go2_walk_forward", "go2_strafe_left", "go2_turn_left")
FOLD_STEPS = 1500  # the fold settles by ~500 steps; measured rest 0.2032 m
OLD_LINE = 0.18  # the unreachable line these specs shipped


@pytest.fixture(scope="module")
def go2_sim():
    """A real Unitree Go2 on flat ground, meshes off (the asset's collision hulls)."""
    sim = Simulation(tool_name="go2_fall_line", mesh=False)
    sim.create_world()
    assert sim.add_robot(name="go2", data_config="unitree_go2")["status"] == "success"
    register_builtin_benchmarks()
    yield sim
    sim.cleanup(policy_stop_timeout=0.5)


def _home_keyframe_base_z(model) -> float:
    """Base height the asset's own ``home`` keyframe declares (namespaced by robot name)."""
    for i in range(model.nkey):
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i) or "").endswith("/home"):
            return float(model.key_qpos[i][2])
    raise AssertionError("the unitree_go2 asset ships a 'home' keyframe")


def test_the_fall_clause_fires_on_a_folded_go2_and_not_on_either_stance(go2_sim):
    """A folded Go2 is a failure; neither standing pose is, and the old line misses the fold."""
    model, data = go2_sim.mj_model, go2_sim.mj_data
    benches = {name: get_benchmark(name) for name in GO2_SPECS}

    # 1. spawn stance: straight legs, feet down, 0.445 m - not a fall.
    spawn_z = float(data.qpos[2])
    assert spawn_z == pytest.approx(0.445, abs=1e-3)
    assert {n: b.is_failure(go2_sim) for n, b in benches.items()} == dict.fromkeys(GO2_SPECS, False)

    # 2. the asset's own home keyframe stance, 0.27 m - also not a fall.
    keyframe_z = _home_keyframe_base_z(model)
    assert keyframe_z == pytest.approx(0.27, abs=1e-3)
    data.qpos[2] = keyframe_z
    mujoco.mj_forward(model, data)
    assert {n: b.is_failure(go2_sim) for n, b in benches.items()} == dict.fromkeys(GO2_SPECS, False)

    # 3. the fold: torque motors hold nothing, so zero ctrl folds the legs.
    mujoco.mj_resetData(model, data)
    data.ctrl[:] = 0.0
    heights = []
    for _ in range(FOLD_STEPS):
        mujoco.mj_step(model, data)
        heights.append(float(data.qpos[2]))
    fold_z, tilt_w = heights[-1], abs(float(data.qpos[3]))
    assert fold_z == pytest.approx(0.203, abs=0.01), f"folded rest height moved: {fold_z}"
    assert tilt_w > 0.99, f"the fold should keep the trunk level, quat w={tilt_w}"
    assert min(heights) > OLD_LINE, f"the fold reached the old {OLD_LINE} m line: min {min(heights)}"
    assert {n: b.is_failure(go2_sim) for n, b in benches.items()} == dict.fromkeys(GO2_SPECS, True)


@pytest.mark.parametrize("spec_name", GO2_SPECS)
def test_the_base_height_reward_target_is_the_assets_home_keyframe(spec_name, go2_sim):
    """The height regularizer peaks at the stance the asset ships, not a written-in number."""
    terms = builtin_benchmark_specs()[spec_name]["dense_reward"]
    targets = [t["target"] for t in terms if t["predicate"] == "base_height"]
    assert targets == [pytest.approx(_home_keyframe_base_z(go2_sim.mj_model), abs=1e-3)]
