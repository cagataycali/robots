"""End-to-end integration test: G1 walks forward using WBC ONNX policy.

Requires:
- onnxruntime
- huggingface_hub (to download nvidia/GR00T-WholeBodyControl)
- mujoco
- MUJOCO_GL=egl (headless rendering on GPU nodes)

Run with: MUJOCO_GL=egl pytest tests_integ/wbc/ -m wbc -v
"""

from __future__ import annotations

import os

import pytest

# Skip if dependencies are missing
pytest.importorskip("onnxruntime")
pytest.importorskip("mujoco")
pytest.importorskip("huggingface_hub")


@pytest.fixture(autouse=True)
def _mujoco_egl():
    """Ensure MUJOCO_GL=egl for headless rendering."""
    os.environ.setdefault("MUJOCO_GL", "egl")


@pytest.mark.wbc
@pytest.mark.integration
def test_g1_balance_30s():
    """G1 stays balanced for 30s with zero velocity command.

    Acceptance criteria: pelvis Z drift < 5cm from initial height.
    """
    from strands_robots.policies.wbc import WBCPolicy
    from strands_robots.policies.wbc.runner import run_wbc_policy
    from strands_robots.simulation.mujoco import Simulation

    sim = Simulation()
    sim.create_world()

    policy = WBCPolicy(
        checkpoint="nvidia/GR00T-WholeBodyControl",
        target_velocity=[0.0, 0.0, 0.0],
    )

    result = run_wbc_policy(
        sim=sim,
        robot_name="unitree_g1",
        policy=policy,
        duration=30.0,
        target_velocity=[0.0, 0.0, 0.0],
        fast_mode=True,
    )

    assert result["status"] == "success", f"WBC failed: {result}"
    metrics = result["metrics"]

    # Pelvis should stay near initial height (0.793m)
    assert not metrics["fell"], f"G1 fell! final_height={metrics['final_height']:.3f}m"
    height_drift = abs(metrics["final_height"] - 0.793)
    assert height_drift < 0.05, (
        f"Height drift too large: {height_drift:.3f}m (final={metrics['final_height']:.3f}m, expected ~0.793m)"
    )


@pytest.mark.wbc
@pytest.mark.integration
def test_g1_walk_forward():
    """G1 walks forward >1.5m in 10s with vx=0.5.

    This is the primary acceptance criterion for WBC integration.
    """
    from strands_robots.policies.wbc import WBCPolicy
    from strands_robots.policies.wbc.runner import run_wbc_policy
    from strands_robots.simulation.mujoco import Simulation

    sim = Simulation()
    sim.create_world()

    policy = WBCPolicy(
        checkpoint="nvidia/GR00T-WholeBodyControl",
        target_velocity=[0.5, 0.0, 0.0],
    )

    result = run_wbc_policy(
        sim=sim,
        robot_name="unitree_g1",
        policy=policy,
        duration=10.0,
        target_velocity=[0.5, 0.0, 0.0],
        fast_mode=True,
    )

    assert result["status"] == "success", f"WBC failed: {result}"
    metrics = result["metrics"]

    # G1 must not fall
    assert not metrics["fell"], f"G1 fell! final_height={metrics['final_height']:.3f}m"
    assert metrics["final_height"] > 0.5, f"G1 too low: {metrics['final_height']:.3f}m (threshold 0.5m)"

    # G1 must walk forward at least 1.5m in 10s
    assert metrics["distance_x"] > 1.5, (
        f"G1 didn't walk far enough: {metrics['distance_x']:.2f}m (need >1.5m in 10s at vx=0.5)"
    )


@pytest.mark.wbc
@pytest.mark.integration
def test_g1_walk_via_sim_run_policy():
    """Test the high-level sim.run_policy() API with WBC provider.

    Verifies the full user-facing API path:
        sim.run_policy(policy_provider='wbc', policy_config={...})
    """
    from strands_robots.simulation.mujoco import Simulation

    sim = Simulation()
    sim.create_world()
    # add_robot is not strictly needed for WBC (runner loads its own XML),
    # but we call it to verify the API doesn't break.
    sim.add_robot("unitree_g1", data_config="unitree_g1")

    result = sim.run_policy(
        robot_name="unitree_g1",
        policy_provider="wbc",
        policy_config={"checkpoint": "nvidia/GR00T-WholeBodyControl"},
        target_velocity=[0.5, 0.0, 0.0],
        duration=10.0,
    )

    assert result["status"] == "success", f"run_policy failed: {result}"
    metrics = result.get("metrics", {})
    if metrics:
        assert not metrics.get("fell", True), "G1 fell"
        assert metrics.get("distance_x", 0) > 1.0, f"Insufficient forward distance: {metrics.get('distance_x', 0):.2f}m"
