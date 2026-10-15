"""Simulation("mujoco", physics="newton"): the same MuJoCo world stepped by MuJoCo-Warp on the GPU, ``num_envs`` copies.

Newton is a solver choice inside a world, never a third engine. Runs where mujoco_warp + warp are installed (a CUDA
box); everywhere else the GPU tests skip with the reason and only the refusals run.
"""

from __future__ import annotations

import importlib.util
import time

import pytest

from strands_robots import Robot, Simulation

has_warp = importlib.util.find_spec("mujoco_warp") is not None and importlib.util.find_spec("warp") is not None
gpu = pytest.mark.skipif(not has_warp, reason="mujoco_warp/warp not installed (needs a CUDA box)")


def test_many_envs_need_newton_physics():
    with pytest.raises(ValueError, match="num_envs=4 needs physics='newton'"):
        Simulation(num_envs=4)


@pytest.mark.skipif(has_warp, reason="checks the install hint, only where mujoco_warp is missing")
def test_newton_without_mujoco_warp_says_how_to_install():
    with pytest.raises(ImportError, match=r"physics='newton' needs mujoco_warp \+ warp"):
        Simulation(physics="newton")


@gpu
@pytest.mark.parametrize("num_envs", [1, 1024])
def test_newton_steps_two_robots_in_one_world(num_envs):
    sim = Simulation("mujoco", physics="newton", num_envs=num_envs)
    try:
        arm = Robot("so101", sim=sim)
        g1 = Robot("g1", sim=sim, position=[1, 0, 0])
        assert sim.physics.name == "newton" and sim.envs == num_envs and arm.sim is g1.sim is sim
        sim.step(10)  # kernel compilation + graph capture
        n = 200
        t = time.perf_counter()
        sim.step(n)
        dt = time.perf_counter() - t
        key = arm.description.action_keys[0]
        t0 = sim.time
        for _ in range(50):
            arm.act({key: 0.3})
        assert sim.time > t0
        assert arm.observe()[key] == pytest.approx(0.3, abs=0.05)
        g1.set_pose(position=[2, 0, 0.8])
        assert g1.pose.position[0] == pytest.approx(2.0, abs=1e-3)
        print(
            f"\n[measured] newton num_envs={num_envs}: {n} steps in {dt * 1e3:.0f} ms = "
            f"{n * num_envs / dt:,.0f} env-steps/s ({n / dt:,.0f} steps/s)",
            flush=True,
        )
    finally:
        sim.close()
