"""GPU-gated: ``apply_force(torque=...)`` on Isaac spins a body the way the caller asked.

The contract is MuJoCo's ``xfrc_applied``: a right-handed world-frame torque.
Isaac negated the torque before handing it to PhysX, so every latched torque -
and every ``point=`` lever arm, which is folded into that torque - spun the
body backwards on Isaac Sim 6.0.1 and 6.1 alike (+0.01 z gave wz -2.96 rad/s).

Run with::

    STRANDS_GPU_TEST=1 hatch run test-integ tests_integ/simulation/test_isaac_apply_force_spins_the_right_way_gpu.py -m gpu -v
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("strands_robots.simulation.isaac")

_GPU_ENABLED = os.environ.get("STRANDS_GPU_TEST", "0") == "1"

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not _GPU_ENABLED, reason="Requires an NVIDIA GPU + Isaac Sim. Set STRANDS_GPU_TEST=1 to enable."
    ),
]


def _wz(sim) -> float:
    state = sim.get_body_state(body_name="box")
    return next(c["json"] for c in state["content"] if "json" in c)["angular_velocity"][2]


def test_a_torque_and_a_lever_arm_spin_a_cube_right_handed() -> None:
    from strands_robots.simulation.isaac import IsaacConfig, IsaacSimulation

    available, reason = IsaacSimulation.is_available()
    if not available:
        pytest.skip(f"Isaac Sim not available: {reason}")
    sim = IsaacSimulation(IsaacConfig(num_envs=1, headless=True))
    try:
        assert sim.create_world(gravity=[0.0, 0.0, 0.0])["status"] == "success"
        r = sim.add_object("box", shape="box", size=[0.1, 0.1, 0.1], position=[0.0, 0.0, 1.0], mass=1.0)
        assert r["status"] == "success", r
        sim.reset()

        # I_zz of a 1 kg, 10 cm cube is 1/600 kg m^2: +0.01 N m for 60 ticks
        # (0.5 s) is wz = +3.0 rad/s.
        assert sim.apply_force("box", torque=[0.0, 0.0, 0.01])["status"] == "success"
        sim.step(60)
        assert _wz(sim) == pytest.approx(3.0, rel=0.05)

        # A +y force 5 cm along +x of the centre is a +z torque of 0.05 N m.
        sim.reset()
        sim.apply_force("box", force=[0.0, 1.0, 0.0], point=[0.05, 0.0, 1.0])
        sim.step(60)
        assert _wz(sim) > 10.0
    finally:
        sim.destroy()
