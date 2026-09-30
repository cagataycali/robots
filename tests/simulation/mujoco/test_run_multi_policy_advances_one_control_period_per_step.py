"""A ``run_multi_policy`` step advances one control period of sim time, as ``run_policy`` does.

The synchronized loop stepped physics ONCE per control step: 2 ms on the
default MuJoCo dt. At 50 Hz a 50-step rollout covered 0.1 s of sim time where
``run_policy`` with the same arguments covers 1.0 s. Every position servo got
2 ms toward each target instead of the 20 ms period (the under-integration
``PolicyRunner._control_substeps`` exists to prevent), and a recording through
this loop - the documented multi-robot path - stamped frames 1/fps apart while
the sim had moved 2 ms. Isaac had the same shape (one 1/120 s tick per step);
``tests/simulation/isaac/test_run_multi_policy_recording.py`` pins its count.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.policies.mock import MockPolicy  # noqa: E402
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine  # noqa: E402


@pytest.fixture
def sim():
    engine = MuJoCoSimEngine(tool_name="multi_period", mesh=False)
    assert engine.create_world()["status"] == "success"
    assert engine.add_robot(name="so100", data_config="so100")["status"] == "success"
    try:
        yield engine
    finally:
        engine.cleanup()


def _advance(sim, run) -> float:
    start = sim._world._data.time
    result = run()
    assert result["status"] == "success", result
    return sim._world._data.time - start


@pytest.mark.parametrize("hz", [50.0, 25.0])
def test_a_synchronized_step_covers_the_same_sim_time_as_run_policy(sim, hz):
    multi = _advance(
        sim,
        lambda: sim.run_multi_policy(
            policies={"so100": MockPolicy()}, instructions="x", n_steps=20, control_frequency=hz
        ),
    )
    single = _advance(
        sim, lambda: sim.run_policy(robot_name="so100", policy_provider="mock", n_steps=20, control_frequency=hz)
    )

    assert multi == pytest.approx(20 / hz, abs=1e-9), f"20 steps at {hz} Hz advanced {multi:.4f} s"
    assert multi == pytest.approx(single, abs=1e-9)


class _Hold(MockPolicy):
    """Commands one fixed target for every joint, so tracking is measurable."""

    async def get_actions(self, observation, instruction, **kwargs):
        return [{key: 0.6 for key in self.robot_state_keys}]


def test_a_servo_reaches_its_target_within_the_rollout(sim):
    """Measured: at 2 ms per step the elbow reached 0.040 of a 0.6 rad target; one period per step, 0.544."""
    sim.run_multi_policy(policies={"so100": _Hold()}, instructions="x", n_steps=25, control_frequency=50.0)

    elbow = next(k for k in sim.robot_action_keys("so100") if k.lower().startswith("elbow"))
    state = sim.get_observation("so100")
    assert state[elbow] > 0.5, state[elbow]
