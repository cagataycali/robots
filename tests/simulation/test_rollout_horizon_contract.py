"""Regression tests for the rollout horizon and control-rate contract.

Three defects in the synchronous rollout path, all silent:

1. **Empty chunk hung forever.** ``while step_count < total_steps`` re-queried the
   policy, and an empty chunk made the inner ``for`` a no-op, so ``step_count``
   never advanced. Measured: 73 890 ``get_actions`` calls in 8 s, zero progress,
   no error and no timeout. The async branch already raised on this; the
   synchronous branch is the DEFAULT for every non-chunk-emitting policy.

2. **``int(duration * control_frequency)`` dropped a control step.** Binary float
   leaves an exact product a hair low (``0.58 * 50 == 28.999999999999996``), so
   truncation lost one step on 109 ``(duration, rate)`` pairs over a 0.01 grid at
   common rates. The ``n_steps`` branch was already fixed for exactly this; the
   ``duration`` branch was left on the lossy form.

3. **The policy was told an unrealisable control rate.** A control period must be
   a whole number of physics steps, so 60 Hz at ``dt=0.002`` actually runs at
   62.5 Hz. ``set_control_frequency`` was still handed the requested 60, making
   an RTC provider convert wall-clock latency into the wrong number of consumed
   action steps - the seam mis-blend that call exists to prevent.
"""

from __future__ import annotations

import threading

import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.policies.base import Policy  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


class _EmptyChunkPolicy(Policy):
    """Returns a well-formed but EMPTY action chunk on every query."""

    def __init__(self) -> None:
        self.calls = 0

    async def get_actions(self, observation_dict, instruction, **kwargs):  # noqa: ANN001, ANN003
        self.calls += 1
        return []

    def set_robot_state_keys(self, keys) -> None:  # noqa: ANN001
        pass

    @property
    def provider_name(self) -> str:
        return "empty_chunk"

    @property
    def requires_images(self) -> bool:
        return False


class _RateProbePolicy(Policy):
    """Records the control frequency the runner reports to the policy."""

    def __init__(self) -> None:
        self.control_frequency: float | None = None

    async def get_actions(self, observation_dict, instruction, **kwargs):  # noqa: ANN001, ANN003
        return [{}]

    def set_robot_state_keys(self, keys) -> None:  # noqa: ANN001
        pass

    def set_control_frequency(self, frequency: float) -> None:
        self.control_frequency = frequency

    @property
    def provider_name(self) -> str:
        return "rate_probe"

    @property
    def requires_images(self) -> bool:
        return False


@pytest.fixture
def sim():
    s = Simulation(tool_name="rollout_horizon", mesh=False)
    s.create_world()
    s.add_robot(name="panda")
    yield s
    s.destroy()


def test_empty_chunk_errors_instead_of_hanging(sim) -> None:
    """An empty chunk must terminate the rollout, not spin forever."""
    policy = _EmptyChunkPolicy()
    outcome: dict[str, object] = {}

    def _run() -> None:
        try:
            outcome["result"] = sim.run_policy(robot_name="panda", policy_object=policy, n_steps=10, async_rtc=False)
        except Exception as exc:  # noqa: BLE001 - the runner may surface it either way
            outcome["error"] = exc

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=30.0)

    assert not thread.is_alive(), "run_policy hung on an empty action chunk"
    # Pre-fix this reached tens of thousands of calls while step_count stayed 0.
    assert policy.calls < 10, f"policy was re-queried {policy.calls} times"
    if "result" in outcome:
        assert outcome["result"]["status"] == "error"  # type: ignore[index]
    else:
        assert "empty action chunk" in str(outcome["error"])


@pytest.mark.parametrize(
    ("duration", "control_frequency", "expected_steps"),
    [
        (0.58, 50, 29),  # 0.58 * 50 == 28.999999999999996 -> int() gave 28
        (1.16, 50, 58),
        (2.30, 50, 115),
        (2.00, 50, 100),  # already exact; must not regress
    ],
)
def test_duration_horizon_is_rounded_not_truncated(sim, duration, control_frequency, expected_steps) -> None:
    result = sim.run_policy(
        robot_name="panda",
        policy_provider="mock",
        duration=duration,
        control_frequency=control_frequency,
    )
    payload = next(block["json"] for block in result["content"] if "json" in block)
    assert payload["n_steps"] == expected_steps


def test_policy_is_told_the_achieved_control_rate(sim) -> None:
    """60 Hz is not realisable at dt=0.002; the policy must hear 62.5 Hz."""
    policy = _RateProbePolicy()
    sim.run_policy(robot_name="panda", policy_object=policy, n_steps=4, control_frequency=60)
    assert policy.control_frequency == pytest.approx(62.5)


def test_exact_control_rate_is_reported_verbatim(sim) -> None:
    """A realisable rate must pass through unchanged (no spurious adjustment)."""
    policy = _RateProbePolicy()
    sim.run_policy(robot_name="panda", policy_object=policy, n_steps=4, control_frequency=50)
    assert policy.control_frequency == pytest.approx(50.0)
