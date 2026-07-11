"""End-to-end: every shipped built-in locomotion benchmark runs on its real robot.

:func:`~strands_robots.simulation.builtin_benchmarks.register_builtin_benchmarks`
ships canonical velocity-tracking locomotion benchmarks (``go2_walk_forward`` and
the humanoid ``g1_walk_forward`` / ``t1_walk_forward``) whose success/failure/
dense-reward clauses read the embodiment-agnostic floating-base surface
(``base_pos`` / ``base_quat`` / ``base_lin_vel`` / ``base_ang_vel``) from
``get_observation``. The unit tests in
``tests/simulation/test_builtin_benchmarks.py`` drive the compiled predicates on
a SYNTHETIC inline freejoint MJCF at known poses - fast and GL-free, but they
never load the real ``default_robot`` each spec targets. So nothing guards the
integration seam between a shipped spec and the robot it names:

  * a ``robot_descriptions`` rename / removal or a broken asset would make a
    shipped benchmark dead-on-arrival (``add_robot`` raises), yet every synthetic
    unit test would still pass;
  * a regression in the floating-base observation surfacing (the base signals
    the DSL reads) would silently zero the reward / never fire the predicates;
  * a drift in a robot's spawn stance below its ``base_below_z`` collapse
    threshold (or a topple past ``base_tipped``) would make the benchmark FAIL
    every episode at ``t=0``, and above its ``base_beyond_x`` line would make it
    SUCCEED spuriously - a standing-spawn regression no synthetic-pose test sees.

This module closes that seam: for EVERY shipped built-in spec it loads the real
``default_robot`` in MuJoCo (auto-downloading the ``robot_descriptions`` asset),
asserts the floating-base observation surfaces, asserts the standing spawn
neither trips the benchmark's ``failure`` predicates nor satisfies its
``success`` predicate, and runs the whole benchmark harness end-to-end via
``evaluate_benchmark`` (a mock policy driving real physics, exactly as
``test_lekiwi_sim`` smoke-tests a real robot) so the reward composition + episode
scoring are exercised on the actual embodiment.

The cases are derived from :func:`builtin_benchmark_specs` so any benchmark added
to the shipped set is covered automatically - no hardcoded robot list to drift.
Network + MuJoCo + GPU-render integration (collected only via ``hatch run
test-integ``), so it is deliberately out of the GL-free unit suite.
"""

from __future__ import annotations

import copy
import math
import os
from typing import Any

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots.registry import has_sim  # noqa: E402 - after MUJOCO_GL default
from strands_robots.simulation import create_simulation  # noqa: E402
from strands_robots.simulation.benchmark import (  # noqa: E402
    get_benchmark,
    register_benchmark,
    unregister_benchmark,
)
from strands_robots.simulation.benchmark_spec import DeclarativeBenchmark  # noqa: E402
from strands_robots.simulation.builtin_benchmarks import builtin_benchmark_specs  # noqa: E402

# A short episode budget: enough to exercise the full control loop + scoring on
# the real robot without a 1000-step rollout. The spawn-contract assertions
# below cover the t=0 predicate behaviour; this only proves the harness runs.
_SMOKE_STEPS = 4

# Shipped benchmark names, derived from the module so a newly-added built-in is
# covered automatically (no hardcoded list to drift out of sync with the specs).
_SHIPPED_SPECS: dict[str, dict[str, Any]] = builtin_benchmark_specs()
_SHIPPED_NAMES: list[str] = sorted(_SHIPPED_SPECS)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Keep the module-global benchmark registry clean around each test."""
    for _n in _SHIPPED_NAMES:
        unregister_benchmark(_n)
    yield
    for _n in _SHIPPED_NAMES:
        unregister_benchmark(_n)


def _base_signal_is_vec(value: Any, length: int) -> bool:
    return isinstance(value, list) and len(value) == length and all(isinstance(v, (int, float)) for v in value)


@pytest.mark.parametrize("bench_name", _SHIPPED_NAMES)
def test_shipped_benchmark_runs_on_its_real_robot(bench_name: str) -> None:
    """A shipped built-in benchmark loads its real robot, surfaces the base state
    its DSL reads, holds the standing-spawn contract, and runs end-to-end."""
    spec = _SHIPPED_SPECS[bench_name]
    robot = spec["default_robot"]

    # Structural: the spec's default robot is a supported, sim-resolvable robot.
    assert robot in spec["supported_robots"], f"{bench_name}: default_robot not in supported_robots"
    assert has_sim(robot), f"{bench_name}: default_robot '{robot}' is not simulatable"

    # Register a copy trimmed to a short episode so the end-to-end run is fast;
    # the trim does not affect the state-reading predicates asserted below.
    trimmed = copy.deepcopy(spec)
    trimmed["max_steps"] = _SMOKE_STEPS
    register_benchmark(bench_name, DeclarativeBenchmark.from_dict(trimmed))

    sim = create_simulation(backend="mujoco")
    sim.create_world(ground_plane=True)
    sim.add_robot(robot)  # auto-downloads the robot_descriptions asset
    try:
        # The floating-base observation surface the base_* DSL reads must exist.
        obs = sim.get_observation(skip_images=True)
        assert _base_signal_is_vec(obs.get("base_pos"), 3), f"{robot}: base_pos not surfaced"
        assert _base_signal_is_vec(obs.get("base_quat"), 4), f"{robot}: base_quat not surfaced"
        assert _base_signal_is_vec(obs.get("base_lin_vel"), 3), f"{robot}: base_lin_vel not surfaced"
        assert _base_signal_is_vec(obs.get("base_ang_vel"), 3), f"{robot}: base_ang_vel not surfaced"

        bench = get_benchmark(bench_name)
        assert bench is not None  # just registered above

        # Standing-spawn contract: a freshly-spawned, upright robot must NOT
        # already be "failed" (its spawn height/orientation is above the
        # base_below_z / base_tipped fall thresholds) nor already "succeeded"
        # (it has not yet walked past the base_beyond_x line). Either would make
        # the benchmark score at t=0 - the exact silent regression the synthetic
        # unit tests, which set poses by hand, cannot catch.
        assert bench.is_failure(sim) is False, f"{bench_name}: standing spawn spuriously trips a failure predicate"
        assert bench.is_success(sim) is False, f"{bench_name}: standing spawn spuriously satisfies success"

        # The dense-reward composition evaluates to a finite scalar on the real
        # robot (the reward terms resolve the base signals, not a degenerate 0).
        info = bench.on_step(sim, {}, {})
        assert math.isfinite(info.reward), f"{bench_name}: dense reward is not finite"
        assert info.done is False

        # Full harness end-to-end: build a policy, run the control loop, score
        # the episode against the compiled spec on the real embodiment.
        res = sim.evaluate_benchmark(bench_name, policy_provider="mock", n_episodes=1)
        assert res["status"] == "success", res
        payload = next(c["json"] for c in res["content"] if "json" in c)
        assert payload["episodes_completed"] == 1
        assert payload["success_measured"] is True
        assert math.isfinite(payload["avg_reward"]), f"{bench_name}: avg_reward is not finite"
    finally:
        sim.destroy()
