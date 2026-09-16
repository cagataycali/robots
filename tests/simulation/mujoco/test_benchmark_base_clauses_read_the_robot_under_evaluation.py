"""Unnamed ``base_*`` clauses and the compatibility check read the robot under evaluation.

Benchmark and ``stop_when`` clauses default ``robot`` to "the sole robot", which
every base reader spelled as ``get_observation(robot_name=None)`` - the FIRST
registered robot in a multi-robot scene. ``evaluate_benchmark(
benchmark_name='go2_walk_forward', robot_name='go2')`` with an arm registered
first therefore probed the arm ("<the sole robot> has no floating base") and
refused; and had the probe passed, ``on_episode_start`` validated EVERY loaded
robot's data_config against ``supported_robots``, so the arm's presence refused
the quadruped benchmark too ("robot 'so101' has data_config='so101'"). With two
floating-base robots the clauses would have scored the wrong one silently.

Pinned: the three surfaces bind the resolved robot on ``sim.predicate_robot``
before probe and rollout; ``_bound_robot`` prefers an explicit name, then the
binding, and drops a stale binding; the compatibility check covers only the
bound robot; the arm-first scene runs the go2 benchmark on the go2 with a
non-zero dense reward; naming the arm is refused with the arm named as
"the robot under evaluation".
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.benchmark import BenchmarkCompatibilityError, get_benchmark
from strands_robots.simulation.builtin_benchmarks import register_builtin_benchmarks
from strands_robots.simulation.mujoco.simulation import Simulation
from strands_robots.simulation.predicates import _bound_robot


@pytest.fixture
def two_robot_sim():
    s = Simulation(tool_name="bench_bound_robot", mesh=False)
    s.create_world()
    assert s.add_robot(name="so101", data_config="so101")["status"] == "success"
    assert s.add_robot(name="go2", data_config="unitree_go2", position=[1.0, 0.0, 0.0])["status"] == "success"
    register_builtin_benchmarks()
    yield s
    s.cleanup(policy_stop_timeout=0.5)


def _text(result: dict) -> str:
    return "\n".join(c["text"] for c in result["content"] if isinstance(c, dict) and "text" in c)


def test_bound_robot_prefers_explicit_then_binding_then_drops_a_stale_one(two_robot_sim):
    sim = two_robot_sim
    assert _bound_robot(sim, "so101") == "so101"
    sim.bind_predicate_robot("go2")
    assert _bound_robot(sim, None) == "go2"
    assert _bound_robot(sim, "so101") == "so101"
    sim.bind_predicate_robot("gone")
    assert _bound_robot(sim, None) is None
    sim.bind_predicate_robot(None)
    assert _bound_robot(sim, None) is None


def test_quadruped_benchmark_runs_on_the_quadruped_with_an_arm_registered_first(two_robot_sim):
    r = two_robot_sim.evaluate_benchmark(
        "go2_walk_forward", robot_name="go2", policy_provider="mock", n_episodes=1, control_frequency=50.0
    )
    assert r["status"] == "success", _text(r)
    text = _text(r)
    assert "on 'go2'" in text, text
    payload = next(c["json"] for c in r["content"] if isinstance(c, dict) and "json" in c)
    # A dense reward built on base_* terms is non-zero only when they read a
    # floating base - the go2's, not the arm's.
    assert payload.get("avg_reward", payload.get("mean_reward", 0.0)) != 0.0, payload
    assert two_robot_sim.predicate_robot == "go2"


def test_naming_the_arm_is_refused_with_the_arm_named_as_the_robot_under_evaluation(two_robot_sim):
    r = two_robot_sim.evaluate_benchmark("go2_walk_forward", robot_name="so101", policy_provider="mock", n_episodes=1)
    assert r["status"] == "error"
    text = _text(r)
    assert "'so101' (the robot under evaluation)" in text, text
    assert "<the sole robot>" not in text
    assert "has no floating base" in text


def test_compatibility_check_covers_only_the_bound_robot(two_robot_sim):
    import random


    spec = get_benchmark("go2_walk_forward")
    two_robot_sim.bind_predicate_robot("go2")
    spec.on_episode_start(two_robot_sim, random.Random(0))  # the arm is a bystander
    two_robot_sim.bind_predicate_robot(None)
    with pytest.raises(BenchmarkCompatibilityError) as ei:
        spec.on_episode_start(two_robot_sim, random.Random(0))  # unbound: every robot is checked
    assert ei.value.robot_name == "so101"


def test_run_policy_binds_the_resolved_robot(two_robot_sim):
    r = two_robot_sim.run_policy(robot_name="go2", policy_provider="mock", duration=0.1, control_frequency=50.0)
    assert r["status"] == "success", _text(r)
    assert two_robot_sim.predicate_robot == "go2"
