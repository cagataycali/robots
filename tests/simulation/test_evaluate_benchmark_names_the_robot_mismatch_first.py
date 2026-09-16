"""``evaluate_benchmark`` names a benchmark written for another robot FIRST.

The runner has always refused a robot outside a spec's ``supported_robots``
(``BenchmarkCompatibilityError`` before episode 1), but ``evaluate_benchmark``
probes the spec's clauses before it hands over to the runner, and for a
fixed-base arm asked to run a locomotion benchmark the probe fired first.
Measured on ``Robot("so101", mode="sim")`` + the built-in ``g1_walk_forward``::

    evaluate_benchmark: benchmark 'g1_walk_forward' (success / failure /
    dense_reward) arms a base_* predicate on ['<the sole robot>'], but
    ['<the sole robot>'] has no floating base - a fixed-base arm reports no
    base_pos/base_quat. ...

A lecture about floating bases, with a placeholder where the robot's name
goes, when the cause is that the benchmark is written for ``unitree_g1``. The
same mismatch on a floating-base robot (a G1 asked to run ``go2_walk_forward``)
reached the runner and said "benchmark DeclarativeBenchmark supports
['unitree_go2']" - the class every declarative benchmark shares, not which one.

Pinned here: the mismatch is the first thing said, by benchmark id and robot
model, with the two ways out; a matching robot is untouched; an any-robot
benchmark (empty ``supported_robots``) still reaches the probe; the probe
spells the resolved robot's name where it used to say ``<the sole robot>``;
and the runner's own refusal - and the summary line a finished eval opens
with - name the benchmark by its registered id.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.benchmark import _BENCHMARK_REGISTRY, register_benchmark
from strands_robots.simulation.benchmark_spec import DeclarativeBenchmark
from strands_robots.simulation.mujoco.simulation import Simulation


@pytest.fixture(autouse=True)
def _clean_registry():
    """House isolation: the specs below are registered globally, so they are
    kept out of whatever runs next in the same process."""
    snapshot = dict(_BENCHMARK_REGISTRY)
    _BENCHMARK_REGISTRY.clear()
    yield
    _BENCHMARK_REGISTRY.clear()
    _BENCHMARK_REGISTRY.update(snapshot)


@pytest.fixture
def arm():
    """so100 - a FIXED-base arm, the shape the probe misreports."""
    s = Simulation(tool_name="mismatch_test", mesh=False)
    s.create_world()
    s.add_robot(name="alice", data_config="so100")
    s.step(5)
    yield s
    s.cleanup()


def _register(name: str, supported: list[str], default: str, **clauses: object) -> str:
    spec = {
        "name": name,
        "max_steps": 10,
        "supported_robots": supported,
        "default_robot": default,
        "success": {"all": [{"predicate": "base_beyond_x", "x": 0.1}]},
        **clauses,
    }
    register_benchmark(name, DeclarativeBenchmark.from_dict(spec))
    return name


def _text(result: dict) -> str:
    return result["content"][0]["text"]


class TestTheMismatchIsNamedFirst:
    def test_a_locomotion_benchmark_on_a_fixed_base_arm_names_the_robot_it_is_for(self, arm):
        name = _register("walk-for-g1", ["unitree_g1"], "unitree_g1")
        result = arm.evaluate_benchmark(name, n_episodes=1)
        assert result["status"] == "error", result
        text = _text(result)
        assert text.startswith("evaluate_benchmark: benchmark 'walk-for-g1' is written for ['unitree_g1']"), text
        assert "robot 'alice' in this scene is a 'so100'" in text, text
        # The two ways out, both concrete.
        assert "Robot('unitree_g1', mode='sim')" in text, text
        assert "list_benchmarks" in text, text
        # Not the floating-base lecture, and no placeholder.
        assert "no floating base" not in text, text
        assert "<the sole robot>" not in text, text

    def test_the_refusal_lands_before_a_policy_is_built(self, arm, monkeypatch):
        import strands_robots.policies as policies_mod

        def _explode(*_a: object, **_k: object) -> None:
            raise AssertionError("create_policy must not run for a benchmark written for another robot")

        monkeypatch.setattr(policies_mod, "create_policy", _explode)
        name = _register("walk-for-go2", ["unitree_go2"], "unitree_go2")
        assert arm.evaluate_benchmark(name, n_episodes=1)["status"] == "error"

    def test_a_matching_robot_passes_the_check(self, arm):
        """so100 is in the list, so the mismatch check stands aside - the probe then
        speaks, because a base predicate on a fixed-base arm still cannot fire."""
        name = _register("for-so100", ["so100"], "so100")
        result = arm.evaluate_benchmark(name, n_episodes=1)
        assert result["status"] == "error", result
        text = _text(result)
        assert "is written for" not in text, text
        assert "no floating base" in text, text

    def test_an_any_robot_benchmark_skips_the_check(self, arm):
        name = _register("for-anyone", [], "so100")
        result = arm.evaluate_benchmark(name, n_episodes=1)
        text = _text(result)
        assert "is written for" not in text, text


class TestTheProbeSpellsTheRobot:
    def test_the_sole_robot_is_named_not_placeheld(self, arm):
        name = _register("for-so100-base", ["so100"], "so100")
        text = _text(arm.evaluate_benchmark(name, n_episodes=1))
        assert "'alice'" in text, text
        assert "<the sole robot>" not in text, text


class TestTheRunnerNamesTheBenchmark:
    def test_the_compatibility_refusal_carries_the_registered_id(self, arm, monkeypatch):
        """Route the same mismatch through the runner (as a spec without the
        up-front check would) and read the id, not the class name."""
        from strands_robots.simulation.policy_runner import PolicyRunner

        name = _register("walk-for-t1", ["booster_t1"], "booster_t1")
        from strands_robots.simulation.benchmark import get_benchmark

        spec = get_benchmark(name)
        policy = arm._build_policy("evaluate_benchmark", "mock", None)
        assert not isinstance(policy, dict), policy
        policy.set_robot_state_keys(arm.robot_action_keys("alice"))
        result = PolicyRunner(arm).evaluate("alice", policy, instruction="", n_episodes=1, spec=spec)
        assert result["status"] == "error", result
        text = _text(result)
        assert "benchmark walk-for-t1 supports ['booster_t1']" in text, text
        assert "DeclarativeBenchmark supports" not in text, text

    def test_the_summary_line_of_a_finished_eval_carries_the_registered_id(self, arm):
        """The line every completed eval opens with. Read on a run that finishes
        rather than one that refuses, because the id has to survive the whole
        eval - a joint clause the arm can actually report keeps the probe quiet."""
        name = _register(
            "joint-bench-for-so100",
            ["so100"],
            "so100",
            success={"all": [{"predicate": "joint_above", "joint": "Rotation", "value": 99.0}]},
        )
        result = arm.evaluate_benchmark(name, n_episodes=1)
        assert result["status"] == "success", result
        text = _text(result)
        assert text.startswith(f"Benchmark: {name} | policy"), text
        assert "DeclarativeBenchmark" not in text, text
