"""An episode already failed at reset is counted, not silently averaged in.

The exact mirror of the ``success`` clause case its sibling module pins, and the
last cell of that grid without a verdict. ``evaluate_benchmark`` samples a spec's
``failure`` clause only AFTER an applied action, so one that already holds at
reset ends every episode on its first step whatever the policy commands, and
``success_rate`` reports a hard ``0.0`` for it beside ``success_measured: true`` -
the same number an honest policy failure reports, reached without the policy.
Unlike a ``success`` clause, a ``failure`` clause is harmless when it can never
fire, so this is the only direction in which it corrupts a reported figure.

It costs more than the success mirror, because the eval loop reads ``is_failure``
BEFORE ``is_success``: the episode is scored a failure with the success criterion
never consulted, so a success the run had already reached on that step is
discarded. That is what
:class:`TestTheFailureClauseDiscardsASuccessTheRunReached` pins.

No compile-time or name-resolution check can see it. The clause is entirely
well-formed and every name in it resolves; whether it holds is a fact about the
initial state. It is a real configuration rather than a contrived one: a "the
object fell" height above where the object already rests does it, as does a
``base_below_z`` collapse line above the robot's spawned stance - which is why the
shipped humanoid benchmarks tune that line to each biped's own measured standing
height.

Reported rather than refused, and every figure is left as measured, the posture
``episodes_successful_at_reset`` already takes: domain randomisation draws initial
states per episode, so a partial count is a fact about those draws rather than a
broken spec, and the count is what distinguishes them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from strands_robots.simulation.policy_runner import failure_at_reset_warning

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.benchmark import (  # noqa: E402
    _BENCHMARK_REGISTRY,
    register_benchmark,
)
from strands_robots.simulation.benchmark_spec import DeclarativeBenchmark  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

#: A one-joint arm plus a cube hanging in the air with no floor under it, so the
#: cube falls under gravity alone. Its descent is the only scene fact the specs
#: below are written against, which keeps every threshold independent of what the
#: policy commands.
_SCENE_XML = """
<mujoco model="reset_failure_probe">
  <worldbody>
    <body name="base" pos="0 0 0">
      <joint name="j1" type="hinge" axis="0 0 1" range="-3 3"/>
      <geom name="link1" type="capsule" fromto="0 0 0 0 0 0.2" size="0.02"/>
    </body>
    <body name="cube" pos="0.5 0 0.5">
      <freejoint name="cube_free"/>
      <geom name="cube_geom" type="box" size="0.025 0.025 0.025" mass="0.05"/>
    </body>
  </worldbody>
  <actuator>
    <position name="a1" joint="j1" kp="10"/>
  </actuator>
</mujoco>
"""

#: Height the cube's origin starts at.
_CUBE_START_Z = 0.5

#: A "the cube fell" threshold ABOVE where the cube already is - satisfied at
#: reset, so the clause ends every episode on its first step.
_Z_ALREADY_SATISFIED = 0.6

#: A fall threshold the cube never reaches within :data:`_MAX_STEPS`, so the same
#: rollout is scored honestly. The control that keeps the count from being a
#: constant.
_Z_NEVER_SATISFIED = 0.01

#: A success threshold the cube's own descent crosses partway through the
#: episode - false at reset, true on a later step.
_Z_SUCCESS_REACHED = 0.495

_N_EPISODES = 3
_MAX_STEPS = 12


@pytest.fixture(autouse=True)
def _clean_registry():
    """Leave the module-global benchmark registry as it was found."""
    before = dict(_BENCHMARK_REGISTRY)
    yield
    _BENCHMARK_REGISTRY.clear()
    _BENCHMARK_REGISTRY.update(before)


@pytest.fixture
def sim():
    engine = Simulation()
    yield engine
    engine.destroy()


@pytest.fixture
def arm(tmp_path: Path, sim) -> str:
    """The arm beside a cube falling freely, with no surface anywhere."""
    sim.create_world(ground_plane=False)
    scene = tmp_path / "probe.xml"
    scene.write_text(_SCENE_XML)
    sim.add_robot("arm1", urdf_path=str(scene))
    return "arm1"


def _task(
    tmp_path: Path,
    sim,
    arm: str,
    name: str,
    *,
    failure_z: float | None,
    success_z: float,
) -> str:
    """Register a fall benchmark; ``failure_z=None`` omits the clause entirely."""
    spec: dict[str, Any] = {
        "name": name,
        "default_robot": arm,
        "supported_robots": [],
        "max_steps": _MAX_STEPS,
        "instruction": "hold still while the cube falls",
        "success": {"all": [{"predicate": "body_below_z", "body": "cube", "z": success_z}]},
    }
    if failure_z is not None:
        spec["failure"] = {"any": [{"predicate": "body_below_z", "body": "cube", "z": failure_z}]}
    spec_path = tmp_path / f"{name}.json"
    spec_path.write_text(json.dumps(spec))
    result = sim.register_benchmark_from_file(name, str(spec_path))
    assert result.get("status") != "error", result
    return name


def _metrics(result: dict) -> dict:
    assert result.get("status") != "error", result
    return next(c["json"] for c in result["content"] if "json" in c)


def _eval(sim, task: str) -> dict:
    return _metrics(sim.evaluate_benchmark(task, policy_provider="mock", n_episodes=_N_EPISODES, seed=0))


class TestTheCountIsReported:
    """The corruption is surfaced, at the aggregate and per episode."""

    def test_an_already_satisfied_failure_clause_is_counted(self, tmp_path, sim, arm) -> None:
        metrics = _eval(
            sim,
            _task(tmp_path, sim, arm, "reset-true", failure_z=_Z_ALREADY_SATISFIED, success_z=_Z_SUCCESS_REACHED),
        )
        assert metrics["episodes_failed_at_reset"] == metrics["episodes_completed"], (
            f"the cube starts at {_CUBE_START_Z}m and the clause fires below {_Z_ALREADY_SATISFIED}m, so every "
            f"episode was already failed before the policy acted, but the payload counts "
            f"{metrics['episodes_failed_at_reset']} of {metrics['episodes_completed']}"
        )
        assert metrics["reset_failure_warning"] is not None, (
            "success_rate is a hard 0.0 here regardless of the policy and the payload says nothing about it"
        )

    def test_the_warning_names_the_field_that_carries_the_count(self, tmp_path, sim, arm) -> None:
        metrics = _eval(
            sim,
            _task(tmp_path, sim, arm, "reset-true", failure_z=_Z_ALREADY_SATISFIED, success_z=_Z_SUCCESS_REACHED),
        )
        assert "episodes_failed_at_reset" in metrics["reset_failure_warning"]

    def test_each_episode_carries_its_own_verdict(self, tmp_path, sim, arm) -> None:
        metrics = _eval(
            sim,
            _task(tmp_path, sim, arm, "reset-true", failure_z=_Z_ALREADY_SATISFIED, success_z=_Z_SUCCESS_REACHED),
        )
        assert [e["failure_at_reset"] for e in metrics["episodes"]] == [True] * _N_EPISODES

    def test_the_reported_rate_is_left_as_measured(self, tmp_path, sim, arm) -> None:
        """Reported, not repaired: the count qualifies the rate, it does not move it."""
        metrics = _eval(
            sim,
            _task(tmp_path, sim, arm, "reset-true", failure_z=_Z_ALREADY_SATISFIED, success_z=_Z_SUCCESS_REACHED),
        )
        assert metrics["success_rate"] == 0.0
        assert metrics["n_failure"] == _N_EPISODES
        assert [e["steps"] for e in metrics["episodes"]] == [1] * _N_EPISODES, (
            "an episode failed at reset ends on its first step, which is the visible symptom"
        )


class TestItDoesNotOverReach:
    """The control group: a clause that does not hold at reset is untouched."""

    @pytest.mark.parametrize(
        ("name", "failure_z"),
        [("honest", _Z_NEVER_SATISFIED), ("no-clause", None)],
        ids=["a_clause_that_never_fires", "no_failure_clause_at_all"],
    )
    def test_an_unsatisfied_clause_reports_nothing(self, tmp_path, sim, arm, name, failure_z) -> None:
        metrics = _eval(sim, _task(tmp_path, sim, arm, name, failure_z=failure_z, success_z=_Z_SUCCESS_REACHED))
        assert metrics["episodes_failed_at_reset"] == 0
        assert metrics["reset_failure_warning"] is None
        assert [e["failure_at_reset"] for e in metrics["episodes"]] == [False] * _N_EPISODES


class TestTheFailureClauseDiscardsASuccessTheRunReached:
    """Why it costs more than the success mirror.

    ``is_failure`` is read before ``is_success``, so a failure clause true at reset
    ends the episode with the success criterion never consulted. The same spec and
    the same scene score ``1.0`` without the clause and ``0.0`` with it, which is
    the whole reported outcome turning on a threshold the policy cannot influence.
    """

    def test_the_same_spec_scores_one_without_the_clause_and_zero_with_it(self, tmp_path, sim, arm) -> None:
        reachable = _eval(sim, _task(tmp_path, sim, arm, "reachable", failure_z=None, success_z=_Z_SUCCESS_REACHED))
        assert reachable["success_rate"] == 1.0, (
            "premise: without a failure clause the run reaches the success criterion, "
            f"got {reachable['success_rate']} over steps {[e['steps'] for e in reachable['episodes']]}"
        )
        assert all(e["steps"] > 1 for e in reachable["episodes"]), (
            "premise: the success is earned on a later step, not at reset"
        )
        assert reachable["episodes_successful_at_reset"] == 0

        pre_empted = _eval(
            sim,
            _task(tmp_path, sim, arm, "pre-empted", failure_z=_Z_ALREADY_SATISFIED, success_z=_Z_SUCCESS_REACHED),
        )
        assert pre_empted["success_rate"] == 0.0
        assert pre_empted["episodes_failed_at_reset"] == _N_EPISODES
        assert pre_empted["episodes_successful_at_reset"] == 0, (
            "the success criterion never held at reset either - the failure clause is what discarded it"
        )


class TestTheProbeIsDiagnosticOnly:
    """A criterion that raises AT RESET does not fail the evaluation.

    The per-step call is deliberately fatal (a ``success_rate`` over episodes whose
    outcome was never determined is not a measurement). This extra call decides no
    reported figure, so it takes the posture the success probe beside it takes: a
    raise is logged and the evaluation proceeds, because a criterion reading state
    that a first ``on_step`` would have set has not had one yet.
    """

    def test_a_reset_time_raise_is_absorbed(self, tmp_path, sim, arm) -> None:
        calls: list[int] = []

        def failure_fn(_sim) -> bool:
            calls.append(1)
            if len(calls) == 1:  # the pre-episode probe
                raise RuntimeError("state a first on_step would have set")
            return False

        register_benchmark(
            "raises-at-reset",
            DeclarativeBenchmark(
                name="raises-at-reset",
                supported_robots=[],
                default_robot=arm,
                max_steps=_MAX_STEPS,
                success_fn=lambda _sim: False,
                failure_fn=failure_fn,
                reward_terms=[],
            ),
        )
        metrics = _eval(sim, "raises-at-reset")
        assert metrics["episodes_failed_at_reset"] == 0, "an unsampled probe counts nothing"
        assert metrics["reset_failure_warning"] is None
        assert metrics["episodes_completed"] == _N_EPISODES, "the evaluation still ran to completion"


class TestTheWarningText:
    """The pure function, at its boundaries."""

    def test_no_episode_failed_at_reset_reports_nothing(self) -> None:
        assert (
            failure_at_reset_warning(surface="evaluate_benchmark", episodes_completed=3, episodes_failed_at_reset=0)
            is None
        )

    def test_no_episode_completed_reports_nothing(self) -> None:
        assert (
            failure_at_reset_warning(surface="evaluate_benchmark", episodes_completed=0, episodes_failed_at_reset=2)
            is None
        )

    @pytest.mark.parametrize(
        ("failed", "expected"),
        [(3, "the reported"), (1, "that part of the")],
        ids=["every_episode", "some_episodes"],
    )
    def test_a_partial_count_is_qualified_differently(self, failed, expected) -> None:
        text = failure_at_reset_warning(
            surface="evaluate_benchmark", episodes_completed=3, episodes_failed_at_reset=failed
        )
        assert text is not None
        assert expected in text
        assert f"{failed} of 3 episode(s)" in text

    def test_the_surface_is_named_so_the_reader_finds_the_call(self) -> None:
        text = failure_at_reset_warning(surface="evaluate_benchmark", episodes_completed=2, episodes_failed_at_reset=2)
        assert text is not None
        assert text.startswith("evaluate_benchmark:")
