"""The benchmark per-episode horizon has one domain on every path that sets it.

``evaluate_benchmark`` documents that ``max_steps`` "comes from the benchmark
(not a parameter here)", so the benchmark object is the sole authority for the
per-episode horizon. It is the one bound of the evaluation's nested loop with no
parameter of its own to validate - ``n_episodes``, ``action_horizon`` and
``control_substeps`` are all checked at the public entry point, and
``eval_policy`` checks its own ``max_steps`` there too.

That left the horizon reachable through four paths, only one of which checked it:

* ``DeclarativeBenchmark.from_dict`` (and ``register_benchmark_from_file``)
  rejected a non-positive-integer horizon.
* ``DeclarativeBenchmark(...)`` coerced with a bare ``int()``, so ``2.7`` became
  ``2``, ``True`` became ``1``, and ``0`` / ``-5`` were stored verbatim.
* a benchmark adapter's ``max_steps=...`` (and its ``from_file`` /
  ``from_text`` classmethods) did the same, one line below a
  validated ``init_jitter``.
* A plain :class:`BenchmarkProtocol` subclass setting the documented
  ``max_steps`` attribute - the extension point the base class invites - was not
  checked at all, and neither was an assignment to it after construction.

The consequence is the one
:meth:`~strands_robots.simulation.base.SimEngine._validate_positive_int`'s
docstring already names for this parameter: "episodes of zero length, that
fabricate a 0% success rate". A benchmark declaring ``max_steps=0`` returned
``status="success"`` reporting ``0.0%`` success over episodes that applied no
action, and a fractional or NaN horizon raised a bare ``TypeError`` out of
``range()``, past the agent-tool envelope.

These tests pin the single domain (:func:`strands_robots.utils.positive_count_error`,
whose own docstring already names ``max_steps``) on every path that sets the
horizon, and pin that it is honored where it is read.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from strands_robots.simulation.benchmark import (
    BenchmarkProtocol,
    StepInfo,
    register_benchmark,
    unregister_benchmark,
)
from strands_robots.simulation.benchmark_spec import DeclarativeBenchmark
from strands_robots.utils import positive_count_error

# A horizon has to be a positive int: the value is consumed directly as a
# ``range()`` bound, where an integral float raises rather than being coerced.
UNUSABLE = [0, -5, 2.7, True, False, float("nan"), float("inf"), "300", None, [300]]
USABLE = [1, 60, 300]

# A minimal single-hinge arm: enough for a real rollout, no asset download.
ARM_XML = """<mujoco model="probe_arm">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 2"/>
    <geom name="floor" type="plane" size="2 2 .05"/>
    <body name="base" pos="0 0 0.02">
      <geom type="box" size=".05 .05 .02"/>
      <body name="link" pos="0 0 .04">
        <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" limited="true" damping="4"/>
        <geom type="capsule" fromto="0 0 0 0 0 .25" size=".03"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a_shoulder" joint="shoulder" kp="40" ctrlrange="-2 2"/>
  </actuator>
</mujoco>
"""


def _declarative(max_steps: Any) -> DeclarativeBenchmark:
    """Construct a ``DeclarativeBenchmark`` directly with the given horizon."""
    return DeclarativeBenchmark(
        name="probe",
        supported_robots=[],
        default_robot="probe_arm",
        max_steps=max_steps,
        success_fn=lambda _sim: False,
        failure_fn=lambda _sim: False,
        reward_terms=[],
    )


def _from_dict(max_steps: Any) -> DeclarativeBenchmark:
    """Compile a spec dict with the given horizon (the already-guarded path)."""
    return DeclarativeBenchmark.from_dict(
        {
            "name": "probe",
            "supported_robots": [],
            "default_robot": "probe_arm",
            "max_steps": max_steps,
        }
    )


class _AttributeBenchmark(BenchmarkProtocol):
    """A plain subclass that sets the documented ``max_steps`` attribute.

    This is the extension point :class:`BenchmarkProtocol` invites ("subclasses
    can set it in ``__init__`` or as a class attribute"), and it has no
    constructor the library can guard - which is why the horizon has to be
    checked where the evaluation loop reads it.
    """

    def __init__(self, max_steps: Any) -> None:
        self.max_steps = max_steps

    @property
    def supported_robots(self) -> list[str]:
        return []

    @property
    def default_robot(self) -> str:
        return "probe_arm"

    def is_success(self, sim: Any) -> bool:
        return False

    def on_step(self, sim: Any, obs: dict[str, Any], action: dict[str, Any]) -> StepInfo:
        return StepInfo(reward=0.0, done=False)


@pytest.fixture
def eval_sim(tmp_path: Path):
    """A MuJoCo sim with one registered arm, plus a bound mock policy."""
    from strands_robots import MockPolicy, Simulation

    arm = tmp_path / "arm.xml"
    arm.write_text(ARM_XML)
    sim = Simulation(backend="mujoco", mesh=False)
    try:
        sim.create_world()
        sim.add_robot(name="probe_arm", urdf_path=str(arm))
        policy = MockPolicy()
        policy.set_robot_state_keys(sim.robot_action_keys("probe_arm"))
        yield sim, policy
    finally:
        try:
            sim.cleanup()
        except Exception:  # noqa: BLE001 - teardown is best effort
            pass


def _evaluate(sim: Any, policy: Any, benchmark: BenchmarkProtocol, n_episodes: int = 2) -> dict[str, Any]:
    """Register ``benchmark`` and evaluate it, returning the result envelope."""
    register_benchmark("horizon_probe", benchmark)
    try:
        return sim.evaluate_benchmark(
            benchmark_name="horizon_probe",
            robot_name="probe_arm",
            policy_object=policy,
            n_episodes=n_episodes,
            control_frequency=50.0,
        )
    finally:
        unregister_benchmark("horizon_probe")


def _json(result: dict[str, Any]) -> dict[str, Any]:
    return next((block["json"] for block in result.get("content", []) if "json" in block), {})


def _text(result: dict[str, Any]) -> str:
    return " ".join(" ".join(block["text"] for block in result.get("content", []) if "text" in block).split())


class TestEveryCreationPathSharesOneDomain:
    """The horizon is refused identically however the benchmark acquires it."""

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_direct_construction_refuses_a_horizon_the_loop_cannot_honor(self, value: Any) -> None:
        with pytest.raises(ValueError, match="max_steps must be a positive integer"):
            _declarative(value)

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_a_spec_dict_refuses_the_same_horizon(self, value: Any) -> None:
        with pytest.raises(ValueError, match="max_steps must be a positive integer"):
            _from_dict(value)

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_the_two_declarative_paths_agree(self, value: Any) -> None:
        """A spec file and a direct construction cannot drift on what they accept.

        The dict path validated the raw value while the constructor it feeds
        coerced with ``int()``, so the same horizon was refused as a spec key and
        accepted as a keyword.
        """

        def verdict(build: Any) -> str:
            try:
                build(value)
            except ValueError:
                return "refused"
            return "accepted"

        assert verdict(_from_dict) == verdict(_declarative), f"verdicts differ for max_steps={value!r}"

    @pytest.mark.parametrize("value", USABLE)
    def test_a_usable_horizon_is_still_accepted_on_both_paths(self, value: int) -> None:
        assert _declarative(value).max_steps == value
        assert _from_dict(value).max_steps == value

    def test_an_omitted_spec_horizon_keeps_the_documented_default(self) -> None:
        """Omitting the key is not a caller mistake - the default still applies."""
        assert (
            DeclarativeBenchmark.from_dict(
                {"name": "probe", "supported_robots": [], "default_robot": "probe_arm"}
            ).max_steps
            == 300
        )

    def test_a_usable_horizon_is_stored_without_coercion(self) -> None:
        """The stored value is the caller's, not an ``int()`` of it."""
        stored = _declarative(720).max_steps
        assert stored == 720
        assert isinstance(stored, int) and not isinstance(stored, bool)


class TestTheHorizonIsCheckedWhereItIsRead:
    """A benchmark whose constructor the library cannot guard is still refused."""

    @pytest.mark.parametrize("value", [0, -5, 2.7, True, float("nan"), float("inf"), "300"])
    def test_an_attribute_horizon_is_refused_with_a_structured_error(self, eval_sim: Any, value: Any) -> None:
        sim, policy = eval_sim
        result = _evaluate(sim, policy, _AttributeBenchmark(value))
        assert result["status"] == "error", result
        text = _text(result)
        assert "max_steps must be a positive integer" in text
        assert "_AttributeBenchmark" in text, text

    def test_a_horizon_assigned_after_construction_is_refused(self, eval_sim: Any) -> None:
        """Validating only the constructors would miss a later assignment."""
        sim, policy = eval_sim
        benchmark = _declarative(60)
        benchmark.max_steps = 0  # type: ignore[assignment]
        result = _evaluate(sim, policy, benchmark)
        assert result["status"] == "error", result
        assert "max_steps must be a positive integer" in _text(result)

    def test_a_zero_horizon_no_longer_reports_a_success_rate(self, eval_sim: Any) -> None:
        """The reported failure mode: a 0% success rate over empty episodes.

        A zero horizon used to return ``status="success"`` with
        ``Avg steps: 0/0`` and ``0.0%`` success over episodes that applied no
        action, which is indistinguishable from a policy that genuinely failed.
        """
        sim, policy = eval_sim
        result = _evaluate(sim, policy, _AttributeBenchmark(0))
        assert result["status"] == "error"
        payload = _json(result)
        assert "success_rate" not in payload, payload
        assert "0.0%" not in _text(result)

    def test_a_refused_horizon_does_not_reseed_the_process_rng(self, eval_sim: Any) -> None:
        """The guard runs before ``set_eval_seed``, which is a global side effect."""
        import random

        sim, policy = eval_sim
        register_benchmark("horizon_probe", _AttributeBenchmark(0))
        try:
            random.seed(1234)
            expected = random.random()
            random.seed(1234)
            result = sim.evaluate_benchmark(
                benchmark_name="horizon_probe",
                robot_name="probe_arm",
                policy_object=policy,
                n_episodes=1,
                seed=99,
                control_frequency=50.0,
            )
            assert result["status"] == "error"
            assert random.random() == expected, "a refused evaluation reseeded the global RNG"
        finally:
            unregister_benchmark("horizon_probe")

    def test_a_usable_horizon_still_runs_and_is_honored(self, eval_sim: Any) -> None:
        """The control: a valid horizon evaluates and caps each episode at it."""
        sim, policy = eval_sim
        result = _evaluate(sim, policy, _AttributeBenchmark(12), n_episodes=2)
        assert result["status"] == "success", _text(result)
        payload = _json(result)
        assert payload["max_steps"] == 12
        assert payload["n_episodes"] == 2
        assert payload["avg_steps"] == pytest.approx(12.0)


class TestParityWithTheAlreadyGuardedSiblingBound:
    """``eval_policy``'s ``max_steps`` and a benchmark's share one domain."""

    @pytest.mark.parametrize("value", UNUSABLE + USABLE)  # type: ignore[operator]
    def test_the_benchmark_horizon_matches_eval_policy_max_steps(self, value: Any) -> None:
        """Both are the same per-episode step cap, so they cannot diverge.

        ``eval_policy(max_steps=...)`` routes through
        ``SimEngine._validate_positive_int``; a benchmark's horizon is the same
        quantity arriving by a different route.
        """
        from strands_robots.simulation.base import SimEngine

        sibling = SimEngine._validate_positive_int(value, "max_steps", "eval_policy")
        assert (sibling is None) == (positive_count_error(value, "max_steps", "evaluate_benchmark") is None)

    def test_the_domain_rejects_bool_and_integral_floats(self) -> None:
        """Pins why this is the count domain and not the whole-number one."""
        assert positive_count_error(True, "max_steps", "evaluate_benchmark") is not None
        assert positive_count_error(300.0, "max_steps", "evaluate_benchmark") is not None
        assert positive_count_error(300, "max_steps", "evaluate_benchmark") is None

    def test_a_non_integer_horizon_no_longer_escapes_as_a_range_error(self) -> None:
        """Pre-fix a fractional/NaN horizon raised out of ``range()``.

        The premise: these values are exactly the ones ``range()`` refuses, so
        without a guard the failure surfaced as a bare ``TypeError`` rather than
        the tool envelope.
        """
        for value in (2.7, math.nan, math.inf):
            # Typed Any so the deliberate misuse reaches the runtime, which is
            # the behaviour under test.
            bound: Any = value
            with pytest.raises(TypeError):
                range(bound)
            assert positive_count_error(value, "max_steps", "evaluate_benchmark") is not None
