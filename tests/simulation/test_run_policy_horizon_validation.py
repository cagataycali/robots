"""Validation contracts for run_policy / start_policy / eval_policy horizons.

These public Simulation methods accept a step-horizon (`n_steps`, with legacy
`max_steps`) and a target `control_frequency`. The horizon is converted to a
wall-clock `duration` via ``duration = n_steps / control_frequency``, so the
inputs must be guarded before that division: a non-positive horizon or a
non-positive frequency is a caller error, not a silent no-op or a ZeroDivision.

Per the agent-tool contract every method returns a structured
``{"status": ..., "content": [...]}`` dict rather than raising past dispatch,
so each guard is asserted to return ``status="error"`` with an actionable,
ASCII-only message naming the offending parameter. The legacy ``max_steps``
alias is asserted to behave identically to ``n_steps`` (it is normalized to
``n_steps`` before the guards run).
"""

from __future__ import annotations

import numpy as np
import pytest

from strands_robots.simulation import create_simulation


@pytest.fixture
def sim():
    s = create_simulation()
    s.create_world()
    s.add_robot("arm1", data_config="so100")
    yield s
    s.cleanup()


@pytest.fixture
def empty_sim():
    s = create_simulation()
    s.create_world()
    yield s
    s.cleanup()


def _err_text(result: dict) -> str:
    assert result["status"] == "error", result
    return result["content"][0]["text"]


class TestRunPolicyHorizonGuards:
    """run_policy must reject malformed step horizons before stepping physics."""

    @pytest.mark.parametrize("bad", [0, -1, -50])
    def test_non_positive_n_steps_errors(self, sim, bad):
        text = _err_text(sim.run_policy("arm1", n_steps=bad))
        assert "n_steps must be > 0" in text
        assert str(bad) in text

    @pytest.mark.parametrize("bad_freq", [0, -10.0])
    def test_non_positive_control_frequency_errors(self, sim, bad_freq):
        # control_frequency is validated at the run_policy entry point (it is a
        # divisor: 1 / control_frequency action period and n_steps /
        # control_frequency duration); a bad frequency would otherwise raise
        # ZeroDivisionError or yield a negative duration deep in the runner.
        text = _err_text(sim.run_policy("arm1", n_steps=5, control_frequency=bad_freq))
        assert "control_frequency must be > 0" in text

    def test_legacy_max_steps_alias_is_validated_like_n_steps(self, sim):
        # max_steps is normalized to n_steps before the guards, so a
        # non-positive max_steps surfaces the same n_steps error.
        text = _err_text(sim.run_policy("arm1", max_steps=0))
        assert "n_steps must be > 0" in text

    def test_error_message_is_ascii(self, sim):
        text = _err_text(sim.run_policy("arm1", n_steps=-1))
        text.encode("ascii")  # raises UnicodeEncodeError if any non-ASCII leaks

    def test_guard_runs_before_robot_lookup(self, sim):
        # A non-positive horizon is reported even when the robot name is also
        # wrong: the horizon guard short-circuits ahead of the robot lookup,
        # so the caller sees the horizon problem first.
        text = _err_text(sim.run_policy("ghost", n_steps=0))
        assert "n_steps must be > 0" in text


class TestStartPolicyHorizonGuards:
    """start_policy must validate the horizon synchronously.

    start_policy runs the rollout on a background thread, so a malformed
    horizon must be caught before submission. Otherwise the caller receives
    a false "started" success while the rollout silently errors in the
    future, and the robot is left marked as running.
    """

    def test_non_positive_n_steps_errors_synchronously(self, sim):
        text = _err_text(sim.start_policy("arm1", n_steps=-1))
        assert "n_steps must be > 0" in text

    def test_non_positive_control_frequency_errors_synchronously(self, sim):
        text = _err_text(sim.start_policy("arm1", n_steps=5, control_frequency=0))
        assert "control_frequency must be > 0" in text

    def test_rejected_start_does_not_mark_robot_running(self, sim):
        # A rejected start must not leave a future registered for the robot,
        # otherwise a subsequent valid start_policy is wrongly gated as
        # "already running".
        result = sim.start_policy("arm1", n_steps=0)
        assert result["status"] == "error"
        assert "arm1" not in sim._policy_threads
        # A well-formed start on the same robot now succeeds.
        ok = sim.start_policy("arm1", n_steps=2, control_frequency=50.0, fast_mode=True)
        assert ok["status"] == "success", ok
        sim.stop_policy("arm1")


class TestEvalPolicyResolution:
    """eval_policy resolves robot_name like run_policy: None auto-selects the
    sole robot, and only errors when the choice is ambiguous or impossible."""

    def test_omitted_robot_name_resolves_sole_robot(self, sim):
        # Single-robot scene + no name -> resolves to that robot and runs,
        # exactly like run_policy() (no spurious "requires robot_name" error).
        result = sim.eval_policy(n_episodes=1, max_steps=2, control_frequency=10.0)
        assert result["status"] == "success", result

    def test_ambiguous_multi_robot_lists_candidates(self, sim):
        sim.add_robot("arm2", data_config="so100", position=[0.5, 0, 0])
        text = _err_text(sim.eval_policy())
        assert "arm1" in text and "arm2" in text

    def test_unknown_robot_name_errors(self, sim):
        text = _err_text(sim.eval_policy(robot_name="ghost"))
        assert "ghost" in text
        assert "not found" in text

    def test_empty_world_reports_no_robots(self, empty_sim):
        text = _err_text(empty_sim.eval_policy(robot_name="arm1"))
        assert "No robots" in text


class TestActionHorizonGuards:
    """run_policy / start_policy / eval_policy must reject a non-positive
    ``action_horizon`` rather than silently clamping it.

    ``action_horizon`` is the number of actions consumed from each policy
    chunk before re-querying. ``resolve_chunk_length`` clamps it to ``>= 1``,
    so a typo like ``action_horizon=0`` or ``-3`` used to be silently coerced
    to 1 and the rollout ran a horizon the caller never asked for. The public
    entry points now surface this as a structured caller error - matching the
    guard ``evaluate_benchmark`` already enforced.
    """

    @pytest.mark.parametrize("bad", [0, -1, -8])
    def test_run_policy_rejects_non_positive_action_horizon(self, sim, bad):
        text = _err_text(sim.run_policy("arm1", n_steps=4, action_horizon=bad))
        assert "action_horizon must be a positive integer" in text
        assert str(bad) in text

    def test_run_policy_rejects_non_int_action_horizon(self, sim):
        text = _err_text(sim.run_policy("arm1", n_steps=4, action_horizon=2.5))
        assert "action_horizon must be a positive integer" in text

    def test_run_policy_accepts_positive_action_horizon(self, sim):
        result = sim.run_policy("arm1", n_steps=2, control_frequency=50.0, fast_mode=True, action_horizon=1)
        assert result["status"] == "success", result

    def test_eval_policy_rejects_non_positive_action_horizon(self, sim):
        text = _err_text(sim.eval_policy(robot_name="arm1", max_steps=4, action_horizon=0))
        assert "action_horizon must be a positive integer" in text

    def test_start_policy_rejects_action_horizon_synchronously(self, sim):
        # The rollout runs on a background thread, so a bad action_horizon must
        # be caught before submission - otherwise the caller gets a false
        # "started" success and the robot is left marked as running.
        result = sim.start_policy("arm1", n_steps=4, action_horizon=-1)
        assert result["status"] == "error"
        assert "action_horizon must be a positive integer" in result["content"][0]["text"]
        assert "arm1" not in sim._policy_threads
        # A well-formed start on the same robot still succeeds afterwards.
        ok = sim.start_policy("arm1", n_steps=2, control_frequency=50.0, fast_mode=True)
        assert ok["status"] == "success", ok
        sim.stop_policy("arm1")

    def test_action_horizon_error_is_ascii(self, sim):
        text = _err_text(sim.run_policy("arm1", n_steps=4, action_horizon=0))
        text.encode("ascii")  # raises UnicodeEncodeError if any non-ASCII leaks


class TestControlFrequencyGuards:
    """control_frequency is validated at EVERY public entry point, on every path.

    ``control_frequency`` (Hz) is a divisor (``1 / control_frequency`` action
    period, ``n_steps / control_frequency`` duration) and is handed to the
    runner's per-period substep arithmetic, which raises a bare
    ``ValueError``/``TypeError``/``ZeroDivisionError`` on a bad value. The
    n_steps path and ``eval_policy`` were already guarded, but three paths still
    leaked a raw traceback instead of the structured tool-error dict:

    - ``run_policy`` duration path (``n_steps`` omitted): never validated.
    - ``start_policy`` duration path: the synchronous guard only covered the
      n_steps path, so a bad rate passed, raised inside the background future,
      and left the robot falsely marked running.
    - a ``bool`` (``True``): an ``int`` subclass, it slipped through the numeric
      check on every path and silently acted as a 1 Hz rate.
    """

    @pytest.mark.parametrize("bad_freq", [0, 0.0, -10.0])
    def test_run_policy_duration_path_rejects_non_positive(self, sim, bad_freq):
        # n_steps omitted -> duration path; pre-fix this reached the runner and
        # raised ValueError instead of returning a structured error.
        text = _err_text(sim.run_policy("arm1", duration=1.0, control_frequency=bad_freq, fast_mode=True))
        assert "control_frequency must be > 0" in text
        assert str(bad_freq) in text

    @pytest.mark.parametrize("bad_freq", ["fast", None])
    def test_run_policy_rejects_non_numeric(self, sim, bad_freq):
        # Pre-fix the n_steps inline check did `bad <= 0`, raising TypeError for
        # a str/None rather than returning a structured error.
        text = _err_text(sim.run_policy("arm1", n_steps=4, control_frequency=bad_freq))
        assert "control_frequency must be > 0" in text

    def test_run_policy_rejects_bool_control_frequency(self, sim):
        # bool is an int subclass; True would sneak through an isinstance(int)
        # check and act as a silent 1 Hz, so it is rejected explicitly.
        text = _err_text(sim.run_policy("arm1", n_steps=4, control_frequency=True))
        assert "control_frequency must be > 0" in text

    def test_eval_policy_rejects_bool_control_frequency(self, sim):
        text = _err_text(sim.eval_policy(robot_name="arm1", max_steps=4, control_frequency=True))
        assert "control_frequency must be > 0" in text

    def test_start_policy_duration_path_rejects_synchronously(self, sim):
        # Duration path on the background-threaded start_policy: pre-fix the
        # n_steps-only horizon guard let this through, it raised in the future,
        # and the robot was left marked running.
        result = sim.start_policy("arm1", duration=1.0, control_frequency=0)
        assert result["status"] == "error"
        assert "control_frequency must be > 0" in result["content"][0]["text"]
        assert "arm1" not in sim._policy_threads
        # A well-formed start on the same robot still succeeds afterwards.
        ok = sim.start_policy("arm1", n_steps=2, control_frequency=50.0, fast_mode=True)
        assert ok["status"] == "success", ok
        sim.stop_policy("arm1")

    def test_run_policy_accepts_positive_control_frequency(self, sim):
        result = sim.run_policy("arm1", n_steps=2, control_frequency=30.0, fast_mode=True)
        assert result["status"] == "success", result

    def test_control_frequency_error_is_ascii(self, sim):
        text = _err_text(sim.run_policy("arm1", duration=1.0, control_frequency=-1.0, fast_mode=True))
        text.encode("ascii")  # raises UnicodeEncodeError if any non-ASCII leaks


class TestControlFrequencyNumpyScalars:
    """control_frequency accepts NumPy real scalars but still rejects junk.

    control_frequency is routinely computed from a config array or an
    observation (``fps = 1.0 / dt`` where ``dt`` is a ``np.float32``), so the
    guard must accept any real scalar. ``isinstance(x, (int, float))`` is
    ``False`` for every NumPy scalar except ``np.float64`` (the only one that
    subclasses Python ``float``), so pre-fix a perfectly valid
    ``np.float32(50.0)`` / ``np.int64(50)`` frequency was rejected with the
    misleading "control_frequency must be > 0" error. The guard now uses
    ``numbers.Real`` while still rejecting ``bool`` / ``np.bool_``, non-finite
    values, and non-positive frequencies. This is shared by run_policy,
    eval_policy and evaluate_benchmark via ``_validate_positive_frequency``.
    """

    @pytest.mark.parametrize("freq", [np.float32(30.0), np.int64(30), np.float64(30.0)])
    def test_run_policy_accepts_numpy_scalar_frequency(self, sim, freq):
        result = sim.run_policy("arm1", n_steps=2, control_frequency=freq, fast_mode=True)
        assert result["status"] == "success", result

    def test_eval_policy_accepts_numpy_scalar_frequency(self, sim):
        result = sim.eval_policy(robot_name="arm1", n_episodes=1, max_steps=2, control_frequency=np.float32(30.0))
        assert result["status"] == "success", result

    def test_run_policy_realtime_path_accepts_numpy_scalar_frequency(self, sim):
        # The default (non-fast_mode) real-time path computes
        # ``action_sleep = 1.0 / control_frequency`` and calls
        # ``time.sleep(action_sleep)``. A NumPy-scalar frequency left
        # action_sleep a ``numpy.float32``, which ``time.sleep`` rejects with
        # "'numpy.float32' object cannot be interpreted as an integer" -- so
        # accepting the scalar at the guard is not enough; it must be coerced to
        # a Python float. A high frequency keeps the per-step sleep negligible.
        result = sim.run_policy("arm1", n_steps=2, control_frequency=np.float32(500.0))
        assert result["status"] == "success", result

    @pytest.mark.parametrize("bad", [np.float32(-1.0), np.int64(-5), np.float32("nan"), np.bool_(True)])
    def test_run_policy_rejects_bad_numpy_scalar_frequency(self, sim, bad):
        # A negative NumPy scalar, np.bool_, or non-finite value is still a
        # caller error and must surface the structured guard, not step physics.
        text = _err_text(sim.run_policy("arm1", n_steps=2, control_frequency=bad, fast_mode=True))
        assert "control_frequency must be > 0" in text

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_run_policy_rejects_non_finite_python_float_frequency(self, sim, bad):
        # A non-finite Python float used to slip through (nan/inf are never
        # ``<= 0``) and feed nan/inf into the ``1 / frequency`` and
        # ``n_steps / frequency`` arithmetic; it is now rejected up front.
        text = _err_text(sim.run_policy("arm1", n_steps=2, control_frequency=bad, fast_mode=True))
        assert "control_frequency must be > 0" in text


class TestEvalPolicyCountGuards:
    """eval_policy must reject non-positive rollout counts and frequency.

    eval_policy used to validate only ``action_horizon`` and ``robot_name``;
    its ``n_episodes`` (number of reset->rollout episodes), ``max_steps``
    (per-episode step cap) and ``control_frequency`` were unvalidated. A
    zero/negative ``n_episodes`` or ``max_steps`` flowed into the eval loop and
    returned ``status="success"`` with a fabricated success rate over zero or
    negative episodes (``Episodes: -2 | Success: 0/-2``) or zero-length episodes
    (``Avg steps: 0/-5``), hiding the caller's mistake; a non-positive
    ``control_frequency`` raised a bare ``ValueError`` from deep inside the
    runner instead of the structured tool-error dict the public API contracts.
    These guards run at the entry point, before ``create_policy``.
    """

    @pytest.mark.parametrize("bad", [0, -2])
    def test_rejects_non_positive_n_episodes(self, sim, bad):
        text = _err_text(sim.eval_policy(robot_name="arm1", n_episodes=bad))
        assert "n_episodes must be a positive integer" in text
        assert str(bad) in text

    def test_rejects_non_int_n_episodes(self, sim):
        text = _err_text(sim.eval_policy(robot_name="arm1", n_episodes=1.5))
        assert "n_episodes must be a positive integer" in text

    @pytest.mark.parametrize("bad", [0, -5])
    def test_rejects_non_positive_max_steps(self, sim, bad):
        text = _err_text(sim.eval_policy(robot_name="arm1", max_steps=bad))
        assert "max_steps must be a positive integer" in text
        assert str(bad) in text

    @pytest.mark.parametrize("bad_freq", [0, -10.0])
    def test_rejects_non_positive_control_frequency(self, sim, bad_freq):
        text = _err_text(sim.eval_policy(robot_name="arm1", max_steps=3, control_frequency=bad_freq))
        assert "control_frequency must be > 0" in text

    def test_count_error_is_ascii(self, sim):
        text = _err_text(sim.eval_policy(robot_name="arm1", n_episodes=-2))
        text.encode("ascii")  # raises UnicodeEncodeError if any non-ASCII leaks

    def test_guard_runs_before_policy_creation(self, sim):
        # A malformed n_episodes is reported even when the policy provider is
        # also bogus: the count guard short-circuits ahead of create_policy, so
        # the caller sees the count problem rather than a provider/download error.
        text = _err_text(sim.eval_policy(robot_name="arm1", policy_provider="no_such_provider", n_episodes=0))
        assert "n_episodes must be a positive integer" in text

    def test_accepts_valid_counts(self, sim):
        result = sim.eval_policy(
            robot_name="arm1", policy_provider="mock", n_episodes=1, max_steps=2, control_frequency=50.0
        )
        assert result["status"] == "success", result


def _reported_n_steps(result: dict) -> int:
    """Pull the executed step count out of a successful run_policy result."""
    assert result["status"] == "success", result
    payloads = [c["json"] for c in result["content"] if "json" in c]
    assert payloads, f"no json payload in result: {result}"
    return int(payloads[0]["n_steps"])


class TestNStepsExactHorizon:
    """run_policy(n_steps=N) must execute EXACTLY N control steps at every rate.

    The step horizon used to round-trip through a float wall-clock duration
    (``duration = n_steps / control_frequency``) which the runner then
    reconverted with ``int(duration * control_frequency)``. Floating-point
    error truncated the count on any frequency that does not divide the horizon
    evenly: ``n_steps=1 @ 49 Hz`` reconverted to ``int(1/49*49) == 0``, so the
    rollout returned ``status="success"`` having executed ZERO steps. The
    integer horizon is now forwarded to the runner verbatim, so the executed
    count equals the requested count exactly and independently of the rate.
    """

    @pytest.mark.parametrize("control_frequency", list(range(1, 121)))
    def test_single_step_runs_exactly_one_step(self, sim, control_frequency):
        # n_steps=1 is the canonical regression: pre-fix this ran 0 steps at
        # every frequency where 1/f*f floors below 1 (e.g. 49, 90, 98 Hz).
        result = sim.run_policy("arm1", n_steps=1, control_frequency=float(control_frequency), fast_mode=True)
        assert _reported_n_steps(result) == 1

    @pytest.mark.parametrize("n_steps", [1, 13, 15, 37, 100])
    @pytest.mark.parametrize("control_frequency", [11.0, 49.0, 50.0, 90.0, 120.0])
    def test_arbitrary_horizon_runs_exact_count(self, sim, n_steps, control_frequency):
        # e.g. 13 @ 90 Hz reconverted to 12, 15 @ 11 Hz to 14 pre-fix.
        result = sim.run_policy("arm1", n_steps=n_steps, control_frequency=control_frequency, fast_mode=True)
        assert _reported_n_steps(result) == n_steps

    def test_legacy_max_steps_alias_runs_exact_count(self, sim):
        # max_steps is normalized to n_steps, so it must be exact too.
        result = sim.run_policy("arm1", max_steps=1, control_frequency=49.0, fast_mode=True)
        assert _reported_n_steps(result) == 1

    def test_duration_path_unchanged_when_n_steps_omitted(self, sim):
        # When no explicit horizon is given the wall-clock duration path still
        # governs: 0.2 s @ 50 Hz -> 10 steps.
        result = sim.run_policy("arm1", duration=0.2, control_frequency=50.0, fast_mode=True)
        assert _reported_n_steps(result) == 10


class TestDurationGuards:
    """The wall-clock ``duration`` horizon must be guarded like the step count.

    ``duration`` is the DEFAULT horizon knob: with no ``n_steps`` the rollout
    length is ``int(duration * control_frequency)`` control steps. A value
    ``<= 0`` produced zero steps and reported ``status="success"`` for a
    rollout that never queried the policy and never stepped physics; a
    non-finite value did not even survive the arithmetic (``nan`` raised
    ``ValueError: cannot convert float NaN to integer``, ``inf`` an
    ``OverflowError``) and surfaced as a message naming a library internal
    instead of the parameter the caller got wrong.
    """

    @pytest.mark.parametrize("bad", [0, 0.0, -1, -0.5, -50.0])
    def test_non_positive_duration_errors(self, sim, bad):
        text = _err_text(sim.run_policy("arm1", duration=bad, fast_mode=True))
        assert "duration must be > 0" in text
        assert repr(bad) in text

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_duration_errors(self, sim, bad):
        # nan is never <= 0, so it must be rejected on finiteness before the
        # comparison; pre-fix it reached int(nan * frequency) and raised.
        text = _err_text(sim.run_policy("arm1", duration=bad, fast_mode=True))
        assert "duration must be > 0" in text

    @pytest.mark.parametrize("bad", ["2.0", None, [1.0]])
    def test_non_numeric_duration_errors(self, sim, bad):
        text = _err_text(sim.run_policy("arm1", duration=bad, fast_mode=True))
        assert "duration must be > 0" in text

    def test_bool_duration_errors(self, sim):
        # bool is an int subclass: True would act as a silent 1-second rollout.
        text = _err_text(sim.run_policy("arm1", duration=True, fast_mode=True))
        assert "duration must be > 0" in text

    def test_duration_error_is_ascii(self, sim):
        text = _err_text(sim.run_policy("arm1", duration=-1.0, fast_mode=True))
        assert text.isascii(), text
        assert "run_policy" in text

    def test_guard_runs_before_robot_lookup(self, sim):
        # The horizon guards precede the robot lookup, so the caller is told
        # about the parameter they control rather than about a robot name that
        # is only wrong in passing.
        text = _err_text(sim.run_policy("no_such_robot", duration=0, fast_mode=True))
        assert "duration must be > 0" in text

    def test_numpy_scalar_duration_accepted(self, sim):
        # Mirrors the control_frequency contract: any finite positive real
        # scalar is valid, including a NumPy scalar read out of a config array.
        result = sim.run_policy("arm1", duration=np.float32(0.2), control_frequency=50.0, fast_mode=True)
        assert result["status"] == "success", result
        assert _reported_n_steps(result) == 10

    def test_explicit_step_horizon_wins_over_unused_duration(self, sim):
        # With an n_steps the duration is recomputed from it and never read, so
        # the guard must not reject a value the rollout does not use.
        result = sim.run_policy("arm1", n_steps=2, duration=0, control_frequency=50.0, fast_mode=True)
        assert result["status"] == "success", result
        assert _reported_n_steps(result) == 2

    def test_rejected_duration_does_not_write_requested_video(self, sim, tmp_path):
        # The most damaging shape of the bug: a recording rollout reported
        # success while writing no MP4 at all, because zero frames were
        # captured. The request is now refused up front.
        out = tmp_path / "rollout.mp4"
        text = _err_text(sim.run_policy("arm1", duration=-1.0, video={"path": str(out)}, fast_mode=True))
        assert "duration must be > 0" in text
        assert not out.exists()


class TestStartPolicyDurationGuard:
    """start_policy must reject a bad duration synchronously.

    The rollout runs on a background thread, so an unguarded duration was
    reported as "started" while the future ran zero steps - the caller had no
    way to learn the rollout never happened.
    """

    @pytest.mark.parametrize("bad", [0, -1.0, float("nan")])
    def test_bad_duration_errors_synchronously(self, sim, bad):
        text = _err_text(sim.start_policy("arm1", duration=bad))
        assert "duration must be > 0" in text

    def test_rejected_start_does_not_mark_robot_running(self, sim):
        result = sim.start_policy("arm1", duration=-1.0)
        assert result["status"] == "error"
        assert "arm1" not in sim._policy_threads
        # A well-formed start on the same robot still succeeds afterwards.
        ok = sim.start_policy("arm1", n_steps=2, control_frequency=50.0, fast_mode=True)
        assert ok["status"] == "success", ok
        sim.stop_policy("arm1")

    def test_horizon_error_names_start_policy(self, sim):
        text = _err_text(sim.start_policy("arm1", n_steps=0))
        assert "start_policy: n_steps must be > 0" in text


class TestRunMultiPolicyHorizonGuards:
    """run_multi_policy shares the horizon contract of run_policy.

    It resolved the horizon with its own inline check that only fired when an
    explicit ``n_steps`` was passed, so the default duration path computed
    ``total_steps = int(duration * control_frequency) == 0`` and reported a
    synchronized multi-robot rollout that never ran as a success.
    """

    @pytest.fixture
    def policies(self):
        from strands_robots.policies import MockPolicy

        return {"arm1": MockPolicy()}

    @pytest.mark.parametrize("bad", [0, -1.0, float("nan"), "2.0"])
    def test_bad_duration_errors(self, sim, policies, bad):
        text = _err_text(sim.run_multi_policy(policies, duration=bad))
        assert "run_multi_policy: duration must be > 0" in text

    def test_non_positive_n_steps_errors(self, sim, policies):
        text = _err_text(sim.run_multi_policy(policies, n_steps=0))
        assert "run_multi_policy: n_steps must be > 0" in text

    def test_non_positive_control_frequency_errors_on_duration_path(self, sim, policies):
        # Pre-fix the frequency was only checked alongside n_steps, so the
        # duration path reached 1 / control_frequency with a zero divisor.
        text = _err_text(sim.run_multi_policy(policies, duration=0.2, control_frequency=0))
        assert "run_multi_policy: control_frequency must be > 0" in text

    def test_valid_horizon_still_runs(self, sim, policies):
        result = sim.run_multi_policy(policies, n_steps=2, control_frequency=50.0)
        assert result["status"] == "success", result


class _CountingMockPolicy:
    """MockPolicy wrapper that counts how often the policy is re-queried.

    ``run_multi_policy`` buffers each ``get_actions`` chunk and only re-queries a
    policy when its queue drains, so the call count is the observable proof that
    the requested ``action_horizon`` was actually honored.
    """

    def __init__(self) -> None:
        from strands_robots.policies import MockPolicy

        self._inner = MockPolicy()
        self.calls = 0

    @property
    def provider_name(self) -> str:
        return self._inner.provider_name

    @property
    def requires_images(self) -> bool:
        return self._inner.requires_images

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self._inner.set_robot_state_keys(robot_state_keys)

    async def get_actions(self, observation_dict: dict, instruction: str, **kwargs: object) -> list[dict]:
        self.calls += 1
        return await self._inner.get_actions(observation_dict, instruction, **kwargs)


@pytest.fixture
def two_robot_sim():
    s = create_simulation()
    s.create_world()
    s.add_robot("arm1", data_config="so100", position=[0.0, 0.0, 0.0])
    s.add_robot("arm2", data_config="so100", position=[0.6, 0.0, 0.0])
    yield s
    s.cleanup()


class TestRunMultiPolicyActionHorizonGuards:
    """run_multi_policy validates action_horizon like every sibling driver.

    ``run_policy`` / ``start_policy`` / ``eval_policy`` / ``evaluate_benchmark``
    all route ``action_horizon`` through the shared positive-integer guard. The
    multi-robot loop instead coerced it with ``max(1, int(...))``, so ``0`` / a
    negative value silently became a 1-action horizon (the policy was re-queried
    every step, not at the requested cadence), a float was truncated, and
    ``nan`` / ``None`` / ``"x"`` reached ``int()`` and escaped as a bare
    ``ValueError`` / ``TypeError`` past the structured-dict contract.
    """

    @pytest.fixture
    def policies(self):
        from strands_robots.policies import MockPolicy

        return {"arm1": MockPolicy(), "arm2": MockPolicy()}

    @pytest.mark.parametrize("bad", [0, -1, -8, 2.7, "4", None, float("nan")])
    def test_rejects_horizon_it_cannot_honor(self, two_robot_sim, policies, bad):
        text = _err_text(two_robot_sim.run_multi_policy(policies, n_steps=4, action_horizon=bad))
        assert "run_multi_policy: action_horizon must be a positive integer" in text
        text.encode("ascii")  # no non-ASCII leaks into the error path

    @pytest.mark.parametrize("bad", [0, -3, 2.9, "x"])
    def test_rejects_per_robot_horizon_it_cannot_honor(self, two_robot_sim, policies, bad):
        text = _err_text(two_robot_sim.run_multi_policy(policies, n_steps=4, action_horizon={"arm1": bad}))
        # The message names the offending ENTRY, not just the parameter.
        assert "run_multi_policy: action_horizon['arm1'] must be a positive integer" in text

    def test_honors_requested_requery_cadence(self, two_robot_sim):
        # 8 steps at horizon 4 must re-query each policy twice; the same rollout
        # at horizon 1 re-queries every step. Pre-fix action_horizon=0 was
        # clamped to 1 and reported success, i.e. it ran THIS cadence while the
        # caller had asked for something else entirely.
        coarse = {"arm1": _CountingMockPolicy(), "arm2": _CountingMockPolicy()}
        assert two_robot_sim.run_multi_policy(coarse, n_steps=8, action_horizon=4)["status"] == "success"
        assert [p.calls for p in coarse.values()] == [2, 2]

        fine = {"arm1": _CountingMockPolicy(), "arm2": _CountingMockPolicy()}
        assert two_robot_sim.run_multi_policy(fine, n_steps=8, action_horizon=1)["status"] == "success"
        assert [p.calls for p in fine.values()] == [8, 8]

    def test_per_robot_mapping_leaves_omitted_robots_on_the_default(self, two_robot_sim):
        # A partial mapping is an override layer: arm1 re-queries every 4 steps,
        # arm2 keeps the default horizon of 8 and is queried once for 8 steps.
        pols = {"arm1": _CountingMockPolicy(), "arm2": _CountingMockPolicy()}
        result = two_robot_sim.run_multi_policy(pols, n_steps=8, action_horizon={"arm1": 4})
        assert result["status"] == "success", result
        assert pols["arm1"].calls == 2
        assert pols["arm2"].calls == 1

    def test_empty_mapping_runs_every_robot_on_the_default(self, two_robot_sim):
        # {} expresses "no per-robot override", which the loop CAN honor - it is
        # identical to omitting the argument, so it is not a caller error.
        pols = {"arm1": _CountingMockPolicy(), "arm2": _CountingMockPolicy()}
        result = two_robot_sim.run_multi_policy(pols, n_steps=8, action_horizon={})
        assert result["status"] == "success", result
        assert [p.calls for p in pols.values()] == [1, 1]


class TestRunMultiPolicyPerRobotMappingKeys:
    """A per-robot mapping key must name a robot this call drives.

    ``instructions`` and ``action_horizon`` were both read with
    ``mapping.get(robot, default)``, which discards every unmatched key. A typo'd
    or stale robot name therefore ran the episode on the defaults - an empty
    instruction, the default horizon - and still reported ``status="success"``,
    so the caller's per-robot request vanished with no diagnostic.
    """

    @pytest.fixture
    def policies(self):
        from strands_robots.policies import MockPolicy

        return {"arm1": MockPolicy(), "arm2": MockPolicy()}

    def test_unknown_instruction_key_rejected(self, two_robot_sim, policies):
        text = _err_text(two_robot_sim.run_multi_policy(policies, instructions={"arm11": "pick cube"}, n_steps=4))
        assert "run_multi_policy: instructions names a robot not driven by this call" in text
        assert "arm11" in text
        assert "Did you mean: arm1" in text
        assert "['arm1', 'arm2']" in text

    def test_unknown_action_horizon_key_rejected(self, two_robot_sim, policies):
        text = _err_text(two_robot_sim.run_multi_policy(policies, n_steps=4, action_horizon={"arm3": 4}))
        assert "run_multi_policy: action_horizon names a robot not driven by this call" in text
        assert "arm3" in text

    def test_robot_in_scene_but_not_driven_is_rejected(self, two_robot_sim):
        # arm2 exists in the world but is NOT driven by this call, so an
        # instruction keyed to it can never be applied.
        from strands_robots.policies import MockPolicy

        text = _err_text(
            two_robot_sim.run_multi_policy({"arm1": MockPolicy()}, instructions={"arm2": "hold"}, n_steps=4)
        )
        assert "instructions names a robot not driven by this call" in text
        assert "['arm1']" in text

    def test_non_mapping_instructions_rejected(self, two_robot_sim, policies):
        # A list reached mapping.get() and escaped as a bare AttributeError.
        text = _err_text(two_robot_sim.run_multi_policy(policies, instructions=["pick", "hold"], n_steps=4))
        assert "run_multi_policy: 'instructions' must be a string" in text
        assert "list" in text

    def test_well_formed_per_robot_mappings_still_run(self, two_robot_sim, policies):
        result = two_robot_sim.run_multi_policy(
            policies,
            instructions={"arm1": "pick cube", "arm2": "hold tray"},
            n_steps=4,
            action_horizon={"arm1": 2, "arm2": 4},
        )
        assert result["status"] == "success", result
