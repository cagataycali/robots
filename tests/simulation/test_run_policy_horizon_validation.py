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
        # control_frequency is only validated on the n_steps path (it divides
        # n_steps to produce a duration); a bad frequency there would otherwise
        # raise ZeroDivisionError or yield a negative duration.
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
    """eval_policy requires an explicit, existing robot (no silent first-pick)."""

    def test_missing_robot_name_errors(self, sim):
        text = _err_text(sim.eval_policy())
        assert "robot_name" in text

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
