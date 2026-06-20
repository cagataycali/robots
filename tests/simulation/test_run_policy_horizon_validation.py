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
