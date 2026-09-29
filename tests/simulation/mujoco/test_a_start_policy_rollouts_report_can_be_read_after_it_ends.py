"""The report of a ``start_policy`` rollout is readable once the rollout ends.

``start_policy`` returned "Policy started" and nothing else; after the worker
finished, ``stop_policy`` answered "Was not running" with no report and only
failures were kept, so the ``run_policy`` envelope (steps, action health,
``video_path``) of every background rollout was lost (#4162). The engine now
keeps the last completed envelope per robot: ``policy_result(robot_name)``
returns it and ``stop_policy`` carries it as ``last_result``.
"""

from __future__ import annotations

import time

import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation import create_simulation  # noqa: E402
from strands_robots.simulation.base import SimEngine  # noqa: E402


def _json(envelope: dict) -> dict:
    return next(c["json"] for c in envelope["content"] if "json" in c)


def _text(envelope: dict) -> str:
    return "\n".join(c["text"] for c in envelope["content"] if "text" in c)


def _wait_for_result(sim, robot: str, timeout_s: float = 15.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        result = sim.policy_result(robot)
        if result is not None:
            return result
        time.sleep(0.05)
    raise AssertionError("the rollout never recorded a result")


@pytest.fixture
def sim():
    s = create_simulation("mujoco", mesh=False)
    s.create_world()
    s.add_robot("so101")
    yield s
    s.cleanup()


class TestTheResultIsKept:
    def test_nothing_is_reported_before_a_rollout_ran(self, sim) -> None:
        assert sim.policy_result("so101") is None
        assert "last_result" not in _json(sim.stop_policy("so101"))

    def test_the_run_policy_envelope_is_readable_after_completion(self, sim) -> None:
        assert (
            sim.start_policy(robot_name="so101", policy_provider="mock", n_steps=12, control_frequency=50.0)["status"]
            == "success"
        )
        result = _wait_for_result(sim, "so101")
        assert result["status"] == "success"
        block = _json(result)
        # The same block run_policy returns, not a summary of it.
        for key in ("n_steps", "actions_applied", "action_resolution_rate", "avg_inference_ms", "instruction_read"):
            assert key in block, key
        assert block["n_steps"] == 12

    def test_stop_after_completion_carries_the_report(self, sim) -> None:
        sim.start_policy(robot_name="so101", policy_provider="mock", n_steps=8, control_frequency=50.0)
        _wait_for_result(sim, "so101")
        stopped = sim.stop_policy("so101")
        verdict = _json(stopped)
        assert verdict["was_running"] is False
        assert verdict["last_result"]["status"] == "success"
        assert _json(verdict["last_result"])["n_steps"] == 8
        assert "Last rollout on 'so101' ended success: Policy complete on 'so101'" in _text(stopped)

    def test_the_reader_hands_out_a_copy(self, sim) -> None:
        sim.start_policy(robot_name="so101", policy_provider="mock", n_steps=4, control_frequency=50.0)
        first = _wait_for_result(sim, "so101")
        first["status"] = "tampered"
        assert sim.policy_result("so101")["status"] == "success"

    def test_a_new_rollout_clears_the_previous_report_until_it_ends(self, sim) -> None:
        sim.start_policy(robot_name="so101", policy_provider="mock", n_steps=4, control_frequency=50.0)
        _wait_for_result(sim, "so101")
        sim.start_policy(robot_name="so101", policy_provider="mock", n_steps=200, control_frequency=50.0)
        # In flight: the stale report is not offered as the current one.
        assert sim.policy_result("so101") is None
        stopped = sim.stop_policy("so101")
        verdict = _json(stopped)
        assert verdict["was_running"] is True
        # The stop that halted the rollout is its answer; the report of the run it
        # cut short is read through policy_result, and no later stop repeats it.
        assert "last_result" not in verdict
        assert _wait_for_result(sim, "so101")["status"] == "success"
        again = sim.stop_policy("so101")
        assert "last_result" not in _json(again)
        assert _text(again) == "Was not running on 'so101'"

    def test_only_the_first_stop_after_completion_carries_the_report(self, sim) -> None:
        sim.start_policy(robot_name="so101", policy_provider="mock", n_steps=4, control_frequency=50.0)
        _wait_for_result(sim, "so101")
        assert "last_result" in _json(sim.stop_policy("so101"))
        assert _json(sim.stop_policy("so101")) == {"robot": "so101", "was_running": False, "exited": None}
        assert sim.policy_result("so101") is not None

    def test_a_rollout_that_died_is_reported_as_an_error_envelope(self, sim) -> None:
        # ``remote`` against nothing: the worker's constructor refuses and the failure
        # was already kept as a reason; the envelope is now kept too.
        sim.start_policy(robot_name="so101", policy_provider="remote", n_steps=4, control_frequency=50.0)
        result = _wait_for_result(sim, "so101")
        assert result["status"] == "error"
        assert _text(result)


def test_the_base_engine_default_is_none() -> None:
    assert SimEngine.policy_result(object(), "so101") is None  # type: ignore[arg-type]
