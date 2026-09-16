"""``stop_policy`` says when the rollout it found nothing to stop had died.

``start_policy`` answers "Policy started" before its worker has built a policy
or taken a step, so a rollout that fails after that fails where no caller is
looking - the reason lands in ``_rollouts_ended_in_error`` and only
``list_policies_running`` read it. That is not the verb an agent reaches for:
every rollout gate names ``stop_policy`` as the way out, and both of its
"nothing is running" answers - the refusal for an empty name and the idempotent
``was_running=False`` success - were silent about the failure, which is the same
reading as a rollout that ran to completion. Measured on
``MuJoCoSimEngine`` before this fix: a policy that raised on its first inference
gave ``stop_policy requires 'robot_name'. No policy is running now; robots:
'so101'.`` and then, following that advice, ``status="success"`` with ``Was not
running on 'so101'``.

The verdict does not change here - a stop that found nothing in flight is still
an idempotent success - only the silence.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import pytest

pytest.importorskip("mujoco")

from strands_robots.policies.mock import MockPolicy
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

_REASON = "policy server dropped the connection"


class _DyingPolicy(MockPolicy):
    """Builds like any other policy and raises on its first inference.

    The shape of every failure this reports in the field - a policy server that
    is not listening, a checkpoint that will not load - without needing one.
    """

    def get_actions(self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any) -> Any:
        raise RuntimeError(_REASON)


def _text(result: dict[str, Any]) -> str:
    return "\n".join(c["text"] for c in result["content"] if isinstance(c, dict) and "text" in c)


def _json(result: dict[str, Any]) -> dict[str, Any]:
    return next(c["json"] for c in result["content"] if isinstance(c, dict) and "json" in c)


@pytest.fixture
def sim():
    engine = MuJoCoSimEngine(tool_name="stop_reports", mesh=False)
    engine.create_world()
    assert engine.add_robot(name="so101", data_config="so101")["status"] == "success"
    yield engine
    engine.cleanup()


def _start_a_rollout_that_dies(engine: MuJoCoSimEngine) -> None:
    """Launch an asynchronous rollout and wait for its worker to fail."""
    assert (
        engine.start_policy("so101", policy_object=_DyingPolicy(), duration=5.0, control_frequency=30.0)["status"]
        == "success"
    ), "the failure has to happen after start_policy answered success"
    deadline = time.monotonic() + 10.0
    while not engine._rollouts_ended_in_error() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert engine._rollouts_ended_in_error(), "the worker did not record its failure in time"
    assert engine._rollouts_in_flight() == (), "nothing is in flight, so both stop answers are the silent ones"


def test_the_bare_stop_names_the_rollout_that_died(sim):
    """The refusal an every-gate remedy leads to carries the reason."""
    _start_a_rollout_that_dies(sim)
    refusal = sim.stop_policy("")
    assert refusal["status"] == "error"
    assert "stop_policy requires 'robot_name'." in _text(refusal)
    assert f"so101: Policy failed: {_REASON}" in _text(refusal)


def test_the_named_stop_says_the_rollout_died_and_keeps_its_verdict(sim):
    """Following that refusal's advice no longer reads as a rollout that finished."""
    _start_a_rollout_that_dies(sim)
    stopped = sim.stop_policy("so101")
    assert stopped["status"] == "success", "a stop with nothing in flight stays idempotent"
    assert _json(stopped)["was_running"] is False, "the verdict is unchanged; the silence is what is fixed"
    assert (
        f"Was not running on 'so101'. Its last start_policy rollout ended in error: Policy failed: {_REASON}"
        in _text(stopped)
    )


def test_a_rollout_that_did_not_fail_leaves_both_answers_untouched(sim):
    """A healthy rollout's stop says exactly what it always said, to the byte."""
    assert (
        sim.start_policy("so101", policy_provider="mock", duration=5.0, control_frequency=30.0)["status"] == "success"
    )
    assert "Stopped on 'so101'" in _text(sim.stop_policy("so101"))
    assert sim._rollouts_ended_in_error() == {}, "a cooperative stop is not a failure"
    assert _text(sim.stop_policy("so101")) == "Was not running on 'so101'"
    assert _text(sim.stop_policy("")) == "stop_policy requires 'robot_name'. No policy is running now; robots: 'so101'."


def test_both_readers_render_the_failure_from_one_seam(sim):
    """The refusal and ``list_policies_running`` cannot drift about the same instant."""
    _start_a_rollout_that_dies(sim)
    block = f"\nRollouts started with start_policy that ended in error (1):\n  - so101: Policy failed: {_REASON}"
    assert block in _text(sim.stop_policy(""))
    assert block in _text(sim.list_policies_running())


def test_a_backend_with_no_asynchronous_entry_appends_nothing():
    """The ABC default records no failure, so the bare requirement stays bare."""
    # Imported here, not at module scope: the rest of this file grades text a
    # caller reads, and must fail on its assertions rather than on a collection
    # error when these renderers are absent.
    from strands_robots.simulation.base import _rollout_error_note, _rollout_error_report
    from tests.simulation.test_stop_policy_base_contract import _MinimalEngine

    engine = _MinimalEngine()
    assert engine._rollouts_ended_in_error() == {}
    assert _rollout_error_report(engine._rollouts_ended_in_error()) == ""
    assert _rollout_error_note(engine._rollouts_ended_in_error(), "anything") == ""
    assert _text(engine.stop_policy("")) == "stop_policy requires 'robot_name'."


def test_a_backend_that_inherits_the_base_stop_policy_reports_it_too():
    """Newton's shape - it keeps a durable claim and inherits ``stop_policy``.

    The note belongs to the base verb, not to MuJoCo's override, so a backend
    that never overrode ``stop_policy`` answers the same way.
    """
    from tests.simulation.test_stop_policy_base_contract import _MinimalEngine

    class _Durable(_MinimalEngine):
        def _request_policy_stop(self, robot_name: str) -> bool | None:
            return False

        def _rollouts_in_flight(self) -> tuple[str, ...] | None:
            return ()

        def _rollouts_ended_in_error(self) -> Mapping[str, str]:
            return {"arm": f"Policy failed: {_REASON}"}

    engine = _Durable()
    named = engine.stop_policy("arm")
    assert named["status"] == "success"
    assert _json(named)["was_running"] is False
    assert (
        _text(named)
        == f"Was not running on 'arm'. Its last start_policy rollout ended in error: Policy failed: {_REASON}"
    )
    assert f"arm: Policy failed: {_REASON}" in _text(engine.stop_policy(""))
