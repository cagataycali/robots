"""Pin the JSON-in-instruction goal extraction of :class:`CuroboPolicy`.

``Robot.start_task(..., policy_provider="curobo")`` lets an LLM pack the
issue-#300 goal fields into the natural-language instruction, and
``CuroboPolicy._parse_target`` is what recovers them. The extraction used to
take the span from the FIRST ``{`` to the LAST ``}`` in the string, which

* dropped a valid goal whenever the instruction carried a brace of its own on
  either side of the payload - the span then covered prose and failed to
  decode, so ``get_actions`` raised "requires at least one of target_pose=..."
  about a goal that was right there in its input; and
* scanned the instruction quadratically when the braces never balance, so a
  long LLM-authored string could stall the 50Hz caller for seconds
  (``py/polynomial-redos``).

The extraction now decodes each candidate object at its own closing brace.
These tests fail on the pre-fix source.
"""

from __future__ import annotations

import asyncio
import logging
import time

import pytest

from strands_robots.policies.curobo import CuroboPolicy
from tests.policies.curobo.test_policy import _StubMotionGen


class TestGoalSurvivesNeighbouringBraces:
    """A goal payload is found even when the prose around it carries braces."""

    def test_brace_in_prose_before_the_goal(self) -> None:
        tp, tj = CuroboPolicy._parse_target('close the {gripper} then move {"target_joints": {"j1": 0.5}}')
        assert tp is None
        assert tj == {"j1": 0.5}

    def test_brace_in_prose_after_the_goal(self) -> None:
        tp, tj = CuroboPolicy._parse_target(
            '{"target_pose": [0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]} then release the {clamp}'
        )
        assert tp == [0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]
        assert tj is None

    def test_braces_on_both_sides_of_the_goal(self) -> None:
        tp, tj = CuroboPolicy._parse_target('from {home} go to {"target_joints": {"j0": -0.25}} at {slow} speed')
        assert tp is None
        assert tj == {"j0": -0.25}

    def test_first_goal_carrying_object_wins(self) -> None:
        """Objects are scanned left to right; the first one with a goal field is
        the goal, so a leading non-goal object does not mask it."""
        tp, _tj = CuroboPolicy._parse_target(
            '{"note": "warm up"} {"target_pose": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]} '
            '{"target_pose": [9.0, 9.0, 9.0, 1.0, 0.0, 0.0, 0.0]}'
        )
        assert tp == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    def test_goal_field_named_in_the_prose_does_not_hide_the_payload(self) -> None:
        """The field name appearing as prose ahead of the payload must not be
        mistaken for the payload's position - the object still opens later."""
        tp, tj = CuroboPolicy._parse_target(
            'set target_pose from the {chart}: {"target_pose": [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]}'
        )
        assert tp == [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]
        assert tj is None

    def test_goal_after_braces_reaches_the_planner(self) -> None:
        """End to end: the recovered goal drives a plan instead of raising."""
        stub = _StubMotionGen(ndof=6, horizon=5)
        policy = CuroboPolicy(motion_gen=stub, action_horizon=4)
        actions = asyncio.run(
            policy.get_actions(
                {"observation.state": [0.0] * 6},
                'pick up the {block} and go to {"target_pose": [0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]}',
            )
        )
        assert actions
        assert stub.plan_calls and stub.plan_calls[0][0] == "plan_single"


class TestGoalExtractionContractUnchanged:
    """Cases whose outcome the wider scan must not change."""

    def test_goal_nested_in_a_wrapper_object_is_not_a_goal(self) -> None:
        """Only a TOP-LEVEL goal field counts. A decoded object without one is
        skipped whole rather than descended into, so ``{"goal": {...}}`` stays
        unparsed exactly as before - honouring it would be a new feature, not a
        fix."""
        assert CuroboPolicy._parse_target('{"goal": {"target_pose": [1, 2, 3, 1, 0, 0, 0]}}') == (None, None)

    def test_malformed_payload_degrades_to_no_goal(self) -> None:
        assert CuroboPolicy._parse_target("do this {target_pose: not, json}") == (None, None)

    def test_object_without_goal_fields_is_not_a_goal(self) -> None:
        assert CuroboPolicy._parse_target('noise {"unrelated": 1} tail') == (None, None)


class TestUnparseablePayloadIsReported:
    """A mentioned-but-unrecoverable goal is logged, not silently dropped."""

    def test_warns_when_a_mentioned_goal_cannot_be_decoded(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="strands_robots.policies.curobo.policy"):
            assert CuroboPolicy._parse_target('go to {"target_pose": [1, 2,,]}') == (None, None)
        assert any("no JSON object carrying" in r.getMessage() for r in caplog.records)

    def test_no_warning_when_no_goal_field_is_mentioned(self, caplog: pytest.LogCaptureFixture) -> None:
        """A plain instruction is the normal kwargs-driven path - it must not
        log about a goal the caller never claimed to embed."""
        with caplog.at_level(logging.WARNING, logger="strands_robots.policies.curobo.policy"):
            assert CuroboPolicy._parse_target("move the gripper down slowly") == (None, None)
        assert caplog.records == []


class TestExtractionIsBounded:
    """The scan is bounded, so an unbalanced instruction cannot stall the loop."""

    def test_unbalanced_braces_return_quickly(self) -> None:
        # 100k unclosed braces followed by the goal field name. The pre-fix
        # ``re.search(r"\{.*\}", ..., re.DOTALL)`` restarts a full-length scan
        # at every one of them (measured 3.1 s); the bounded decode scan
        # returns in well under a millisecond.
        instruction = "{" * 100_000 + '"target_pose"'
        started = time.perf_counter()
        assert CuroboPolicy._parse_target(instruction) == (None, None)
        assert time.perf_counter() - started < 1.0

    def test_prose_braces_do_not_consume_the_candidate_budget(self) -> None:
        """A brace that cannot open a JSON object is skipped without spending a
        decode attempt, so the bound is a backstop against a pathological string
        rather than a limit on how much braced prose may precede the goal."""
        instruction = "{step} " * 200 + '{"target_pose": [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]}'
        tp, _tj = CuroboPolicy._parse_target(instruction)
        assert tp == [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]

    def test_long_instruction_with_no_braces_is_not_scanned(self) -> None:
        instruction = "move the arm slowly " * 20_000
        started = time.perf_counter()
        assert CuroboPolicy._parse_target(instruction) == (None, None)
        assert time.perf_counter() - started < 1.0
