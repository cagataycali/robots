"""The agent-facing ``stop`` verb reports the halt outcome it establishes.

:meth:`~strands_robots.drivers.g1._ControlLoop.stop` signals the loop and joins
its thread within a budget, and *returns whether the thread joined*.  Its own
docstring says what a ``False`` needs: "the honest ``stopped=False`` in the
:meth:`G1Driver.stop_task` envelope rather than a ``success`` claim the
payload's ``running=True`` contradicts".

``stop_task`` reads it.  ``cleanup`` and ``stop`` were taught to read it in the
same pass.  The verb an *agent* reaches - ``stream({"action": "stop"})`` - built
its envelope beside ``await self.stop()``, and ``stop`` returns ``None``: the
protocol's shutdown hook carries no verdict, so an envelope written next to it
can only restate the intent.  A caller-supplied policy that outlasts the join
budget - a remote inference call is the ordinary case - therefore left the loop
publishing frames on ``rt/lowcmd`` while the agent read ``status="success"``.

One driver, two stop surfaces, two contracts, and the agent got the one that
cannot say it failed.  The cells that grade the fix park a policy past the join
budget, which is why they cost about two seconds each; the fast path cannot
distinguish a reported outcome from an asserted one.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import textwrap
from typing import Any

import pytest

import strands_robots.drivers.g1 as g1_mod

# Reused wholesale from the sibling that taught ``cleanup``/``stop`` to read the
# same outcome: the rollout fixtures, the recording publisher, and the autouse
# ``unitree_sdk2py`` stub the loop's write path needs.  A fixture is resolved in
# the namespace of the module that declares it, so importing the helpers without
# the stub would run these cells against a real-SDK import the loop swallows.
from tests.drivers.test_g1_cleanup_reads_the_loop_halt_outcome import (  # noqa: F401
    _Rollout,
    _stub_unitree_sdk,
)


@pytest.fixture
def unjoined() -> Any:
    """A rollout whose policy outlasts the join budget.  Always released.

    Wraps the sibling's ``_Rollout`` rather than importing its fixture:
    importing a fixture re-binds the name in this module, so every cell that
    takes it as a parameter reads as a redefinition (ruff ``F811``).
    """
    rollout = _Rollout(blocking=True)
    try:
        yield rollout
    finally:
        rollout.release()


@pytest.fixture
def joins() -> Any:
    """A rollout whose policy returns immediately, so the join succeeds."""
    rollout = _Rollout(blocking=False)
    try:
        yield rollout
    finally:
        rollout.release()


def _stop_verb(driver: Any) -> dict[str, Any]:
    """Invoke the agent-facing ``stop`` verb and return its one tool result."""

    async def _run() -> Any:
        async for event in driver.stream({"toolUseId": "u1", "name": "g1", "input": {"action": "stop"}}, {}):
            return event
        return None  # pragma: no cover - the generator always yields once

    event = asyncio.run(_run())
    assert event is not None
    return dict(event)


def _payload(envelope: dict[str, Any]) -> dict[str, Any]:
    """Return the envelope's JSON block, or fail naming what arrived instead."""
    for block in envelope.get("content", []):
        if "json" in block:
            return dict(block["json"])
    raise AssertionError(f"no json block in {envelope!r}")


class TestThePremise:
    """The facts the regression cells rest on, stated independently."""

    def test_the_halt_primitive_returns_a_verdict(self) -> None:
        """``_ControlLoop.stop`` is annotated ``bool``, so an outcome exists to read."""
        signature = inspect.signature(g1_mod._ControlLoop.stop)
        assert signature.return_annotation in (bool, "bool")

    def test_the_protocol_hook_carries_no_verdict(self) -> None:
        """``stop`` returns ``None``: an envelope beside it cannot report a halt.

        This is why the verb has to delegate rather than build its own
        envelope next to the shutdown hook.
        """
        signature = inspect.signature(g1_mod.G1Driver.stop)
        assert signature.return_annotation in (None, "None")

    def test_the_sibling_envelope_reports_the_timeout(self, unjoined: Any) -> None:
        """``stop_task`` - the surface the verb delegates to - already reports it."""
        envelope = unjoined.driver.stop_task()
        assert envelope["status"] == "error"
        assert _payload(envelope)["stopped"] is False

    def test_the_loop_really_is_still_writing_when_the_verb_returns(self, unjoined: Any) -> None:
        """The situation the report is about: the verb returns, the thread lives.

        True on either tree - what changed is whether the envelope says so.
        """
        _stop_verb(unjoined.driver)
        assert unjoined.loop.is_running is True


class TestAnUnjoinedLoopIsReportedNotClaimedStopped:
    """A policy outlasting the join budget must not read as a stopped task."""

    def test_the_verb_reports_an_error(self, unjoined: Any) -> None:
        assert _stop_verb(unjoined.driver)["status"] == "error"

    def test_the_payload_says_the_loop_did_not_stop(self, unjoined: Any) -> None:
        assert _payload(_stop_verb(unjoined.driver))["stopped"] is False

    def test_the_payload_says_the_loop_is_still_running(self, unjoined: Any) -> None:
        """The frames are still going out; the agent must be able to see that."""
        assert _payload(_stop_verb(unjoined.driver))["running"] is True

    def test_the_payload_names_the_timeout(self, unjoined: Any) -> None:
        assert "did not join within timeout" in _payload(_stop_verb(unjoined.driver))["reason"]


class TestTheJoinedOutcomeIsAlsoReported:
    """A halt that succeeded says so in the payload, not only in the status."""

    def test_the_payload_says_the_loop_stopped(self, joins: Any) -> None:
        assert _payload(_stop_verb(joins.driver))["stopped"] is True


class TestAJoinedLoopStillReportsSuccess:
    """Over-reach controls: reporting honestly must not refuse the ordinary halt.

    Both cells hold on either tree.  They are here so a future change that
    reports a timeout by failing *every* stop is caught.
    """

    def test_the_verb_reports_success(self, joins: Any) -> None:
        assert _stop_verb(joins.driver)["status"] == "success"

    def test_the_loop_has_left_its_thread(self, joins: Any) -> None:
        """Read the fixture's own handle: the driver releases ``_loop`` on exit."""
        _stop_verb(joins.driver)
        assert joins.loop.is_running is False


class TestTheVerdictIsSingleSourced:
    """The verb returns the reporting surface's envelope, not its own reading.

    ``stop_task`` already decides ``stopped`` from the join outcome.  A verb
    that re-derived it - from ``self._loop.is_running``, say - would put the
    decision in two places and could drift from the surface a caller polls.
    """

    def test_the_stop_branch_delegates_to_stop_task(self) -> None:
        source = textwrap.dedent(inspect.getsource(g1_mod.G1Driver.stream))
        tree = ast.parse(source)
        calls = {
            ast.unparse(node.func)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "self.stop_task" in calls

    def test_the_branch_does_not_restate_the_verdict(self) -> None:
        """No hardcoded halt claim survives in the verb's own source."""
        source = inspect.getsource(g1_mod.G1Driver.stream)
        assert "control loop halted" not in source

    def test_the_spec_and_the_envelope_agree_that_the_outcome_is_reported(self) -> None:
        """The schema promises a report, so the description must say so."""
        description = g1_mod.G1Driver(port="127.0.0.1", network_interface="lo").tool_spec["inputSchema"]["json"][
            "properties"
        ]["action"]["description"]
        assert "report whether it joined" in description


class TestTheStaleRefusalStaysGone:
    """Controls: the pre-#361 text claimed no motion path was wired.  It is wired.

    These hold on either tree - the stale refusal was already removed - and
    guard it against coming back through a rebase.
    """

    @pytest.mark.parametrize("stale", ["no motion path wired", "#358", "no-op"])
    def test_no_outcome_text_carries_the_stale_refusal(self, unjoined: Any, stale: str) -> None:
        envelope = _stop_verb(unjoined.driver)
        assert stale not in str(envelope)
