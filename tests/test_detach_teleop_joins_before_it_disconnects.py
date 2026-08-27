# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""``detach_teleop`` must join the teleop loop before it removes what that loop reads.

``_teleop_loop`` indexes ``self._teleops[name]`` on every tick, and
``detach_teleop`` removes entries from that mapping. So a detach which would
leave the loop with nothing to drive has to join it *before* touching a device -
which is what ``detach_teleop``'s own docstring said ("Stops the local loop
first") while the implementation stopped it last, and discarded the outcome.

``stop_teleoperate`` declines exactly this teardown for itself: on a leader whose
``get_action()`` blocks past the join budget it answers ``status="error"`` with
``stopped: False`` and deliberately leaves the devices connected, because
"Tearing the bus down under a live writer is what ``G1Driver.cleanup`` refuses
for the same reason". Measured on one wedged leader, the same state through each
verb:

    verb                  status   stopped   leader disconnected   loop alive
    stop_teleoperate()    error    False     no                    yes
    detach_teleop()       success  -         YES                   yes

Three things were wrong, and joining first repairs all three:

- the verdict. ``detach_teleop`` answered ``success`` where its own callee
  answered ``error``/``stopped: False``, and carried no ``json`` block, so
  ``_stop_reported_stopped`` on the detach envelope read ``True``: a caller had
  no way to tell.
- the teardown. The leader the loop was parked reading from was disconnected,
  undoing the protection ``stop_teleoperate`` had just decided to apply.
- the diagnosis. Because the entries were popped first, the refusal that got
  discarded described "the loop is still polling ``[]``" and claimed "The devices
  are left connected", which this caller had already falsified.

There are two routes in, and the second is the one an operator is told to take:
``stop_teleoperate`` clears ``_teleop_running`` *before* it joins and keeps the
thread handle when the join fails, so after a failed stop the session flag alone
reports an idle session while the loop is still writing. A guard keyed on
``_teleop_running`` sees route 1 only; the thread handle sees both.

What these cells pin:

    - a live writer makes ``detach_teleop`` refuse, detach nothing, and leave
      every device attached and connected so a later call re-joins that loop;
    - the refusal is machine-readable (``detached: []``, ``stopped: False``) and
      forwards the callee's reason, whose device list is now accurate;
    - both routes in are covered, including the failed-stop remedy path;
    - a clean stop still detaches and disconnects exactly as before, a bogus name
      still gets the not-found refusal, and a partial detach is untouched;
    - the outcome is read through the shared ``_stop_reported_stopped`` helper
      rather than re-derived, from a single call site that precedes the pop.

A partial detach - one of several devices, leaving the loop with something to
drive - is deliberately out of scope and pinned unchanged below. Whether the loop
should be re-selected or the detach refused there is a contract question about
multi-device sessions, not this ordering defect.

No serial port and no camera is opened; the device doubles are the ones
``test_cleanup_defers_the_close_under_a_live_teleop_writer`` uses, so port
exclusivity is modelled and the consequence is asserted rather than the call.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any

import pytest

import strands_robots.teleop_mixin as teleop_mixin
from strands_robots.teleop_mixin import TeleopMixin, _stop_reported_stopped
from tests.test_cleanup_defers_the_close_under_a_live_teleop_writer import _Session

# Short enough to keep every cell fast. The budget is pre-existing and pinned by
# ``test_stop_teleoperate_reports_the_join_outcome``; what matters here is only
# that a join can fail.
_FAST_JOIN_S = 0.3


@pytest.fixture(autouse=True)
def _fast_join(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(teleop_mixin, "_TELEOP_JOIN_TIMEOUT_S", _FAST_JOIN_S)


def _text(envelope: dict[str, Any]) -> str:
    return " ".join(b["text"] for b in envelope.get("content", []) if "text" in b)


def _json(envelope: dict[str, Any]) -> dict[str, Any] | None:
    return next((b["json"] for b in envelope.get("content", []) if "json" in b), None)


def _second_device(session: _Session) -> Any:
    """Attach a second stream so a detach can be partial."""

    class _Extra:
        is_connected = True

        def get_action(self) -> dict[str, float]:
            return {"j1.pos": 0.0}

        def disconnect(self) -> None:
            type(self).is_connected = False

    device = _Extra()
    session.robot._teleops["second"] = type("_Att", (), {"device": device, "map_fn": None})()
    return device


class TestThePremiseStopTeleoperateDeclinesThisTeardown:
    """What ``detach_teleop`` has to respect: the outcome exists and is acted on."""

    def test_a_wedged_leader_makes_the_join_fail_and_says_so(self) -> None:
        session = _Session(wedged=True)
        envelope = session.robot.stop_teleoperate()
        assert envelope["status"] == "error"
        assert _json(envelope)["stopped"] is False
        assert not _stop_reported_stopped(envelope)
        session.release()

    def test_stop_teleoperate_alone_leaves_the_leader_connected(self) -> None:
        session = _Session(wedged=True)
        session.robot.stop_teleoperate()
        assert session.leader.is_connected, "premise: it declines the teardown"
        assert session.thread.is_alive()
        session.release()

    def test_a_failed_stop_leaves_the_flag_clear_while_the_loop_writes(self) -> None:
        """Route 2's state: the session flag alone cannot see this live writer."""
        session = _Session(wedged=True)
        session.robot.stop_teleoperate()
        assert session.robot._teleop_running is False, "the flag is cleared before the join"
        assert session.robot._teleop_thread is not None, "the handle is kept for a re-join"
        assert session.thread.is_alive()
        session.release()

    def test_the_loop_reads_the_registry_by_key_on_every_tick(self) -> None:
        """Why the order matters: the loop indexes what this method removes."""
        body = textwrap.dedent(inspect.getsource(TeleopMixin._teleop_loop))
        assert "self._teleops[n]" in body


class TestALiveWriterMakesTheDetachRefuse:
    """Route 1: the loop is running when the detach arrives."""

    def test_the_detach_reports_the_failure_rather_than_success(self) -> None:
        session = _Session(wedged=True)
        envelope = session.robot.detach_teleop()
        assert envelope["status"] == "error"
        session.release()

    def test_the_leader_is_not_disconnected_under_the_live_loop(self) -> None:
        session = _Session(wedged=True)
        session.robot.detach_teleop()
        assert session.thread.is_alive(), "premise: the loop must still be running"
        assert session.leader.is_connected
        session.release()

    def test_nothing_is_detached_so_a_later_call_can_re_join(self) -> None:
        session = _Session(wedged=True)
        session.robot.detach_teleop()
        assert sorted(session.robot._teleops) == ["leader"]
        assert session.robot._teleop_thread is not None
        session.release()

    def test_the_refusal_is_machine_readable(self) -> None:
        session = _Session(wedged=True)
        payload = _json(session.robot.detach_teleop())
        assert payload == {"detached": [], "stopped": False}
        session.release()

    def test_a_caller_can_read_the_outcome_with_the_shared_helper(self) -> None:
        """The detach envelope answers the same question its callee's does."""
        session = _Session(wedged=True)
        assert not _stop_reported_stopped(session.robot.detach_teleop())
        session.release()

    def test_the_reason_is_forwarded_rather_than_re_invented(self) -> None:
        session = _Session(wedged=True)
        text = _text(session.robot.detach_teleop())
        assert "Nothing was detached" in text
        assert f"did not stop within {_FAST_JOIN_S:.1f}s" in text
        session.release()

    def test_the_forwarded_diagnosis_names_the_device_it_is_polling(self) -> None:
        """Popping first made the callee's own message report ``devices: []``.

        Matched on the phrase only that diagnosis produces: a bare ``['leader']``
        is also in the success message this used to return.
        """
        session = _Session(wedged=True)
        assert "still polling ['leader']" in _text(session.robot.detach_teleop())
        session.release()


class TestTheFailedStopRemedyPathRefusesToo:
    """Route 2: ``_teleop_running`` is already False while the loop still writes."""

    def test_the_detach_still_refuses_after_a_failed_stop(self) -> None:
        session = _Session(wedged=True)
        session.robot.stop_teleoperate()
        envelope = session.robot.detach_teleop()
        assert envelope["status"] == "error"
        assert _json(envelope) == {"detached": [], "stopped": False}
        session.release()

    def test_the_devices_survive_the_remedy_path(self) -> None:
        session = _Session(wedged=True)
        session.robot.stop_teleoperate()
        session.robot.detach_teleop()
        assert session.leader.is_connected
        assert sorted(session.robot._teleops) == ["leader"]
        session.release()


class TestWhatIsUnchanged:
    """Every path that did not involve a live writer answers exactly as before."""

    def test_a_clean_stop_still_detaches_and_disconnects(self) -> None:
        session = _Session(wedged=False)
        envelope = session.robot.detach_teleop()
        assert envelope["status"] == "success"
        assert "Detached: ['leader']" in _text(envelope)
        assert session.robot._teleops == {}
        assert not session.leader.is_connected

    def test_a_clean_stop_leaves_no_running_session(self) -> None:
        session = _Session(wedged=False)
        session.robot.detach_teleop()
        assert session.robot._teleop_running is False
        assert session.robot._teleop_thread is None

    def test_a_name_that_matches_nothing_is_still_refused(self) -> None:
        session = _Session(wedged=True)
        envelope = session.robot.detach_teleop("ghost")
        assert envelope["status"] == "error"
        assert "No teleop named 'ghost'." in _text(envelope)
        assert sorted(session.robot._teleops) == ["leader"]
        session.release()

    def test_an_empty_name_still_does_not_stop_a_running_session(self) -> None:
        """The membership contract: ``""`` names no device, so nothing happens."""
        session = _Session(wedged=True)
        _second_device(session)
        envelope = session.robot.detach_teleop("")
        assert envelope["status"] == "error"
        assert session.robot._teleop_running is True
        assert session.leader.is_connected
        session.release()

    def test_a_partial_detach_is_out_of_scope_and_unchanged(self) -> None:
        """One of two devices: the loop keeps something to drive, so it is not joined."""
        session = _Session(wedged=True)
        extra = _second_device(session)
        envelope = session.robot.detach_teleop("second")
        assert envelope["status"] == "success"
        assert sorted(session.robot._teleops) == ["leader"]
        assert not extra.is_connected
        session.release()

    def test_a_clean_join_whose_every_frame_errored_still_detaches(self) -> None:
        """Why the verdict is ``stopped`` and not ``status``.

        ``_teleop_stats`` derives the status from the session counters, so a
        session whose every frame errored answers ``status="error"`` with
        ``stopped: True`` after a perfectly clean join. Keying the guard on the
        status would refuse this detach, which is safe.
        """
        session = _Session(wedged=False)
        session.robot._teleop_frames = 1
        session.robot._teleop_errors = 1
        envelope = session.robot.detach_teleop()
        assert envelope["status"] == "success"
        assert session.robot._teleops == {}

    def test_an_idle_robot_with_devices_still_detaches(self) -> None:
        session = _Session(wedged=False)
        session.robot.stop_teleoperate()
        envelope = session.robot.detach_teleop()
        assert envelope["status"] == "success"
        assert session.robot._teleops == {}


class TestTheOutcomeIsReadFromOneSingleSourcedCallSite:
    """Structural: the drift this fixes is a discarded return value."""

    def _detach_source(self) -> ast.FunctionDef:
        module = ast.parse(textwrap.dedent(inspect.getsource(TeleopMixin.detach_teleop)))
        node = module.body[0]
        assert isinstance(node, ast.FunctionDef)
        return node

    def test_the_stop_is_called_exactly_once_and_its_value_is_never_discarded(self) -> None:
        fn = self._detach_source()
        calls = [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "stop_teleoperate"
        ]
        assert len(calls) == 1, f"expected one stop_teleoperate() call site, found {len(calls)}"
        discarded = [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Attribute)
            and n.value.func.attr == "stop_teleoperate"
        ]
        assert discarded == [], "the stop outcome is discarded as a bare expression"

    def test_the_verdict_comes_from_the_shared_helper(self) -> None:
        """Not re-derived from ``status`` or from the thread handle."""
        fn = self._detach_source()
        names = {n.func.id for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert "_stop_reported_stopped" in names

    def test_the_join_precedes_the_removal(self) -> None:
        fn = self._detach_source()
        stop_line = next(
            n.lineno
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "stop_teleoperate"
        )
        pop_line = next(
            n.lineno
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "pop"
        )
        assert stop_line < pop_line, "the loop must be joined before the registry is emptied"

    def test_the_guard_consults_the_thread_handle_not_only_the_session_flag(self) -> None:
        """``_teleop_running`` is already False on the failed-stop remedy path."""
        fn = self._detach_source()
        source = ast.unparse(fn)
        assert "_teleop_thread" in source
