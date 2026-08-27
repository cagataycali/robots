"""A fleet-wide sim stop must not report a refused rollout as halted.

:meth:`strands_robots.mesh.Mesh.emergency_stop` broadcasts ``{"action":
"stop"}`` and carries no ``robot_name``, so on a simulation peer it lands in the
fleet-wide branch of ``_dispatch``: read every active rollout from
``_active_policy_robots()`` and call ``stop_policy`` once per robot. That branch
is the only stop path that AGGREGATES -- the named-robot branch and the child
``_sim_parent`` branch each return the stop verb's own answer, so a refusal
reaches the accounting through them unchanged.

It aggregated by assuming the answer: the per-robot results were collected into
``results`` and the envelope hardcoded ``ok=True``.
:func:`strands_robots.mesh.core._peers_that_did_not_stop` reads only the
top-level ``ok``/``status``, never ``results``, so a ``stop_policy`` that refused
was scored as a halt -- the refusal was in the payload, unread. The operator was
told the fleet had stopped, which is the affirmative lie the surrounding branches
are commented against and the exact shape that function's own docstring lists as
something it must catch.

The two paths therefore disagreed about one refusal: asked to stop ``bob`` by
name the peer answered ``{"status": "error"}`` and was flagged; asked to stop
everything with ``bob`` among the rollouts it answered ``ok=True`` with ``bob``
listed under ``stopped``.

Pinned here: the refusal is reported and flagged, both paths agree about it, the
verdict is derived through one shared predicate rather than a second copy of the
rule, and -- the half that keeps the fix from being over-broad -- a fleet stop in
which every rollout really did stop still answers affirmatively.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any

import pytest

from strands_robots.mesh import Mesh
from strands_robots.mesh.core import _peers_that_did_not_stop, _reports_failure_to_stop

#: Refusal envelope ``Simulation.stop_policy`` returns for a robot it cannot
#: resolve. Grounded against the real method by
#: :class:`TestTheStandInMatchesTheRealStopVerb`.
_REFUSAL: dict[str, Any] = {"status": "error", "content": [{"text": "Unknown robot 'bob'."}]}


class _SimPeer:
    """A simulation peer with two rollouts, one of which may refuse to stop.

    Deliberately not a ``Mock``: ``hasattr`` is the router's test, and a mock
    answers every ``hasattr`` truthfully-by-fabrication, so a mock would route
    into branches this fixture is not exercising.
    """

    def __init__(self, active: list[str], refuse: tuple[str, ...] = ()) -> None:
        self._active = list(active)
        self._refuse = set(refuse)
        self.stopped_calls: list[str] = []

    def _active_policy_robots(self) -> list[str]:
        return list(self._active)

    def stop_policy(self, robot_name: str = "") -> dict[str, Any]:
        self.stopped_calls.append(robot_name)
        if robot_name in self._refuse:
            return dict(_REFUSAL)
        return {"status": "success", "content": [{"text": f"Stopped on '{robot_name}'"}]}


def _dispatch_stop(peer: _SimPeer, **cmd: Any) -> dict[str, Any]:
    """Answer ``_dispatch`` gives a stop command aimed at *peer*."""
    mesh = Mesh(peer, peer_id="sim-1", peer_type="simulation")
    return mesh._dispatch({"action": "stop", **cmd})


def _flagged(result: dict[str, Any]) -> bool:
    """Whether ``emergency_stop``'s accounting counts *result* as not stopped."""
    return bool(_peers_that_did_not_stop([{"responder_id": "sim-1", "result": result}]))


class TestAFleetStopReportsARefusedRollout:
    """The refusal reaches the envelope the safety accounting reads."""

    def test_a_refused_rollout_makes_the_fleet_answer_negative(self) -> None:
        peer = _SimPeer(["alice", "bob"], refuse=("bob",))

        result = _dispatch_stop(peer)

        assert peer.stopped_calls == ["alice", "bob"], peer.stopped_calls
        assert result["ok"] is False, result

    def test_the_refusing_robot_is_named_and_not_counted_as_stopped(self) -> None:
        result = _dispatch_stop(_SimPeer(["alice", "bob"], refuse=("bob",)))

        assert result["not_stopped"] == ["bob"], result
        assert result["stopped"] == ["alice"], result
        assert "bob" in result["error"], result

    def test_the_accounting_counts_the_peer_as_not_stopped(self) -> None:
        result = _dispatch_stop(_SimPeer(["alice", "bob"], refuse=("bob",)))

        assert _flagged(result) is True, result

    def test_every_rollout_refusing_is_reported_in_full(self) -> None:
        result = _dispatch_stop(_SimPeer(["alice", "bob"], refuse=("alice", "bob")))

        assert result["ok"] is False, result
        assert result["not_stopped"] == ["alice", "bob"], result
        assert result["stopped"] == [], result

    def test_the_per_robot_answers_are_still_carried(self) -> None:
        """The detail stays available; ``ok`` no longer contradicts it."""
        result = _dispatch_stop(_SimPeer(["alice", "bob"], refuse=("bob",)))

        assert result["results"]["bob"] == _REFUSAL, result
        assert result["results"]["alice"]["status"] == "success", result


class TestBothStopPathsAgreeAboutARefusal:
    """One refusal, one verdict, whether or not the command named the robot."""

    @pytest.mark.parametrize("cmd", [{"robot_name": "bob"}, {}], ids=["named-robot", "fleet-wide"])
    def test_a_refusal_is_flagged_whichever_path_carried_it(self, cmd: dict[str, Any]) -> None:
        result = _dispatch_stop(_SimPeer(["bob"], refuse=("bob",)), **cmd)

        assert _flagged(result) is True, (cmd, result)

    @pytest.mark.parametrize("cmd", [{"robot_name": "bob"}, {}], ids=["named-robot", "fleet-wide"])
    def test_a_stop_that_succeeded_is_not_flagged_on_either_path(self, cmd: dict[str, Any]) -> None:
        result = _dispatch_stop(_SimPeer(["bob"]), **cmd)

        assert _flagged(result) is False, (cmd, result)


class TestAFleetStopThatSucceededIsUnchanged:
    """Over-reach controls: only a refusal may turn the answer negative."""

    def test_every_rollout_stopping_answers_affirmatively(self) -> None:
        peer = _SimPeer(["alice", "bob"])

        result = _dispatch_stop(peer)

        assert result["ok"] is True, result
        assert result["stopped"] == ["alice", "bob"], result
        assert _flagged(result) is False, result

    def test_a_fully_successful_stop_names_nothing_as_unstopped(self) -> None:
        result = _dispatch_stop(_SimPeer(["alice", "bob"]))

        assert "not_stopped" not in result, result
        assert "error" not in result, result

    def test_a_peer_with_no_rollouts_still_answers_affirmatively(self) -> None:
        result = _dispatch_stop(_SimPeer([]))

        assert result == {"ok": True, "stopped": [], "note": "no policies running"}, result
        assert _flagged(result) is False, result

    def test_a_robot_less_peer_is_unaffected(self) -> None:
        """The gateway answer this branch sits below is not disturbed."""
        result = Mesh(None, peer_id="gateway-1", peer_type="gateway")._dispatch({"action": "stop"})

        assert result["ok"] is True, result
        assert result["stopped"] == [], result
        assert _flagged(result) is False, result


class TestTheFailureRuleHasASingleOwner:
    """Two readers, one predicate: a second copy is how the branch drifted."""

    def test_the_predicate_recognises_both_envelope_spellings(self) -> None:
        assert _reports_failure_to_stop({"ok": False}) is True
        assert _reports_failure_to_stop({"status": "error"}) is True

    def test_the_predicate_stays_conservative_about_shapes_it_cannot_read(self) -> None:
        """An unrecognised shape is not a failure report -- a false "did not
        stop" on the safety path is what trains operators to ignore warnings."""
        assert _reports_failure_to_stop({}) is False
        assert _reports_failure_to_stop({"ok": True}) is False
        assert _reports_failure_to_stop({"status": "success"}) is False
        assert _reports_failure_to_stop({"stopped": []}) is False

    def test_both_readers_call_the_shared_predicate(self) -> None:
        from strands_robots.mesh import core as core_mod

        aggregation = inspect.getsource(core_mod._peers_that_did_not_stop)
        dispatch = inspect.getsource(core_mod.Mesh._dispatch)

        assert "_reports_failure_to_stop(" in aggregation, aggregation
        assert "_reports_failure_to_stop(" in dispatch, "the fleet branch must not re-derive the rule"

    def test_the_rule_itself_is_written_down_exactly_once(self) -> None:
        """Derived, so a third copy of the comparison fails rather than drifts."""
        from strands_robots.mesh import core as core_mod

        source = Path(str(core_mod.__file__)).read_text(encoding="utf-8")
        owner = inspect.getsource(core_mod._reports_failure_to_stop)
        rule = re.compile(r'\.get\(\s*["\']status["\']\s*\)\s*==\s*["\']error["\']')

        assert len(rule.findall(owner)) == 1, owner
        assert len(rule.findall(source)) == 1, "the status=='error' rule is spelled outside its owner"


class TestTheStandInMatchesTheRealStopVerb:
    """Premise: the fixture stands in for the surface production exposes."""

    def test_the_simulation_exposes_the_two_attributes_the_branch_routes_on(self) -> None:
        mujoco_sim = pytest.importorskip("strands_robots.simulation.mujoco.simulation")
        engine = mujoco_sim.MuJoCoSimEngine

        for attribute in ("stop_policy", "_active_policy_robots"):
            assert callable(getattr(engine, attribute, None)), attribute

    def test_the_real_stop_policy_refuses_an_unresolvable_robot_with_the_fixture_shape(self) -> None:
        """A refusal really is ``status="error"`` -- the shape the branch grades."""
        pytest.importorskip("mujoco")
        from strands_robots.simulation import create_simulation

        sim: Any = create_simulation("mujoco")
        try:
            refusal = sim.stop_policy("no-such-robot")
        finally:
            sim.cleanup()

        assert refusal["status"] == "error", refusal
        assert _reports_failure_to_stop(refusal) is True, refusal
