"""``getDaemonStatus``'s envelope reports this driver's verdict, not the daemon's.

``ReachyMiniDriver`` answers every RPC with the Device Connect envelope --
``{"status": "success", ...}`` or ``{"status": "error", "reason": ...}``. Five
of its REST RPCs carry the daemon's reply back to the caller and every one of
them nests it under a key of its own (``{"status": "success", "result":
result}``). ``getDaemonStatus`` merged it into the envelope instead, and that
one difference cost the envelope two properties:

* **The daemon decided the verdict.** Spread last, a reply carrying a
  ``status`` field of its own replaced the driver's. A daemon reporting
  ``status="idle"`` made a healthy RPC answer ``"idle"`` -- outside the
  envelope's two-value vocabulary, so a caller branching on
  ``status == "success"`` reads a call that worked as a call that did not.
  :meth:`strands_robots.mesh.sensors.SensorLoopsMixin._stamp_local_keys`
  resolves the same collision the other way for the sensor records, and its
  docstring gives the reason: a record must not name something other than what
  the surface that built it decided.

* **A daemon that was never reached answered ``success``.**
  :func:`~strands_robots.device_connect.reachy_transport.api` reports every
  HTTP and connection failure as ``{"error": ...}`` rather than raising -- the
  driver states this in two places already
  (:meth:`ReachyMiniDriver.__init__`'s port comment and
  :meth:`ReachyMiniDriver._stop_motion_impl`'s docstring), and the native
  driver refuses that shape when reading this same endpoint
  (:meth:`strands_robots.drivers.reachy.ReachyDriver.connect_eagerly`). Merged
  into the envelope, the reason travelled *beside* a success verdict: the one
  RPC whose subject is whether the daemon is up was the one that could not say
  it was down.

A third shape follows from the same statement: a body that decodes to a JSON
array or scalar is not a mapping, so spreading it raised ``TypeError`` out of a
method whose entire contract is the envelope.

The second point is a property of every REST RPC, not of the merge: nesting the
reply under a key of the envelope's own keeps it away from ``status`` but does
not decide one. ``playMove``, ``listMoves``, ``wakeUp`` and ``sleep`` each
answered ``status="success"`` carrying the transport's reason inside ``result``,
so a caller was told the move played, the catalogue was read and the motors were
woken by a daemon that was never reached -- three of the four command motion.
``TestNoRestVerbReportsACallThatDidNotLandAsASuccess`` grades the rule over the
whole family, and every verb is driven a second time against a daemon that
*did* answer, because a verb refused by the authorization gate in front of the
daemon read would otherwise satisfy the failure rows for the wrong reason.

``TestTheHealthyPayloadIsUnchanged`` and ``TestWhyTheEnvelopeOwnsTheVerdict``
pass on both trees -- the first pins the reading that must not change (both
payload shapes the pre-existing suites drive), the second pins the premises
that make this driver's own verdict the one to keep.
"""

from __future__ import annotations

import ast
import asyncio
import importlib
import socket
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("device_connect_edge")


def _force_real_device_connect_edge() -> None:
    """Restore the genuine device_connect_edge modules and re-import the driver.

    Sibling test modules install MagicMock stand-ins in ``sys.modules`` for
    ``device_connect_edge`` at import time. A real module exposes ``__file__``;
    a MagicMock does not, so drop the fakes, re-import the real package from
    disk, and purge ``strands_robots.device_connect.*`` so it re-binds to the
    real ``@rpc`` / ``DeviceDriver``.
    """
    for key in (
        "device_connect_edge.drivers",
        "device_connect_edge.types",
        "device_connect_edge.device",
        "device_connect_edge",
    ):
        mod = sys.modules.get(key)
        if mod is not None and not hasattr(mod, "__file__"):
            sys.modules.pop(key, None)
    importlib.import_module("device_connect_edge")
    importlib.import_module("device_connect_edge.drivers")
    importlib.import_module("device_connect_edge.types")
    for key in list(sys.modules):
        if key.startswith("strands_robots.device_connect"):
            sys.modules.pop(key, None)


@pytest.fixture
def rmd() -> Any:
    """The reachy_mini_driver module bound to the real device_connect_edge."""
    _force_real_device_connect_edge()
    from strands_robots.device_connect import reachy_mini_driver as module

    return module


_HOST = "reachy.local"
_PORT = 8000


def _bare(rmd: Any) -> Any:
    """A driver instance without running ``__init__`` (no transport needed)."""
    driver = rmd.ReachyMiniDriver.__new__(rmd.ReachyMiniDriver)
    driver._host = _HOST
    driver._api_port = _PORT
    return driver


def _status_with(rmd: Any, monkeypatch: pytest.MonkeyPatch, payload: Any) -> dict[str, Any]:
    """Drive ``getDaemonStatus`` with ``payload`` standing in for the daemon body.

    The stand-in replaces :func:`reachy_transport.api` at the seam the driver
    reads it from, so the whole method -- including the merge -- runs.
    """
    monkeypatch.setattr(rmd, "api", lambda *_args, **_kwargs: payload)
    result: dict[str, Any] = asyncio.run(_bare(rmd).getDaemonStatus())
    return result


# A daemon status endpoint reporting a state of its own, in the three spellings
# that collide with the envelope's own key. None is exotic: "status" is the
# ordinary name for the field a /status endpoint answers with.
_COLLIDING_BODIES = [
    pytest.param({"status": "idle", "freq": 100}, id="idle"),
    pytest.param({"status": "error", "detail": "motor fault"}, id="error"),
    pytest.param({"status": "running", "motors_on": True}, id="running"),
]

# The shapes reachy_transport.api answers a failed call with. It reports a
# connection failure as ``{"error": str(exc)}`` and an HTTP error as
# ``{"error": <response body>, "code": <status>}`` -- and an error response with
# an empty body puts an empty string there, so presence rather than truthiness
# is what separates a failure from a reply.
_FAILURE_BODIES = [
    pytest.param({"error": "<urlopen error [Errno 111] Connection refused>"}, id="connection-refused"),
    pytest.param({"error": "service unavailable", "code": 503}, id="http-503"),
    pytest.param({"error": "", "code": 503}, id="http-503-empty-body"),
]

# Valid JSON that does not decode to an object.
_NON_OBJECT_BODIES = [
    pytest.param([1, 2, 3], "list", id="array"),
    pytest.param("ok", "str", id="string"),
    pytest.param(7, "int", id="number"),
    pytest.param(None, "NoneType", id="null"),
]


class TestADaemonReplyCannotOverwriteTheVerdict:
    """The envelope's ``status`` is this driver's, whatever the daemon calls its own."""

    @pytest.mark.parametrize("body", _COLLIDING_BODIES)
    def test_the_verdict_is_success(self, rmd: Any, monkeypatch: pytest.MonkeyPatch, body: dict[str, Any]) -> None:
        assert _status_with(rmd, monkeypatch, body)["status"] == "success"

    def test_no_envelope_reports_a_verdict_the_vocabulary_does_not_carry(
        self, rmd: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``idle`` is a legitimate daemon state and an illegitimate envelope verdict."""
        for body in ({"status": "idle"}, {"status": "running"}, {"status": "degraded"}):
            assert _status_with(rmd, monkeypatch, body)["status"] in {"success", "error"}


class TestReAssertingTheVerdictCostsNothingElse:
    """Over-reach guard. Holds on both trees: only ``status`` may change hands."""

    @pytest.mark.parametrize("body", _COLLIDING_BODIES)
    def test_the_other_payload_keys_still_arrive(
        self, rmd: Any, monkeypatch: pytest.MonkeyPatch, body: dict[str, Any]
    ) -> None:
        """Re-asserting one key must not cost the reading the caller asked for."""
        envelope = _status_with(rmd, monkeypatch, body)
        for key, value in body.items():
            if key != "status":
                assert envelope[key] == value

    @pytest.mark.parametrize("body", _COLLIDING_BODIES)
    def test_exactly_one_key_differs_from_the_daemon_payload(
        self, rmd: Any, monkeypatch: pytest.MonkeyPatch, body: dict[str, Any]
    ) -> None:
        """The envelope is the payload plus a verdict, not a filtered payload."""
        envelope = _status_with(rmd, monkeypatch, body)
        assert set(envelope) == set(body) | {"status"}


class TestADaemonThatWasNotReachedIsNotASuccess:
    """The RPC whose subject is daemon reachability can say the daemon is down."""

    @pytest.mark.parametrize("body", _FAILURE_BODIES)
    def test_the_verdict_is_error(self, rmd: Any, monkeypatch: pytest.MonkeyPatch, body: dict[str, Any]) -> None:
        assert _status_with(rmd, monkeypatch, body)["status"] == "error"

    @pytest.mark.parametrize("body", _FAILURE_BODIES)
    def test_the_reason_names_the_daemon_that_was_not_reached(
        self, rmd: Any, monkeypatch: pytest.MonkeyPatch, body: dict[str, Any]
    ) -> None:
        """An operator needs the address, not only that something went wrong."""
        assert f"{_HOST}:{_PORT}" in _status_with(rmd, monkeypatch, body)["reason"]

    @pytest.mark.parametrize("body", _FAILURE_BODIES)
    def test_the_reason_carries_the_transport_cause(
        self, rmd: Any, monkeypatch: pytest.MonkeyPatch, body: dict[str, Any]
    ) -> None:
        """An empty cause is carried vacuously, which is why the id is separate."""
        assert body["error"] in _status_with(rmd, monkeypatch, body)["reason"]

    @pytest.mark.parametrize("body", _FAILURE_BODIES)
    def test_a_success_verdict_never_travels_beside_a_transport_error(
        self, rmd: Any, monkeypatch: pytest.MonkeyPatch, body: dict[str, Any]
    ) -> None:
        """The self-contradicting envelope the RPC layer logs as ``PARTIAL``.

        ``device_connect_edge``'s ``@rpc`` wrapper classifies a result carrying
        an ``error`` key as partial and returns it to the caller unchanged, so
        the contradiction was observable and shipped anyway.
        """
        envelope = _status_with(rmd, monkeypatch, body)
        assert not (envelope["status"] == "success" and "error" in envelope)


#: Every REST RPC that carries a daemon reply back to the caller, with a body the
#: daemon really answers that endpoint with. The catalogue endpoint is declared
#: ``-> list[str]``, so its healthy body is an array rather than an object - the
#: shape that makes "not a mapping" unusable as the failure test.
_REST_VERBS = [
    pytest.param("playMove", lambda d: d.playMove("happy"), {"ok": True}, id="playMove"),
    pytest.param("listMoves", lambda d: d.listMoves(), ["happy", "sad"], id="listMoves"),
    pytest.param("wakeUp", lambda d: d.wakeUp(), {"ok": True}, id="wakeUp"),
    pytest.param("sleep", lambda d: d.sleep(), {"ok": True}, id="sleep"),
    pytest.param("getDaemonStatus", lambda d: d.getDaemonStatus(), {"ok": True}, id="getDaemonStatus"),
]


def _authorized(rmd: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Satisfy the authorization gate that sits in front of the daemon read.

    An unset ``DEVICE_CONNECT_RPC_ALLOW`` authorizes nobody, so without this
    every gated verb answers ``authz_error`` - an error envelope that would
    satisfy the failure rows below while the daemon read never ran.
    """
    monkeypatch.setenv("DEVICE_CONNECT_RPC_ALLOW", "operator")
    monkeypatch.setattr(rmd, "get_rpc_source_device", lambda: "operator")


def _rest_reply(rmd: Any, monkeypatch: pytest.MonkeyPatch, call: Any, body: Any) -> dict[str, Any]:
    """Drive one REST RPC with ``body`` standing in for the daemon's reply."""
    _authorized(rmd, monkeypatch)
    monkeypatch.setattr(rmd, "api", lambda *_args, **_kwargs: body)
    result: dict[str, Any] = asyncio.run(call(_bare(rmd)))
    return result


class TestNoRestVerbReportsACallThatDidNotLandAsASuccess:
    """The rule ``getDaemonStatus`` established, over every verb that reads the daemon."""

    @pytest.mark.parametrize(("verb", "call", "healthy"), _REST_VERBS)
    @pytest.mark.parametrize("body", _FAILURE_BODIES)
    def test_the_verdict_is_error(
        self, rmd: Any, monkeypatch: pytest.MonkeyPatch, verb: str, call: Any, healthy: Any, body: dict[str, Any]
    ) -> None:
        assert _rest_reply(rmd, monkeypatch, call, body)["status"] == "error"

    @pytest.mark.parametrize(("verb", "call", "healthy"), _REST_VERBS)
    @pytest.mark.parametrize("body", _FAILURE_BODIES)
    def test_the_reason_names_the_verb_the_daemon_and_the_cause(
        self, rmd: Any, monkeypatch: pytest.MonkeyPatch, verb: str, call: Any, healthy: Any, body: dict[str, Any]
    ) -> None:
        """Which call did not land, against which address, and why."""
        reason = _rest_reply(rmd, monkeypatch, call, body)["reason"]
        assert verb in reason
        assert f"{_HOST}:{_PORT}" in reason
        assert body["error"] in reason

    @pytest.mark.parametrize(("verb", "call", "healthy"), _REST_VERBS)
    @pytest.mark.parametrize("body", _FAILURE_BODIES)
    def test_no_verb_reports_the_motion_it_did_not_command(
        self, rmd: Any, monkeypatch: pytest.MonkeyPatch, verb: str, call: Any, healthy: Any, body: dict[str, Any]
    ) -> None:
        """``move``/``moves``/``result`` are readings of a call that never landed."""
        envelope = _rest_reply(rmd, monkeypatch, call, body)
        assert not (set(envelope) & {"move", "moves", "result"})

    @pytest.mark.parametrize(("verb", "call", "healthy"), _REST_VERBS)
    def test_a_daemon_that_answered_is_still_a_success(
        self, rmd: Any, monkeypatch: pytest.MonkeyPatch, verb: str, call: Any, healthy: Any
    ) -> None:
        """Non-vacuity, and the over-reach guard.

        The failure rows above are only about the daemon read if the same call,
        through the same helper, reaches it when the daemon answers. This cell
        holds on both trees: the reading a caller already gets is unchanged.
        """
        assert _rest_reply(rmd, monkeypatch, call, healthy)["status"] == "success"

    def test_an_unset_allowlist_would_have_refused_these_verbs_first(
        self, rmd: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Why :func:`_authorized` exists: the gate in front answers first.

        Without it the failure rows would be graded against ``authz_error``,
        which is an error envelope for a daemon read that never happened.
        """
        monkeypatch.delenv("DEVICE_CONNECT_RPC_ALLOW", raising=False)
        monkeypatch.setattr(rmd, "api", lambda *_a, **_k: {"ok": True})
        envelope = asyncio.run(_bare(rmd).wakeUp())
        assert envelope["status"] == "error"
        assert "not authorized" in envelope["reason"]

    def test_the_stop_verb_keeps_its_stronger_refusal(self, rmd: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """The one member that raises instead, and why it is not converted.

        A caller acting on a false success from a *stop* stops nothing, so that
        failure must not be catchable as a result - the criterion
        :meth:`ReachyMiniDriver._transport_failure` records.
        """
        _authorized(rmd, monkeypatch)
        monkeypatch.setattr(rmd, "api", lambda *_a, **_k: {"error": "boom"})
        with pytest.raises(RuntimeError, match="transport failure"):
            asyncio.run(_bare(rmd).stopMotion())


class TestEveryDaemonReadInThisDriverConsultsTheReply:
    """A drift guard: the next REST verb inherits the rule without being listed."""

    def test_no_method_returns_without_judging_the_reply(self, rmd: Any) -> None:
        assert _daemon_reads_that_ignore_the_reply(rmd) == set()

    def test_the_scan_sees_the_daemon_reads_at_all(self, rmd: Any) -> None:
        """Non-vacuity: the scan must find the family it grades."""
        assert {"playMove", "listMoves", "wakeUp", "sleep", "getDaemonStatus"} <= _methods_that_read_the_daemon(rmd)

    def test_the_rule_grades_constructed_exemplars(self) -> None:
        """The shipped class satisfies the rule, so grade the rule directly too."""
        ignores = 'async def f(self):\n    r = await asyncio.to_thread(api, h, p, "/x")\n    return {"result": r}\n'
        judges = (
            'async def f(self):\n    r = await asyncio.to_thread(api, h, p, "/x")\n'
            '    if (bad := self._transport_failure(r, "f")) is not None:\n        return bad\n'
            '    return {"result": r}\n'
        )
        assert _ignores_the_reply(ast.parse(ignores).body[0]) is True
        assert _ignores_the_reply(ast.parse(judges).body[0]) is False


class TestABodyThatIsNotAnObjectIsReportedNotRaised:
    """Valid JSON that is not a mapping cannot be spread into the envelope."""

    @pytest.mark.parametrize(("body", "type_name"), _NON_OBJECT_BODIES)
    def test_the_envelope_is_returned(
        self, rmd: Any, monkeypatch: pytest.MonkeyPatch, body: Any, type_name: str
    ) -> None:
        envelope = _status_with(rmd, monkeypatch, body)
        assert envelope["status"] == "error"

    @pytest.mark.parametrize(("body", "type_name"), _NON_OBJECT_BODIES)
    def test_the_reason_names_the_type_that_could_not_be_merged(
        self, rmd: Any, monkeypatch: pytest.MonkeyPatch, body: Any, type_name: str
    ) -> None:
        assert type_name in _status_with(rmd, monkeypatch, body)["reason"]


class TestTheHealthyPayloadIsUnchanged:
    """The reading a caller already gets. Every expectation here held pre-fix."""

    @pytest.mark.parametrize(
        "body",
        [
            pytest.param({"motors_on": True, "freq": 100}, id="motors-and-freq"),
            pytest.param({"state": "ready", "version": "1.0"}, id="state-and-version"),
            pytest.param({"wireless_version": True}, id="variant-flag"),
            pytest.param({}, id="empty-object"),
        ],
    )
    def test_every_payload_key_stays_at_the_top_level(
        self, rmd: Any, monkeypatch: pytest.MonkeyPatch, body: dict[str, Any]
    ) -> None:
        envelope = _status_with(rmd, monkeypatch, body)
        assert envelope == {**body, "status": "success"}

    def test_the_endpoint_and_verb_are_unchanged(self, rmd: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}

        def fake_api(host: str, port: int, path: str, method: str = "GET", data: Any = None) -> dict[str, Any]:
            seen.update(host=host, port=port, path=path, method=method)
            return {"ok": True}

        monkeypatch.setattr(rmd, "api", fake_api)
        asyncio.run(_bare(rmd).getDaemonStatus())
        assert seen == {"host": _HOST, "port": _PORT, "path": "/api/daemon/status", "method": "GET"}


class TestWhyTheEnvelopeOwnsTheVerdict:
    """Premises. Each holds on both trees; together they pick the owner."""

    def test_api_reports_an_unreachable_daemon_as_a_result_rather_than_raising(self) -> None:
        """The real transport, against a port nothing is listening on."""
        from strands_robots.device_connect.reachy_transport import api

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            closed_port = probe.getsockname()[1]

        result = api("127.0.0.1", closed_port, "/api/daemon/status")
        assert isinstance(result, dict)
        assert "error" in result

    def test_the_stop_rpc_refuses_the_same_reply_shape(self, rmd: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """The sibling that already declines to call a failed call a success."""
        monkeypatch.setattr(rmd, "api", lambda *_a, **_k: {"error": "boom"})
        with pytest.raises(RuntimeError, match="transport failure"):
            asyncio.run(_bare(rmd)._stop_motion_impl())

    def test_this_is_the_only_rpc_that_merges_the_reply_into_its_envelope(self, rmd: Any) -> None:
        """Its four REST siblings nest the reply, so no spread can reach their keys."""
        assert _envelopes_that_spread_a_mapping(rmd) == {"getDaemonStatus"}


def _class_body(rmd: Any) -> ast.ClassDef:
    """The ``ReachyMiniDriver`` class node, parsed from its own source file."""
    source = Path(rmd.__file__).read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef) and node.name == "ReachyMiniDriver":
            return node
    raise AssertionError("ReachyMiniDriver not found in the driver module")


def _returned_dicts(function: ast.AST) -> list[ast.Dict]:
    """Every dict literal this function returns directly."""
    return [
        node.value for node in ast.walk(function) if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
    ]


def _merges_a_foreign_mapping_before_its_own_status(literal: ast.Dict) -> bool:
    """``True`` when a spread can reach a ``status`` key declared ahead of it.

    A ``**`` spread appears as a ``None`` key. A ``status`` key declared before
    one is overwritten by whatever the spread mapping carries under that name.
    """
    spreads = [i for i, key in enumerate(literal.keys) if key is None]
    if not spreads:
        return False
    status = [i for i, key in enumerate(literal.keys) if isinstance(key, ast.Constant) and key.value == "status"]
    return bool(status) and min(status) < max(spreads)


def _envelopes_that_spread_a_mapping(rmd: Any) -> set[str]:
    """Methods of the driver whose returned envelope merges a mapping at all.

    A spread is what makes the precedence question reachable: the four REST
    siblings nest the daemon's reply under ``result``, so nothing it carries can
    land on a key of the envelope's own.
    """
    merging = set()
    for member in _class_body(rmd).body:
        if not isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for literal in _returned_dicts(member):
            if any(key is None for key in literal.keys):
                merging.add(member.name)
    return merging


def _calls_the_daemon(function: ast.AST) -> bool:
    """``True`` when this method calls :func:`reachy_transport.api`.

    Both spellings count: a direct call, and the ``asyncio.to_thread(api, ...)``
    hand-off every async RPC uses to keep the blocking HTTP call off the loop.
    """
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "api":
            return True
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "to_thread"
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "api"
        ):
            return True
    return False


def _judges_the_reply(function: ast.AST) -> bool:
    """``True`` when this method decides whether the daemon's reply landed.

    Three spellings are accepted, because the driver uses all three: the shared
    :meth:`ReachyMiniDriver._transport_failure`, the ``"error" in result`` test
    the stop raises on, and a direct ``result.get("error")``.
    """
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "_transport_failure":
                return True
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "get"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "error"
            ):
                return True
        if (
            isinstance(node, ast.Compare)
            and isinstance(node.ops[0], ast.In)
            and isinstance(node.left, ast.Constant)
            and node.left.value == "error"
        ):
            return True
    return False


def _ignores_the_reply(function: ast.AST) -> bool:
    """``True`` for a method that reads the daemon and never judges the reply."""
    return _calls_the_daemon(function) and not _judges_the_reply(function)


def _methods_that_read_the_daemon(rmd: Any) -> set[str]:
    """Every method of the driver that calls the daemon's REST API."""
    return {
        member.name
        for member in _class_body(rmd).body
        if isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef) and _calls_the_daemon(member)
    }


def _daemon_reads_that_ignore_the_reply(rmd: Any) -> set[str]:
    """Every method that reads the daemon and reports without judging the reply.

    ``connect`` is exempt: it reads the status endpoint only to pick a link
    variant, inside a ``try`` whose fallback is the Wireless link, and answers no
    envelope at all.
    """
    return {
        member.name
        for member in _class_body(rmd).body
        if isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef)
        and member.name != "connect"
        and _ignores_the_reply(member)
    }


def _envelopes_that_merge_a_foreign_mapping(rmd: Any) -> set[str]:
    """Methods of the driver whose returned envelope a spread can overwrite."""
    offenders = set()
    for member in _class_body(rmd).body:
        if not isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if any(_merges_a_foreign_mapping_before_its_own_status(d) for d in _returned_dicts(member)):
            offenders.add(member.name)
    return offenders


class TestNoEnvelopeInThisDriverDeclaresStatusAheadOfASpread:
    """A drift guard: the next RPC to merge a reply is held to the same rule."""

    def test_no_method_declares_status_before_a_spread(self, rmd: Any) -> None:
        assert _envelopes_that_merge_a_foreign_mapping(rmd) == set()

    def test_the_scan_reaches_the_method_this_file_is_about(self, rmd: Any) -> None:
        """Non-vacuity: the scan must see a spread-bearing envelope at all."""
        assert "getDaemonStatus" in _envelopes_that_spread_a_mapping(rmd)

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            pytest.param('def f():\n    return {"status": "success", **result}\n', True, id="status-first"),
            pytest.param('def f():\n    return {**result, "status": "success"}\n', False, id="status-last"),
            pytest.param('def f():\n    return {"status": "success", "result": result}\n', False, id="nested"),
            pytest.param("def f():\n    return {**result}\n", False, id="no-status-key"),
            pytest.param(
                'def f():\n    return {"status": "s", **a, "status": "s", **b}\n', True, id="status-between-spreads"
            ),
        ],
    )
    def test_the_rule_grades_constructed_exemplars(self, source: str, expected: bool) -> None:
        """The shipped class satisfies the rule, so grade the rule directly too."""
        function = ast.parse(source).body[0]
        literals = _returned_dicts(function)
        assert len(literals) == 1
        assert _merges_a_foreign_mapping_before_its_own_status(literals[0]) is expected

    def test_the_exemplars_reach_both_outcomes(self) -> None:
        cases = [
            'def f():\n    return {"status": "success", **result}\n',
            'def f():\n    return {**result, "status": "success"}\n',
        ]
        outcomes = {
            _merges_a_foreign_mapping_before_its_own_status(_returned_dicts(ast.parse(s).body[0])[0]) for s in cases
        }
        assert outcomes == {True, False}
