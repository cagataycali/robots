#!/usr/bin/env python3
"""``Mesh.send`` and the command reply over a ``DirectSender`` transport.

No AWS, no network, no awsiot. A hand-built transport that satisfies both
``MeshTransport`` and ``DirectSender`` records every ``send_direct`` call and
answers a scripted ``DirectResult``, so each test pins one arm of the Mesh
side of AWS IoT Core Direct Messaging:

  - ``send`` goes point to point first: confirmation on, whole-second timeout
    clamped to ``[1, 10]``, Response Topic ``strands/{self}/response/{target}/{turn}``,
    the turn as Correlation Data, and NO publish when delivered;
  - an ``offline`` target answers at once with the error envelope, the turn
    is freed, and nothing is published;
  - ``forbidden`` falls back to publish and is reported once per peer;
    ``throttled`` / ``unconfirmed`` / ``error`` / ``unavailable`` fall back to
    publish for that call;
  - a transport whose memo already says forbidden is not asked again;
  - ``STRANDS_MESH_IOT_DIRECT=0`` leaves the transport unused;
  - the reply to a command that carried a Response Topic goes out as a
    direct message to the validated sender, and ONLY when the topic is
    exactly ``strands/{sender}/response/{self}/{turn}``: a topic naming another
    operator, another robot, another turn or a stray suffix is refused at
    WARNING and the reply is published on the computed key as before;
  - a Zenoh-shaped session (no ``send_direct`` on its class) leaves ``send``
    and the reply path byte for byte on ``publish``.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from strands_robots.mesh import core as mesh_core
from strands_robots.mesh import session as mesh_session
from strands_robots.mesh.core import Mesh
from strands_robots.mesh.transport.base import DirectResult, DirectSender
from strands_robots.mesh.transport.iot_transport import DIRECT_ENV_VAR

TURN_RE = r"[0-9a-f]{32}"


class _FakeRobot:
    tool_name_str = "fakebot"

    def get_task_status(self) -> dict[str, Any]:
        return {"status": "idle"}


class _DirectTransport:
    """Satisfies MeshTransport and DirectSender; records everything."""

    def __init__(self, script: list[DirectResult] | None = None) -> None:
        self.script = list(script or [])
        self.direct_calls: list[dict[str, Any]] = []
        self.forbidden: set[str] = set()
        self.subscribed: list[str] = []

    # MeshTransport
    def put(self, key: str, data: dict[str, Any]) -> None:
        pass

    def declare_subscriber(self, key_expr: str, handler: Any) -> Any:
        self.subscribed.append(key_expr)
        return MagicMock()

    def is_alive(self) -> bool:
        return True

    def close(self) -> None:
        pass

    # DirectSender
    def send_direct(
        self,
        peer_id: str,
        key: str,
        data: dict[str, Any],
        *,
        confirm: bool = False,
        timeout: float = 5.0,
        response_key: str | None = None,
        correlation: str | None = None,
    ) -> DirectResult:
        self.direct_calls.append(
            {
                "peer_id": peer_id,
                "key": key,
                "data": data,
                "confirm": confirm,
                "timeout": timeout,
                "response_key": response_key,
                "correlation": correlation,
            }
        )
        if self.script:
            return self.script.pop(0)
        return DirectResult(delivered=True, reason="", latency_ms=1.0)

    def direct_forbidden(self, peer_id: str) -> bool:
        return peer_id in self.forbidden


def _ok() -> DirectResult:
    return DirectResult(delivered=True, reason="", latency_ms=1.0)


def _fail(reason: str, detail: str = "") -> DirectResult:
    return DirectResult(delivered=False, reason=reason, latency_ms=1.0, detail=detail)


@pytest.fixture
def puts() -> Iterator[list[tuple[str, dict[str, Any]]]]:
    seen: list[tuple[str, dict[str, Any]]] = []
    with patch.object(mesh_core, "put", side_effect=lambda k, d: seen.append((k, d))):
        yield seen


def _start(session: Any, peer_id: str = "operator-1") -> Mesh:
    patches = (
        patch.object(mesh_session, "get_session", return_value=session),
        patch.object(mesh_session, "current_session", return_value=session),
        patch.object(mesh_core, "get_session", return_value=session),
        patch.object(mesh_core, "current_session", return_value=session),
        patch.object(mesh_core, "release_session"),
    )
    for p in patches:
        p.start()
    m = Mesh(_FakeRobot(), peer_id=peer_id, peer_type="operator")
    m.start()
    m._test_patches = patches  # type: ignore[attr-defined]
    return m


def _stop(m: Mesh) -> None:
    m.stop()
    for p in m._test_patches:  # type: ignore[attr-defined]
        p.stop()


def _answer_later(m: Mesh, turn_holder: dict[str, str], transport: _DirectTransport) -> threading.Thread:
    """Feed the response for the turn the send is about to create."""

    def _run() -> None:
        deadline = threading.Event()
        for _ in range(200):
            if transport.direct_calls:
                break
            deadline.wait(0.01)
        call = transport.direct_calls[-1]
        turn = call["correlation"]
        turn_holder["turn"] = turn
        m._on_response(
            _sample(
                call["response_key"], {"responder_id": "so101", "turn_id": turn, "type": "result", "result": {"ok": 1}}
            )
        )

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def _sample(key: str, payload: dict[str, Any], response_topic: str | None = None) -> Any:
    s = MagicMock(spec=["key_expr", "payload"])
    s.key_expr = key
    s.payload.to_bytes.return_value = json.dumps(payload).encode()
    if response_topic is not None:
        s.response_topic = response_topic
    return s


class TestSendGoesDirectFirst:
    def test_delivered_means_no_publish_and_the_wire_shape(self, puts):
        t = _DirectTransport([_ok()])
        m = _start(t)
        try:
            holder: dict[str, str] = {}
            th = _answer_later(m, holder, t)
            out = m.send("so101", {"action": "status"}, timeout=7.6)
            th.join(2)
        finally:
            _stop(m)
        assert out.get("type") == "result", out
        call = t.direct_calls[0]
        assert call["peer_id"] == "so101"
        assert call["key"] == "strands/so101/cmd"
        assert call["confirm"] is True
        assert call["timeout"] == 7.6
        assert call["response_key"] == f"strands/operator-1/response/so101/{holder['turn']}"
        assert call["correlation"] == holder["turn"]
        assert call["data"]["sender_id"] == "operator-1"
        assert call["data"]["turn_id"] == holder["turn"]
        assert call["data"]["command"] == {"action": "status"}
        assert [k for k, _ in puts if k.endswith("/cmd")] == []

    @pytest.mark.parametrize(("timeout", "expected"), [(0.3, 1.0), (30.0, 10.0), (4.0, 4.0)])
    def test_confirmation_timeout_is_clamped_to_one_to_ten(self, puts, timeout, expected):
        # ``offline`` so the call returns at once instead of waiting the budget.
        t = _DirectTransport([_fail("offline")])
        m = _start(t)
        try:
            m.send("so101", {"action": "status"}, timeout=timeout)
        finally:
            _stop(m)
        assert t.direct_calls[0]["timeout"] == expected

    def test_offline_target_answers_at_once_and_publishes_nothing(self, puts):
        t = _DirectTransport([_fail("offline", "not connected")])
        m = _start(t)
        try:
            out = m.send("so101", {"action": "status"}, timeout=30.0)
            assert out == {"status": "error", "error": "peer offline (iot 404)", "peer": "so101"}
            assert m._pending == {} and m._expected_responders == {} and m._responses == {}
        finally:
            _stop(m)
        assert [k for k, _ in puts if k.endswith("/cmd")] == []

    @pytest.mark.parametrize("reason", ["throttled", "unconfirmed", "error", "unavailable", "too_large"])
    def test_other_failures_fall_back_to_publish_for_this_call(self, puts, reason):
        t = _DirectTransport([_fail(reason)])
        m = _start(t)
        try:
            out = m.send("so101", {"action": "status"}, timeout=0.05)
        finally:
            _stop(m)
        assert out == {"status": "timeout"}
        cmd_puts = [(k, d) for k, d in puts if k == "strands/so101/cmd"]
        assert len(cmd_puts) == 1
        assert cmd_puts[0][1]["command"] == {"action": "status"}

    def test_forbidden_falls_back_and_is_reported_once_per_peer(self, puts, caplog):
        t = _DirectTransport([_fail("forbidden", "Authorization failed"), _fail("forbidden", "Authorization failed")])
        m = _start(t)
        try:
            with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.core"):
                m.send("so101", {"action": "status"}, timeout=0.05)
                m.send("so101", {"action": "status"}, timeout=0.05)
        finally:
            _stop(m)
        warnings = [r for r in caplog.records if "refused by policy" in r.getMessage()]
        assert len(warnings) == 1
        assert "CN=<thing name>" in warnings[0].getMessage()
        assert len([k for k, _ in puts if k == "strands/so101/cmd"]) == 2

    def test_a_peer_the_transport_remembers_as_forbidden_is_not_asked_again(self, puts):
        t = _DirectTransport()
        t.forbidden.add("so101")
        m = _start(t)
        try:
            m.send("so101", {"action": "status"}, timeout=0.05)
        finally:
            _stop(m)
        assert t.direct_calls == []
        assert len([k for k, _ in puts if k == "strands/so101/cmd"]) == 1

    def test_switch_off_leaves_the_transport_unused(self, puts, monkeypatch):
        monkeypatch.setenv(DIRECT_ENV_VAR, "0")
        t = _DirectTransport()
        m = _start(t)
        try:
            assert m._direct is None
            m.send("so101", {"action": "status"}, timeout=0.05)
        finally:
            _stop(m)
        assert t.direct_calls == []
        assert len([k for k, _ in puts if k == "strands/so101/cmd"]) == 1

    def test_the_subscriptions_are_kept(self, puts):
        t = _DirectTransport()
        m = _start(t)
        try:
            assert "strands/operator-1/cmd" in t.subscribed
            assert "strands/operator-1/response/**" in t.subscribed
        finally:
            _stop(m)

    def test_broadcast_stays_on_publish(self, puts):
        t = _DirectTransport()
        m = _start(t)
        try:
            m.broadcast({"action": "status"}, timeout=0.05)
        finally:
            _stop(m)
        assert t.direct_calls == []
        assert [k for k, _ in puts if k == "strands/broadcast"]


def _exec_and_wait(m: Mesh, payload: dict[str, Any], response_topic: str | None) -> None:
    done = threading.Event()
    real = m._exec_cmd

    def _wrapped(data: dict[str, Any], reply_to: str | None = None) -> None:
        try:
            real(data, reply_to=reply_to)
        finally:
            done.set()

    m._exec_cmd = _wrapped  # type: ignore[method-assign]
    m._on_cmd(_sample("strands/so101/cmd", payload, response_topic))
    assert done.wait(5.0), "command never dispatched"


class TestReplyOverTheResponseTopic:
    TURN = "c" * 32

    def _payload(self) -> dict[str, Any]:
        return {"sender_id": "operator-1", "turn_id": self.TURN, "command": {"action": "status"}}

    def test_valid_response_topic_gets_a_direct_reply_and_no_publish(self, puts):
        t = _DirectTransport([_ok()])
        m = _start(t, peer_id="so101")
        try:
            _exec_and_wait(m, self._payload(), f"strands/operator-1/response/so101/{self.TURN}")
        finally:
            _stop(m)
        assert len(t.direct_calls) == 1
        call = t.direct_calls[0]
        assert call["peer_id"] == "operator-1"
        assert call["key"] == f"strands/operator-1/response/so101/{self.TURN}"
        assert call["correlation"] == self.TURN
        assert call["confirm"] is False
        assert call["data"]["turn_id"] == self.TURN
        assert call["data"]["responder_id"] == "so101"
        assert [k for k, _ in puts if "/response/" in k] == []

    def test_undelivered_direct_reply_is_published_on_the_computed_key(self, puts):
        t = _DirectTransport([_fail("error", "socket")])
        m = _start(t, peer_id="so101")
        try:
            _exec_and_wait(m, self._payload(), f"strands/operator-1/response/so101/{self.TURN}")
        finally:
            _stop(m)
        assert len(t.direct_calls) == 1
        assert [k for k, _ in puts if k == f"strands/operator-1/response/so101/{self.TURN}"]

    @pytest.mark.parametrize(
        "bad",
        [
            "strands/operator-2/response/so101/" + "c" * 32,  # another operator
            "strands/operator-1/response/so102/" + "c" * 32,  # another robot's segment
            "strands/operator-1/response/so101/" + "d" * 32,  # another turn
            "strands/operator-1/response/so101/" + "c" * 32 + "/x",  # stray suffix
            "strands/operator-1/response/so101/" + "c" * 32 + "\n",  # trailing newline ($ would accept it)
            "strands/operator-1/cmd",  # not a response topic
            "",
        ],
        ids=["other-operator", "other-robot", "other-turn", "suffix", "newline", "cmd", "empty"],
    )
    def test_a_response_topic_that_is_not_ours_is_refused_and_published(self, puts, caplog, bad):
        t = _DirectTransport([_ok()])
        m = _start(t, peer_id="so101")
        try:
            with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.core"):
                _exec_and_wait(m, self._payload(), bad)
        finally:
            _stop(m)
        assert t.direct_calls == []
        assert [k for k, _ in puts if k == f"strands/operator-1/response/so101/{self.TURN}"]
        assert any("does not name" in r.getMessage() for r in caplog.records)

    def test_no_response_topic_publishes_as_before(self, puts):
        t = _DirectTransport([_ok()])
        m = _start(t, peer_id="so101")
        try:
            _exec_and_wait(m, self._payload(), None)
        finally:
            _stop(m)
        assert t.direct_calls == []
        assert [k for k, _ in puts if k == f"strands/operator-1/response/so101/{self.TURN}"]


class TestZenohIsUntouched:
    def test_a_session_without_send_direct_on_its_class_publishes_everything(self, puts):
        sess = MagicMock(spec=["put", "declare_subscriber", "is_alive", "close"])
        assert not callable(getattr(type(sess), "send_direct", None))
        m = _start(sess)
        try:
            assert m._direct is None
            m.send("so101", {"action": "status"}, timeout=0.05)
            _exec_and_wait(m, {"sender_id": "peer-b", "turn_id": "e" * 32, "command": {"action": "status"}}, None)
        finally:
            _stop(m)
        assert [k for k, _ in puts if k == "strands/so101/cmd"]
        assert [k for k, _ in puts if k == "strands/peer-b/response/operator-1/" + "e" * 32]

    def test_stop_forgets_the_direct_transport(self, puts):
        t = _DirectTransport()
        m = _start(t)
        assert isinstance(m._direct, DirectSender)
        _stop(m)
        assert m._direct is None
