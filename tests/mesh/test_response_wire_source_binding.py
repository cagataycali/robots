"""Regression tests: an RPC response is attributed to the session that sent it.

``Mesh._on_response`` used to authorise a reply from the ``responder_id`` field
in the JSON body alone, and a broadcast turn skipped even that comparison. Any
admitted mesh member could therefore answer a fleet emergency stop in another
robot's name and the operator saw a clean "fleet halted" verdict for a robot
that never received the stop.

The fix binds every reply to the TLS-bound wire identity Zenoh attaches to the
sample (``sample.source_info.source_id.zid``), learned per peer from the
presence path, with the same three-state rule the safety envelopes already use:
both present and equal, both absent (bridge and IoT transports never carry a
wire zid), or refuse. A broadcast turn accepts one reply per verified session,
and ``emergency_stop`` reports the peers that produced no acknowledgement.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from strands_robots.mesh import core as mesh_core
from strands_robots.mesh import session as mesh_session
from strands_robots.mesh.core import BROADCAST_RESPONDER, Mesh

_ZID_B = "0123456789abcdef0123456789abcdef"
_ZID_EVIL = "fedcba9876543210fedcba9876543210"


class _FakeRobot:
    def __init__(self) -> None:
        self.tool_name_str = "fakebot"

    def stop_task(self) -> dict[str, object]:
        return {"status": "success", "content": [{"text": "stopped"}]}


class _Zid:
    def __init__(self, text: str) -> None:
        self._text = text

    def __str__(self) -> str:
        return self._text


def _sample(payload: dict[str, Any], *, zid: str | None, key: str = "") -> Any:
    """A zenoh-shaped sample; ``zid=None`` means no ``source_info`` (bridge path)."""
    body = SimpleNamespace(to_bytes=lambda: json.dumps(payload).encode())
    source_info = None if zid is None else SimpleNamespace(source_id=SimpleNamespace(zid=_Zid(zid)))
    return SimpleNamespace(payload=body, source_info=source_info, key_expr=key)


def _presence(peer: str, *, zid: str | None) -> Any:
    return _sample(
        {"robot_id": peer, "robot_type": "robot", "hostname": "h", "timestamp": time.time()},
        zid=zid,
        key=f"strands/{peer}/presence",
    )


def _response(turn: str, responder: str, *, zid: str | None, me: str = "op") -> Any:
    return _sample(
        {"type": "response", "turn_id": turn, "responder_id": responder, "result": {"status": "success"}},
        zid=zid,
        key=f"strands/{me}/response/{responder}/{turn}",
    )


def _register(m: Mesh, turn: str, expected: str) -> threading.Event:
    event = threading.Event()
    with m._rpc_lock:
        m._pending[turn] = event
        m._responses[turn] = []
        m._expected_responders[turn] = expected
    return event


@pytest.fixture
def mesh() -> Mesh:
    return Mesh(_FakeRobot(), peer_id="op", peer_type="operator")


@pytest.fixture
def audits(monkeypatch: pytest.MonkeyPatch, mesh: Mesh) -> list[tuple[str, dict[str, Any]]]:
    seen: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(mesh, "_audit_local", lambda event, payload: seen.append((event, payload)))
    return seen


@pytest.fixture(autouse=True)
def _clean_registry() -> Any:
    with mesh_session._PEERS_LOCK:
        mesh_session._PEERS.clear()
    yield
    with mesh_session._PEERS_LOCK:
        mesh_session._PEERS.clear()


class TestPresenceLearnsTheBinding:
    def test_presence_binds_peer_id_to_the_wire_zid(self, mesh: Mesh) -> None:
        mesh._on_presence(_presence("robot-b", zid=_ZID_B))

        assert mesh.peer_wire_zid("robot-b") == _ZID_B

    def test_presence_without_wire_zid_leaves_the_peer_unbound(self, mesh: Mesh) -> None:
        mesh._on_presence(_presence("robot-b", zid=None))

        assert mesh.peer_wire_zid("robot-b") is None

    def test_a_second_session_cannot_rebind_a_live_peer(self, mesh: Mesh, audits: list) -> None:
        mesh._on_presence(_presence("robot-b", zid=_ZID_B))
        mesh._on_presence(_presence("robot-b", zid=_ZID_EVIL))

        assert mesh.peer_wire_zid("robot-b") == _ZID_B
        assert [e for e, _ in audits] == ["presence_identity_conflict"]
        assert audits[0][1]["peer_id"] == "robot-b"

    def test_a_peer_that_went_silent_may_come_back_on_a_new_session(
        self, mesh: Mesh, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mesh._on_presence(_presence("robot-b", zid=_ZID_B))
        later = time.monotonic() + mesh_session.PEER_TIMEOUT + 1.0
        monkeypatch.setattr(time, "monotonic", lambda: later)

        mesh._on_presence(_presence("robot-b", zid=_ZID_EVIL))

        assert mesh.peer_wire_zid("robot-b") == _ZID_EVIL


class TestBroadcastTurn:
    def test_reply_from_another_session_in_a_bound_peers_name_is_refused(
        self, mesh: Mesh, audits: list, caplog: pytest.LogCaptureFixture
    ) -> None:
        mesh._on_presence(_presence("robot-b", zid=_ZID_B))
        event = _register(mesh, "t1", BROADCAST_RESPONDER)

        with caplog.at_level(logging.WARNING):
            mesh._on_response(_response("t1", "robot-b", zid=_ZID_EVIL))

        assert mesh._responses["t1"] == []
        assert not event.is_set()
        assert [e for e, _ in audits] == ["response_hijack_rejected"]
        assert audits[0][1]["responder_id"] == "robot-b"
        assert any("source" in rec.message for rec in caplog.records)

    def test_reply_with_a_wire_zid_from_a_peer_never_seen_is_refused(self, mesh: Mesh, audits: list) -> None:
        event = _register(mesh, "t2", BROADCAST_RESPONDER)

        mesh._on_response(_response("t2", "ghost", zid=_ZID_EVIL))

        assert mesh._responses["t2"] == []
        assert not event.is_set()
        assert [e for e, _ in audits] == ["response_hijack_rejected"]

    def test_reply_from_the_bound_session_is_recorded(self, mesh: Mesh, audits: list) -> None:
        mesh._on_presence(_presence("robot-b", zid=_ZID_B))
        event = _register(mesh, "t3", BROADCAST_RESPONDER)

        mesh._on_response(_response("t3", "robot-b", zid=_ZID_B))

        assert [r["responder_id"] for r in mesh._responses["t3"]] == ["robot-b"]
        assert event.is_set()
        assert audits == []

    def test_one_reply_per_session_per_turn(self, mesh: Mesh, audits: list) -> None:
        mesh._on_presence(_presence("robot-b", zid=_ZID_B))
        _register(mesh, "t4", BROADCAST_RESPONDER)

        mesh._on_response(_response("t4", "robot-b", zid=_ZID_B))
        mesh._on_response(_response("t4", "robot-b", zid=_ZID_B))

        assert len(mesh._responses["t4"]) == 1
        assert [e for e, _ in audits] == ["response_duplicate_rejected"]

    def test_bound_peer_replying_without_a_wire_zid_is_refused(self, mesh: Mesh, audits: list) -> None:
        """A stripped SourceInfo on a peer we know by session is a forgery, not a legacy publisher."""
        mesh._on_presence(_presence("robot-b", zid=_ZID_B))
        _register(mesh, "t5", BROADCAST_RESPONDER)

        mesh._on_response(_response("t5", "robot-b", zid=None))

        assert mesh._responses["t5"] == []
        assert [e for e, _ in audits] == ["response_hijack_rejected"]

    def test_a_reply_nobody_can_attribute_is_refused_on_zenoh(self, mesh: Mesh, audits: list) -> None:
        """No wire zid on the sample and none bound to the name: on Zenoh that is anyone's reply."""
        mesh._on_presence(_presence("robot-b", zid=None))
        event = _register(mesh, "t6", BROADCAST_RESPONDER)

        mesh._on_response(_response("t6", "robot-b", zid=None))

        assert mesh._responses["t6"] == []
        assert not event.is_set()
        assert [e for e, _ in audits] == ["response_hijack_rejected"]
        assert "does not bind the sender" in audits[0][1]["reason"]

    @pytest.mark.parametrize("backend", ["iot", "bridge"])
    def test_a_broker_that_binds_the_topic_still_accepts_zid_less_replies(
        self, mesh: Mesh, audits: list, monkeypatch: pytest.MonkeyPatch, backend: str
    ) -> None:
        """The IoT policy pins ``response/${ThingName}``, so the topic segment names the sender there."""
        monkeypatch.setattr(mesh_core, "select_backend", lambda: backend)
        mesh._on_presence(_presence("robot-b", zid=None))
        event = _register(mesh, "t6", BROADCAST_RESPONDER)

        mesh._on_response(_response("t6", "robot-b", zid=None))
        mesh._on_response(_response("t6", "robot-c", zid=None))

        assert [r["responder_id"] for r in mesh._responses["t6"]] == ["robot-b", "robot-c"]
        assert event.is_set()
        assert audits == []

    def test_a_reply_on_the_legacy_key_shape_is_refused(self, mesh: Mesh, audits: list) -> None:
        """``strands/<me>/response/<turn>`` names no responder, so the topic cannot vouch for the body."""
        mesh._on_presence(_presence("robot-b", zid=_ZID_B))
        _register(mesh, "t7", BROADCAST_RESPONDER)
        payload = {"type": "response", "turn_id": "t7", "responder_id": "robot-b", "result": {"status": "success"}}

        mesh._on_response(_sample(payload, zid=_ZID_B, key="strands/op/response/t7"))

        assert mesh._responses["t7"] == []
        assert [e for e, _ in audits] == ["response_hijack_rejected"]


class TestPointToPointTurn:
    def test_expected_responder_name_from_the_wrong_session_is_refused(self, mesh: Mesh, audits: list) -> None:
        mesh._on_presence(_presence("robot-b", zid=_ZID_B))
        event = _register(mesh, "t8", "robot-b")

        mesh._on_response(_response("t8", "robot-b", zid=_ZID_EVIL))

        assert mesh._responses["t8"] == []
        assert not event.is_set()
        assert [e for e, _ in audits] == ["response_hijack_rejected"]

    def test_expected_responder_from_its_own_session_is_recorded(self, mesh: Mesh, audits: list) -> None:
        mesh._on_presence(_presence("robot-b", zid=_ZID_B))
        event = _register(mesh, "t9", "robot-b")

        mesh._on_response(_response("t9", "robot-b", zid=_ZID_B))

        assert len(mesh._responses["t9"]) == 1
        assert event.is_set()
        assert audits == []


class TestEmergencyStopAccounting:
    def test_peers_that_never_acknowledged_are_named(
        self, mesh: Mesh, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        mesh._on_presence(_presence("arm-1", zid=_ZID_B))
        mesh._on_presence(_presence("arm-2", zid=_ZID_EVIL))
        mesh._running = True
        monkeypatch.setattr(
            mesh, "broadcast", lambda cmd, timeout=3.0: [{"responder_id": "arm-1", "result": {"status": "success"}}]
        )
        published: list[tuple[str, dict[str, Any]]] = []
        monkeypatch.setattr(mesh, "_publish_safety_envelope", lambda topic, env: published.append((topic, env)))
        events: list[dict[str, Any]] = []
        monkeypatch.setattr(mesh, "publish_safety_event", lambda **kw: events.append(kw))
        monkeypatch.setattr(mesh, "_local_session_zid", lambda: None)

        with caplog.at_level(logging.CRITICAL):
            mesh.emergency_stop()

        assert events[0]["payload"]["peers_silent"] == ["arm-2"]
        assert published[0][1]["peers_silent"] == ["arm-2"]
        msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.CRITICAL]
        assert any("arm-2" in m and "no acknowledgement" in m for m in msgs), msgs

    def _estop_against(self, mesh: Mesh, monkeypatch: pytest.MonkeyPatch, replies: list[Any]) -> dict[str, Any]:
        """Run ``emergency_stop`` with *replies* arriving on the real ``_on_response`` path."""
        mesh._running = True
        broadcast = mesh.broadcast

        def fanout(key: str, msg: dict[str, Any]) -> None:
            for make in replies:
                mesh._on_response(make(msg["turn_id"]))

        monkeypatch.setattr(mesh, "publish", fanout)
        monkeypatch.setattr(mesh, "broadcast", lambda cmd, timeout=3.0: broadcast(cmd, timeout=0.2))
        monkeypatch.setattr(mesh, "_publish_safety_envelope", lambda topic, env: None)
        events: list[dict[str, Any]] = []
        monkeypatch.setattr(mesh, "publish_safety_event", lambda **kw: events.append(kw))
        monkeypatch.setattr(mesh, "_local_session_zid", lambda: None)
        responses = mesh.emergency_stop()
        return {"responses": responses, **events[0]["payload"]}

    def test_a_forged_stop_ack_never_marks_an_unattributed_peer_as_stopped(
        self, monkeypatch: pytest.MonkeyPatch, audits: list
    ) -> None:
        """An admitted peer answers first in the victim's name; neither reply has a wire source."""
        operator = Mesh(None, peer_id="op", peer_type="operator")
        monkeypatch.setattr(operator, "_audit_local", lambda event, payload: audits.append((event, payload)))
        operator._on_presence(_presence("victim", zid=None))

        def forged(turn: str) -> Any:
            body = {"type": "response", "turn_id": turn, "responder_id": "victim", "result": {"ok": True}}
            return _sample(body, zid=None, key=f"strands/op/response/victim/{turn}")

        out = self._estop_against(operator, monkeypatch, [forged])

        assert out["responses"] == []
        assert out["peers_silent"] == ["victim"]
        assert out["peers_not_stopped"] == []

    def test_the_victims_own_reply_is_counted_after_a_forgery(
        self, monkeypatch: pytest.MonkeyPatch, audits: list
    ) -> None:
        """The honest reply carries its session zid, so the forgery cannot claim its slot."""
        operator = Mesh(None, peer_id="op", peer_type="operator")
        monkeypatch.setattr(operator, "_audit_local", lambda event, payload: audits.append((event, payload)))
        operator._on_presence(_presence("victim", zid=_ZID_B))

        def reply(zid: str | None, result: dict[str, Any]) -> Any:
            def make(turn: str) -> Any:
                body = {"type": "response", "turn_id": turn, "responder_id": "victim", "result": result}
                return _sample(body, zid=zid, key=f"strands/op/response/victim/{turn}")

            return make

        out = self._estop_against(
            operator, monkeypatch, [reply(None, {"ok": True}), reply(_ZID_B, {"ok": False, "error": "no stop_task"})]
        )

        assert [r["result"] for r in out["responses"]] == [{"ok": False, "error": "no stop_task"}]
        assert out["peers_not_stopped"] == ["victim"]
        assert out["peers_silent"] == []
        assert [e for e, _ in audits] == ["response_hijack_rejected"]


class TestPresenceAndRepliesCarryTheSessionZid:
    """What an honest peer puts on a real Zenoh wire is what the receiver can verify."""

    @pytest.fixture
    def wire(self, monkeypatch: pytest.MonkeyPatch) -> Any:
        zenoh = pytest.importorskip("zenoh")
        config = zenoh.Config()
        config.insert_json5("mode", '"peer"')
        config.insert_json5("scouting/multicast/enabled", "false")
        config.insert_json5("listen/endpoints", '["tcp/127.0.0.1:0"]')
        session = zenoh.open(config)
        monkeypatch.setattr(mesh_session, "_SESSION", session)
        seen: list[Any] = []
        arrived = threading.Event()

        def keep(sample: Any) -> None:
            seen.append(sample)
            arrived.set()

        sub = session.declare_subscriber("strands/**", keep)
        try:
            yield SimpleNamespace(session=session, seen=seen, arrived=arrived)
        finally:
            sub.undeclare()
            session.close()

    def test_presence_binds_the_announcing_session_on_the_receiver(self, wire: Any, mesh: Mesh) -> None:
        robot = Mesh(_FakeRobot(), peer_id="robot-b")
        robot._running = True
        loop = threading.Thread(target=robot._heartbeat_loop, daemon=True)
        loop.start()
        try:
            assert wire.arrived.wait(5.0), "no presence reached the wire"
        finally:
            robot._running = False
            robot._stop_event.set()
            loop.join(5.0)

        mesh._on_presence(wire.seen[0])

        assert mesh.peer_wire_zid("robot-b") == str(wire.session.info.zid())

    def test_a_command_reply_is_recorded_from_the_session_it_was_sent_on(self, wire: Any, mesh: Mesh) -> None:
        robot = Mesh(_FakeRobot(), peer_id="robot-b")
        mesh._bind_peer_wire_zid("robot-b", str(wire.session.info.zid()))
        event = _register(mesh, "f" * 32, BROADCAST_RESPONDER)
        key = f"strands/op/response/robot-b/{'f' * 32}"

        robot._reply("op", "f" * 32, key, {"type": "response", "turn_id": "f" * 32, "responder_id": "robot-b"}, None)
        assert wire.arrived.wait(5.0), "no reply reached the wire"
        mesh._on_response(wire.seen[0])

        assert [r["responder_id"] for r in mesh._responses["f" * 32]] == ["robot-b"]
        assert event.is_set()
